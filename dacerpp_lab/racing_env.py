"""DACER++ 레이싱 DirectRLEnv (각 환경에 2대: car_a=CVaR, car_b=Pow).

설계 요지
  - num_envs = 환경 수(기준 4096). 각 환경에 car_a, car_b 두 대 스폰.
    트랙은 competition_tracks 1024종을 env i -> track (i % 1024) 로 균형 배정
    (V자 512 + U자/ㄷ자 512 — 2026-08-13 끝단 노치 형상 확장, make_competition_tracks.py 참조).
  - 물리는 평면. 트랙은 중심선+가변 폭 프로파일(1~5m, 실 대회장 덕트 간격)로
    해석적 정의(이탈=종료). 이탈판정/보상/관측 정규화는 지역 half-width 기준.
  - 스캔은 좌/우 벽 폴리라인(덕트) 레이캐스트 + 상대차량 footprint 오버레이.
    실차의 Livox 3D LiDAR 는 높이밴드+방위각 섹터 최소거리(의사-2D 스캔)로
    동일 포맷을 만든다 (_scan docstring 참고).
  - 관측 도메인 랜덤화(RacingCfg.obs_noise): 스캔 노이즈/드롭아웃, 측위 노이즈,
    상대차량 검출 노이즈/드롭아웃. 보상/종료 계산에는 참값 사용.
  - 리셋 시 두 차량은 트랙 중심선 위의 랜덤 지점에 스폰된다
    (car_b 는 car_a 로부터 최소 spawn_min_gap_s 호길이만큼 떨어진 랜덤 지점 -> 스폰 충돌 방지).
  - 관측은 dict {"car_a","car_b"} 로 반환. 보상/종료는 차량별로 계산해 dual_info() 로 전달.
  - 에피소드는 트랙 단위: 둘 중 하나라도 종료(이탈/충돌)하거나 시간초과면 그 트랙 리셋.
  - 차량별 terminated/truncated 를 따로 산출 -> 각 DACER++ 에이전트가 자기 done 으로 학습.

DirectRLEnv.step() 호출 순서를 정확히 따른다:
    _pre_physics_step -> (decimation) _apply_action+sim -> _get_dones -> _get_rewards
    -> _reset_idx(reset envs) -> _get_observations
  => 보상/종료는 리셋 전 상태에서 _get_dones 에서 계산하고,
     관측은 리셋 후 상태에서 _get_observations 에서 새로 계산한다.
  => 부트스트랩용 next_obs(리셋 "이전" 상태의 관측)는 _get_dones 에서 계산해
     dual_info()["final_obs_*"] 로 전달한다. 리셋된 환경의 리셋 후 관측을
     next_obs 로 쓰면 가치 부트스트랩이 오염되기 때문이다.

주의(버전 의존): 조인트 제어/루트상태 쓰기 API 는 Isaac Lab 2.x 기준.
당신 USD 의 조인트 이름(car_cfg.py)과 패치 버전에 따라 소폭 수정이 필요할 수 있다.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
import isaaclab.sim as sim_utils

from .env_cfg import RacingEnvCfg
from .car_cfg import make_car_cfg, STEERING_JOINTS, DRIVE_JOINTS, FRONT_WHEEL_JOINTS
from .track_field import TrackField
from .vectorized_track import (VectorizedTracks, wrap_to_pi, overlay_opponent,
                               overlay_obstacles)


class RacingEnv(DirectRLEnv):
    cfg: RacingEnvCfg

    # ------------------------------------------------------------------ #
    def __init__(self, cfg: RacingEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        rc = self.cfg.racing
        N = self.num_envs
        dev = self.device

        # 트랙 자원 (가변 폭: 덕트 간격 1~5m 프로파일). _setup_scene 에서 이미
        # 생성했으면(벽 시각화용) 재사용.
        if not hasattr(self, "_field"):
            self._field = TrackField(self.cfg.track_field)
        self._vt = VectorizedTracks(self._field.centerlines, device=dev,
                                    half_widths=self._field.half_widths)
        self._vt.build_walls(rc.wall_offset)                             # 스캔용 덕트 벽
        self._track_type = torch.as_tensor(self._field.env_track_type, device=dev)
        self._total_s = self._vt.total_s_of(self._track_type)            # (N,)
        self._curv_offsets = tuple(rc.curv_lookahead)
        self._width_offsets = (0,) + self._curv_offsets                  # 현재 + 전방 폭
        self._hw_ref = 0.5 * self.cfg.track_field.params.width_max       # 폭 관측 정규화 기준

        # 진행도/이전상태 버퍼 (차량별)
        z = lambda: torch.zeros(N, device=dev)
        self._prev_s_a, self._prev_s_b = z(), z()
        self._travel_a, self._travel_b = z(), z()
        self._last_act_a = torch.zeros(N, 2, device=dev)   # 이번 스텝 '인가된' 행동 a_t
        self._last_act_b = torch.zeros(N, 2, device=dev)
        self._prev_act_a = torch.zeros(N, 2, device=dev)   # 직전 스텝 인가 행동 a_{t-1}
        self._prev_act_b = torch.zeros(N, 2, device=dev)

        # ---- 실차 루프 지연 모델 (env_cfg.act_delay_min/max 주석 참조) ----
        # 명령은 즉시가 아니라 차량별 _delay_* 스텝 뒤에 인가된다(링버퍼).
        self._delay_len = int(rc.act_delay_max) + 1
        self._act_buf_a = torch.zeros(self._delay_len, N, 2, device=dev)
        self._act_buf_b = torch.zeros(self._delay_len, N, 2, device=dev)
        self._act_ptr = 0
        self._ar_n = torch.arange(N, device=dev)
        self._delay_a = self._sample_delay(N)
        self._delay_b = self._sample_delay(N)
        # 최근 K개 '명령'(인가값 아님) 행동, 최신이 앞 — 지연 하 Markov 복원용
        self._cmd_a = torch.zeros(N, 2, device=dev)
        self._cmd_b = torch.zeros(N, 2, device=dev)
        self._hist_a = torch.zeros(N, 2 * rc.act_hist_len, device=dev)
        self._hist_b = torch.zeros(N, 2 * rc.act_hist_len, device=dev)
        self._prev_v_a, self._prev_v_b = z(), z()          # 직전 스텝 속력(가속 벌점용)
        # 횡속도 관측 이동평균 버퍼 (env_cfg.vy_obs_ma_steps — 실차 carstate_3d 의 MA(10) 모사)
        self._vy_ma = max(1, int(rc.vy_obs_ma_steps))
        self._vy_buf_a = torch.zeros(N, self._vy_ma, device=dev)
        self._vy_buf_b = torch.zeros(N, self._vy_ma, device=dev)
        self._vy_obs_a, self._vy_obs_b = z(), z()          # 스텝당 1회 갱신된 관측용 vy
        # 추월 이벤트 감지용: 직전 스텝의 s 간격(A-B, 랩 래핑) + 재지급 쿨다운
        self._prev_gap_ab = z()
        self._ot_cd_a = torch.zeros(N, dtype=torch.long, device=dev)
        self._ot_cd_b = torch.zeros(N, dtype=torch.long, device=dev)
        # 근접 페널티의 접근속도 게이트용 (초기값 크게 -> 첫 스텝 게이트 0)
        self._prev_car_dist = torch.full((N,), 100.0, device=dev)

        # dual_info 로 내보낼 차량별 보상/종료/최종관측
        self._rew_a, self._rew_b = z(), z()
        self._term_a = torch.zeros(N, dtype=torch.bool, device=dev)
        self._term_b = torch.zeros(N, dtype=torch.bool, device=dev)
        self._trunc = torch.zeros(N, dtype=torch.bool, device=dev)
        self._final_obs_a = torch.zeros(N, rc.obs_dim(), device=dev)
        self._final_obs_b = torch.zeros(N, rc.obs_dim(), device=dev)
        zb = lambda: torch.zeros(N, dtype=torch.bool, device=dev)
        self._causes = {k: zb() for k in ("off_a", "off_b", "crash_a", "crash_b",
                                          "flip_a", "flip_b", "spun_a", "spun_b",
                                          "obs_a", "obs_b")}

        # ---- 장애물 (env_cfg.obstacles_enabled 주석 참조) ----
        # footprint 다각형(로컬 좌표, CCW, Vmax=8 패딩: box4/tri3/cyl8, 미사용정점=마지막 반복)
        self._obs_kmax = int(rc.obstacle_max)
        self._obs_vmax = 8
        self._obs_verts = torch.zeros(N, self._obs_kmax, self._obs_vmax, 2, device=dev)
        self._obs_active = torch.zeros(N, self._obs_kmax, dtype=torch.bool, device=dev)
        # 시각화/진단용 메타 (test.py 프림 생성에 사용)
        self._obs_center = torch.zeros(N, self._obs_kmax, 2, device=dev)
        self._obs_rad = torch.zeros(N, self._obs_kmax, device=dev)       # 외접 반경
        self._obs_shape = torch.zeros(N, self._obs_kmax, dtype=torch.long, device=dev)  # 0 box/1 tri/2 cyl
        self._obs_size = torch.zeros(N, self._obs_kmax, device=dev)
        self._obs_yaw = torch.zeros(N, self._obs_kmax, device=dev)
        # 스폰 직후 장애물 충돌 면제(리스폰 즉시충돌 루프 방지) — 차량별 경과 스텝
        self._since_spawn_a = torch.zeros(N, dtype=torch.long, device=dev)
        self._since_spawn_b = torch.zeros(N, dtype=torch.long, device=dev)

        # 빔 각도(전방 중심 ±fov)
        self._beam_angles = torch.linspace(-rc.scan_fov, rc.scan_fov, rc.n_beams, device=dev)
        # 섹터 축약용 빔 내 오프셋: 각 빔의 섹터(폭 spacing)를 균등 분할한 S개 레이.
        # S=1 -> [0] (구 동작, 빔 각 그대로). S=3 -> [-spacing/2, 0, +spacing/2].
        # (env_cfg.scan_sector_rays 주석 참조 — 실차 의사 스캔의 '섹터 최소거리' 모사)
        spacing = 2.0 * rc.scan_fov / max(rc.n_beams - 1, 1)
        S = max(1, int(rc.scan_sector_rays))
        self._sector_offsets = (torch.zeros(1, device=dev) if S == 1 else
                                torch.linspace(-0.5 * spacing, 0.5 * spacing, S, device=dev))

        # 속도 커리큘럼: 명령 속도 상한(행동 -> 속도 매핑의 상단).
        # train.py 가 매 iteration set_curriculum_progress(it) 로 갱신하고,
        # 호출하지 않는 경로(test.py 등)는 기본값 v_max(커리큘럼 없음)로 동작한다.
        self._v_cmd_max = float(rc.v_max)

        # 레이캐스트는 (N,B,S) 원소별 연산 체인이라 torch.compile 융합 이득이 큼 (~5x)
        self._raycast_fn = self._vt.raycast
        if hasattr(torch, "compile"):
            try:
                self._raycast_fn = torch.compile(self._vt.raycast, dynamic=False)
            except Exception:
                pass

        # 조인트 인덱스 해석
        self._steer_ids_a, _ = self.car_a.find_joints(STEERING_JOINTS)
        self._drive_ids_a, _ = self.car_a.find_joints(DRIVE_JOINTS)
        self._steer_ids_b, _ = self.car_b.find_joints(STEERING_JOINTS)
        self._drive_ids_b, _ = self.car_b.find_joints(DRIVE_JOINTS)

        # 해석 타이어 힘 인가 대상 바디 (섀시) + 전륜 비주얼 회전용 조인트
        self._base_id_a, _ = self.car_a.find_bodies(["base_link"])
        self._base_id_b, _ = self.car_b.find_bodies(["base_link"])
        self._front_ids_a, _ = self.car_a.find_joints(FRONT_WHEEL_JOINTS)
        self._front_ids_b, _ = self.car_b.find_joints(FRONT_WHEEL_JOINTS)

        # Car A 속도 핸디캡 (스폰마다 재추첨 — env_cfg.v_cap_a_range 주석 참조)
        vr = rc.v_cap_a_range
        self._v_cap_a = (None if vr is None else
                         torch.empty(N, device=dev).uniform_(float(vr[0]), float(vr[1])))

        # k_slip 커리큘럼 현재값 (train.py 미호출 경로는 최종값 사용)
        self._k_slip_cur = float(rc.k_slip)

        # 지면 마찰 도메인 랜덤화: env 별 지면 속성(두 차 공유), 스폰마다 재추첨.
        # (env_cfg.TireModelCfg.mu_range 주석 참조)
        mr = self.cfg.tire.mu_range
        self._mu = torch.full((N,), float(self.cfg.tire.mu), device=dev)
        if mr is not None:
            self._mu.uniform_(float(mr[0]), float(mr[1]))

    # ------------------------------------------------------------------ #
    def _setup_scene(self):
        pd = self.cfg.project_dir
        # Direct 워크플로: {ENV_REGEX_NS} 토큰은 치환되지 않으므로 리터럴 정규식 경로를 직접 사용.
        # (scene 이 /World/envs/env_0, env_1, ... 로 복제 -> env_.* 가 모든 환경에 매칭)
        self.car_a = Articulation(make_car_cfg("/World/envs/env_.*/Car_A", pd))
        self.car_b = Articulation(make_car_cfg("/World/envs/env_.*/Car_B", pd))

        # 평면 지형(전역). env_origins 를 잡아주므로 clone 전에 생성.
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # 표준 순서: 생성 -> 지형 -> clone -> (scene 등록) -> 조명
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["car_a"] = self.car_a
        self.scene.articulations["car_b"] = self.car_b

        if self.cfg.wall_visuals:
            self._spawn_wall_visuals()
            if self.cfg.racing.obstacles_enabled:
                self._spawn_obstacle_visuals()
            self._spawn_det_box_visuals()

        light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
        light.func("/World/Light", light)

    def _spawn_wall_visuals(self):
        """디버깅용 덕트 벽 시각화 (렌더 전용, 물리 없음).

        트랙이 env 마다 달라 clone 을 못 쓰고 env 별로 USD Mesh 를 정의한다
        -> num_envs 에 비례해 느려지므로 GUI 확인용(작은 num_envs)에만 켤 것.
        """
        import omni.usd
        from pxr import UsdGeom

        if not hasattr(self, "_field"):
            self._field = TrackField(self.cfg.track_field)
        stage = omni.usd.get_context().get_stage()
        h = float(self.cfg.track_field.params.wall_height)

        # 트랙 타입별 벽 지오메트리(로컬 좌표) 1회 생성
        geoms = []
        for g in range(len(self._field.centerlines)):
            cl = self._field.centerlines[g]
            hw = self._field.half_widths[g]
            psi = self._field.features[g]["psi"]
            nrm = np.stack([-np.sin(psi), np.cos(psi)], axis=1)
            pts, counts, indices = [], [], []
            for side in (+1.0, -1.0):
                b = cl + side * nrm * hw[:, None]
                n = len(b)
                base = len(pts)
                pts += [(float(x), float(y), 0.0) for x, y in b]
                pts += [(float(x), float(y), h) for x, y in b]
                for i in range(n):
                    j = (i + 1) % n
                    indices += [base + i, base + j, base + n + j, base + n + i]
                    counts.append(4)
            geoms.append((pts, counts, indices))

        for i in range(self.scene.cfg.num_envs):
            pts, counts, indices = geoms[int(self._field.env_track_type[i])]
            mesh = UsdGeom.Mesh.Define(stage, f"/World/envs/env_{i}/WallVisual")
            mesh.CreatePointsAttr(pts)
            mesh.CreateFaceVertexCountsAttr(counts)
            mesh.CreateFaceVertexIndicesAttr(indices)
            mesh.CreateDisplayColorAttr([(0.85, 0.4, 0.1)])
            mesh.CreateDoubleSidedAttr(True)

    def _spawn_det_box_visuals(self):
        """감지 박스 시각 프림(렌더 전용) — 각 차량 뒷부분(10~30cm 높이)에 12×12cm 직육면체.
        차량 root 아래 자식이라 차와 함께 움직인다(스캔 검출은 해석적, 이건 GUI 확인용)."""
        try:
            import omni.usd
            from pxr import UsdGeom, Gf
            stage = omni.usd.get_context().get_stage()
            rc = self.cfg.racing
            zc = 0.5 * (rc.det_box_z[0] + rc.det_box_z[1])      # 지면 기준 박스 중심 높이
            hgt = rc.det_box_z[1] - rc.det_box_z[0]             # 박스 높이(10~30cm=0.20)
            for i in range(self.scene.cfg.num_envs):
                for car in ("A", "B"):
                    p = f"/World/envs/env_{i}/Car_{car}/DetBox"
                    cube = UsdGeom.Cube.Define(stage, p)
                    cube.CreateSizeAttr(1.0)                    # 단위 큐브(±0.5) -> scale 로 치수
                    xf = UsdGeom.Xformable(cube)
                    xf.AddTranslateOp().Set(Gf.Vec3d(-rc.det_box_rear, 0.0, zc))  # 뒷부분(-x)
                    xf.AddScaleOp().Set(Gf.Vec3f(rc.det_box_size, rc.det_box_size, hgt))
                    cube.CreateDisplayColorAttr([(0.1, 0.6, 0.95)])
        except Exception as e:
            print(f"[RacingEnv] det_box 시각 프림 생성 실패(비치명, GUI 전용): {e}")

    def _spawn_obstacle_visuals(self):
        """장애물 시각화 프림(렌더 전용) — env×슬롯마다 빈 Mesh 를 만들어 숨겨 두고,
        리셋 때 _update_obstacle_visuals 가 footprint 를 압출해 점/가시성을 갱신한다.
        (장애물은 리셋마다 바뀌므로 정적 벽과 달리 동적 갱신이 필요.)"""
        import omni.usd
        from pxr import UsdGeom
        stage = omni.usd.get_context().get_stage()
        self._obs_vis = []
        kmax = int(self.cfg.racing.obstacle_max)   # _setup_scene 은 텐서 할당 전이라 cfg 사용
        for i in range(self.scene.cfg.num_envs):
            row = []
            for k in range(kmax):
                mesh = UsdGeom.Mesh.Define(stage, f"/World/envs/env_{i}/Obstacle_{k}")
                mesh.CreateDisplayColorAttr([(0.95, 0.75, 0.1)])
                mesh.CreateDoubleSidedAttr(True)
                UsdGeom.Imageable(mesh).MakeInvisible()
                row.append(mesh)
            self._obs_vis.append(row)

    def _update_obstacle_visuals(self, env_ids):
        """env_ids 의 장애물 프림 점/가시성 갱신 (footprint 를 height 로 압출)."""
        if not getattr(self, "_obs_vis", None) or not hasattr(self, "_obs_verts"):
            return
        from pxr import UsdGeom
        verts = self._obs_verts.detach().cpu().numpy()      # (N,K,8,2) 로컬
        active = self._obs_active.detach().cpu().numpy()
        shape = self._obs_shape.detach().cpu().numpy()
        size = self._obs_size.detach().cpu().numpy()
        nvmap = {0: 4, 1: 3, 2: 8}
        for i in [int(x) for x in env_ids.tolist()]:
            for k in range(self._obs_kmax):
                mesh = self._obs_vis[i][k]
                img = UsdGeom.Imageable(mesh)
                if not active[i, k]:
                    img.MakeInvisible()
                    continue
                nv = nvmap[int(shape[i, k])]
                h = float(min(max(size[i, k], 0.20), 0.50))     # 시각 높이(빔 12.2cm 위)
                fp = verts[i, k, :nv]
                pts = [(float(x), float(y), 0.0) for x, y in fp] + \
                      [(float(x), float(y), h) for x, y in fp]
                counts = [nv, nv]                               # 바닥, 천장
                idx = list(range(nv)) + list(range(2 * nv - 1, nv - 1, -1))
                for j in range(nv):                            # 옆면 quad
                    a, b = j, (j + 1) % nv
                    idx += [a, b, b + nv, a + nv]
                    counts.append(4)
                mesh.CreatePointsAttr(pts)
                mesh.CreateFaceVertexCountsAttr(counts)
                mesh.CreateFaceVertexIndicesAttr(idx)
                img.MakeVisible()

    # ------------------------------------------------------------------ #
    # 행동
    # ------------------------------------------------------------------ #
    def _sample_delay(self, n: int) -> torch.Tensor:
        """차량별 루프 지연(제어스텝) 추첨. high 는 배타적이라 +1."""
        rc = self.cfg.racing
        return torch.randint(int(rc.act_delay_min), int(rc.act_delay_max) + 1,
                             (n,), device=self.device)

    def _pre_physics_step(self, actions: torch.Tensor):
        # actions:(N,4) = [car_a(steer,thr), car_b(steer,thr)] — 정책이 낸 '명령'
        a = actions.to(self.device).clamp(-1.0, 1.0)
        self._cmd_a, self._cmd_b = a[:, 0:2], a[:, 2:4]
        # 루프 지연: 지금 명령은 차량별 _delay_* 스텝 뒤에 인가된다.
        # (delay=0 이면 방금 쓴 값을 그대로 읽어 기존 동작과 동일)
        self._act_buf_a[self._act_ptr] = self._cmd_a
        self._act_buf_b[self._act_ptr] = self._cmd_b
        ra = (self._act_ptr - self._delay_a) % self._delay_len
        rb = (self._act_ptr - self._delay_b) % self._delay_len
        self._last_act_a = self._act_buf_a[ra, self._ar_n]
        self._last_act_b = self._act_buf_b[rb, self._ar_n]
        self._act_ptr = (self._act_ptr + 1) % self._delay_len

    def _cap_a(self):
        """Car A 의 유효 명령속도 상한 = min(핸디캡, 커리큘럼 상한). 스칼라 또는 (N,)."""
        if self._v_cap_a is None:
            return self._v_cmd_max
        return self._v_cap_a.clamp(max=self._v_cmd_max)

    def _apply_action(self):
        # 조인트 명령: 조향(실제 액추에이터 동역학 -> 타이어 모델의 조향각 입력이 됨)
        # + 바퀴 회전(비주얼). 지면 마찰이 0 이므로 바퀴 접촉은 수직 지지 전용.
        cap_a = self._cap_a()
        self._drive_one(self.car_a, self._last_act_a, self._steer_ids_a, self._drive_ids_a, cap_a)
        self._drive_one(self.car_b, self._last_act_b, self._steer_ids_b, self._drive_ids_b,
                        self._v_cmd_max)
        # 전륜 비주얼 회전: 현재 차속에 맞는 롤링 속도 (물리 영향 없음, 지면 μ=0)
        for car, fids in ((self.car_a, self._front_ids_a), (self.car_b, self._front_ids_b)):
            v = torch.linalg.norm(car.data.root_lin_vel_w[:, :2], dim=1)
            w = (v / max(self.cfg.racing.wheel_radius, 1e-3)).unsqueeze(1).expand(-1, len(fids))
            car.set_joint_velocity_target(w, joint_ids=fids)
        # 해석 타이어 힘: 물리 서브스텝(120Hz)마다 최신 상태로 재계산해 인가
        self._apply_tire_forces(self.car_a, self._last_act_a, self._steer_ids_a,
                                self._base_id_a, cap_a)
        self._apply_tire_forces(self.car_b, self._last_act_b, self._steer_ids_b,
                                self._base_id_b, self._v_cmd_max)

    def _apply_tire_forces(self, car: Articulation, act: torch.Tensor, steer_ids, base_id,
                           v_cmd_max):
        """동적 자전거 모델 타이어 힘 -> 섀시 외력 (바디 로컬 프레임 인가).

        - 슬립각 -> 축별 횡력: F_y = -mu*N*tanh(alpha/alpha_char) (포화 슬립 곡선)
        - 종방향: ESC 속도서보 F_x = k*(v_cmd - vx), 모터 상한 + 뒤축(RWD)
          마찰서클 제한. 종하중이동(가감속 시 앞/뒤 하중 재분배) 반영.
        - 조향각은 명령이 아니라 '실제 조향 조인트 각'(액추에이터 지연 포함).
        (근거/파라미터 식별 방법: env_cfg.TireModelCfg docstring)
        """
        tc = self.cfg.tire
        rc = self.cfg.racing
        g = 9.81
        L = tc.lf + tc.lr

        # 차체 좌표 속도 (yaw 만 사용 — 평면 주행; 전복은 termination 이 처리)
        quat = car.data.root_quat_w
        w, x, y, zq = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq))
        vw = car.data.root_lin_vel_w
        cosy, siny = torch.cos(yaw), torch.sin(yaw)
        vx = vw[:, 0] * cosy + vw[:, 1] * siny
        vy = -vw[:, 0] * siny + vw[:, 1] * cosy
        r = car.data.root_ang_vel_w[:, 2]
        delta = car.data.joint_pos[:, steer_ids].mean(dim=1)

        # 종방향 요구력 (속도서보 + 모터 상한)
        v_cmd = rc.v_min + (act[:, 1] + 1.0) * 0.5 * (v_cmd_max - rc.v_min)
        fx_des = (tc.k_drive * (v_cmd - vx)).clamp(-tc.f_drive_max, tc.f_drive_max)

        # 종하중이동 (요구 가속 기반 준정적 근사)
        ax_est = fx_des / tc.mass
        nf = (tc.mass * g * tc.lr / L - tc.mass * ax_est * tc.com_height / L).clamp(min=0.0)
        nr = (tc.mass * g * tc.lf / L + tc.mass * ax_est * tc.com_height / L).clamp(min=0.0)

        # 슬립각 -> 횡력 (저속 테이퍼: 정지 시 조향만으로 횡력 생기는 비물리 방지)
        # μ 는 env 별 지면 랜덤화 값(self._mu) — env_cfg.mu_range 주석 참조
        vx_eff = vx.abs().clamp(min=0.5)
        taper = torch.tanh(vx.abs() / max(tc.v_lat_taper, 1e-3))
        alpha_f = torch.atan2(vy + tc.lf * r, vx_eff) - delta
        alpha_r = torch.atan2(vy - tc.lr * r, vx_eff)
        fyf = -self._mu * nf * torch.tanh(alpha_f / tc.alpha_char) * taper
        fyr = -self._mu * nr * torch.tanh(alpha_r / tc.alpha_char) * taper

        # 뒤축 마찰서클 (RWD: 구동/제동이 뒤축 횡그립을 잠식)
        cap_r = self._mu * nr
        norm_r = torch.sqrt(fx_des ** 2 + fyr ** 2).clamp(min=1e-6)
        scale_r = (cap_r / norm_r).clamp(max=1.0)
        fx_r = fx_des * scale_r
        fyr = fyr * scale_r

        # 구름저항 + 합력/요모멘트 (앞축 힘은 조향각만큼 차체로 회전)
        f_roll = -tc.c_roll * tc.mass * g * torch.tanh(vx / 0.2)
        cosd, sind = torch.cos(delta), torch.sin(delta)
        # 전복 게이트: 뒤집힌 차(중력 z_b > -0.5)에는 타이어 힘 인가 중단
        # (바퀴가 지면에 없으므로 물리적으로도 0; 전복 후 폭주/텔레포트 방지)
        up = (car.data.projected_gravity_b[:, 2] < -0.5).float()
        fx_tot = (fx_r + f_roll - fyf * sind) * up
        fy_tot = (fyr + fyf * cosd) * up
        mz = (tc.lf * fyf * cosd - tc.lr * fyr) * up

        n = fx_tot.shape[0]
        forces = torch.zeros(n, 1, 3, device=self.device)
        forces[:, 0, 0] = fx_tot
        forces[:, 0, 1] = fy_tot
        torques = torch.zeros(n, 1, 3, device=self.device)
        torques[:, 0, 2] = mz
        # set_external_force_and_torque 는 deprecated (IsaacLab 0.54+) — 내부적으로
        # 그대로 아래 composer 를 호출하므로 동작은 동일하고 경고만 사라진다.
        car.permanent_wrench_composer.set_forces_and_torques(
            forces=forces, torques=torques, body_ids=base_id)

    def set_curriculum_progress(self, it: int):
        """속도 커리큘럼 램프: 명령 속도 상한을 v0 -> v_max 로 선형 증가.

        관측 정규화(speed/v_max)는 그대로라 관측 분포는 비정상성이 없고,
        행동->속도 매핑의 상단만 서서히 열린다. resume 시에는 start_it 부터
        호출되므로 자동으로 이어진다."""
        rc = self.cfg.racing
        # k_slip 램프 (env_cfg.k_slip_v0 주석 참조 — 초기 저속 함정 방지)
        if rc.k_slip_ramp_steps > 0:
            f = min(max(it / rc.k_slip_ramp_steps, 0.0), 1.0)
            self._k_slip_cur = float(rc.k_slip_v0 + f * (rc.k_slip - rc.k_slip_v0))
        else:
            self._k_slip_cur = float(rc.k_slip)
        if rc.curriculum_steps <= 0:
            self._v_cmd_max = float(rc.v_max)
            return
        frac = min(max(it / rc.curriculum_steps, 0.0), 1.0)
        self._v_cmd_max = float(rc.curriculum_v0 + frac * (rc.v_max - rc.curriculum_v0))

    def _drive_one(self, car: Articulation, act: torch.Tensor, steer_ids, drive_ids, v_cmd_max):
        rc = self.cfg.racing
        steer = act[:, 0] * rc.max_steering_angle                       # (N,)
        speed = rc.v_min + (act[:, 1] + 1.0) * 0.5 * (v_cmd_max - rc.v_min)
        wheel_w = speed / max(rc.wheel_radius, 1e-3)                     # rad/s
        steer_t = steer.unsqueeze(1).expand(-1, len(steer_ids))
        drive_t = wheel_w.unsqueeze(1).expand(-1, len(drive_ids))
        car.set_joint_position_target(steer_t, joint_ids=steer_ids)
        car.set_joint_velocity_target(drive_t, joint_ids=drive_ids)

    # ------------------------------------------------------------------ #
    # 상태 추출
    # ------------------------------------------------------------------ #
    def _car_state(self, car: Articulation):
        pos_w = car.data.root_pos_w                       # (N,3) world
        quat = car.data.root_quat_w                       # (N,4) wxyz
        vel = car.data.root_lin_vel_w                     # (N,3) world
        local_xy = pos_w[:, :2] - self.scene.env_origins[:, :2]
        w, x, y, zq = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        yaw = torch.atan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq))
        speed = torch.linalg.norm(vel[:, :2], dim=1)
        return local_xy, yaw, speed

    def _body_dyn(self, car: Articulation, yaw: torch.Tensor):
        """차체 좌표 종/횡속도 + 요레이트 (β 벌점 + 동역학 관측용)."""
        vel = car.data.root_lin_vel_w
        cosy, siny = torch.cos(yaw), torch.sin(yaw)
        vx = vel[:, 0] * cosy + vel[:, 1] * siny
        vy = -vel[:, 0] * siny + vel[:, 1] * cosy
        return vx, vy, car.data.root_ang_vel_w[:, 2]

    def _push_vy(self, buf: torch.Tensor, vy: torch.Tensor) -> torch.Tensor:
        """횡속도 이동평균 버퍼에 최신값을 넣고 평균을 반환 (in-place, 최신이 앞).

        실차 carstate_3d_node 는 포즈 차분 vy 를 10샘플(=100ms @EKF 100Hz) 이동평균
        해서 /car_state/odom 에 싣는다. 30Hz 제어에서 같은 시간창은 3스텝이다.
        평균이므로 지연뿐 아니라 고주파 성분 감쇠까지 함께 재현된다.
        """
        if self._vy_ma <= 1:
            return vy
        buf[:, 1:] = buf[:, :-1].clone()
        buf[:, 0] = vy
        return buf.mean(dim=1)

    def _wrap_ds(self, ds):
        tot = self._total_s
        ds = torch.where(ds < -0.5 * tot, ds + tot, ds)
        ds = torch.where(ds > 0.5 * tot, ds - tot, ds)
        return ds

    # ------------------------------------------------------------------ #
    # 종료 + 보상 (리셋 전, 물리 직후 상태에서 계산)  ← step 순서상 가장 먼저
    # ------------------------------------------------------------------ #
    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        rc = self.cfg.racing
        la, ya, va = self._car_state(self.car_a)
        lb, yb, vb = self._car_state(self.car_b)
        pa = self._vt.project(la, self._track_type)
        pb = self._vt.project(lb, self._track_type)

        # 진행도
        ds_a = self._wrap_ds(pa["s"] - self._prev_s_a)
        ds_b = self._wrap_ds(pb["s"] - self._prev_s_b)
        self._travel_a = self._travel_a + ds_a
        self._travel_b = self._travel_b + ds_b
        self._prev_s_a = pa["s"]
        self._prev_s_b = pb["s"]
        lap_a = self._travel_a >= self._total_s
        lap_b = self._travel_b >= self._total_s
        # 랩 완주 시 초과분(나머지)은 다음 랩 진행도로 이월
        self._travel_a = torch.where(lap_a, self._travel_a - self._total_s, self._travel_a)
        self._travel_b = torch.where(lap_b, self._travel_b - self._total_s, self._travel_b)

        # 이탈/충돌 (이탈 판정은 지역 half-width 기준 — 가변 폭 트랙)
        hw_a = self._vt.lookahead_width(pa["idx"], self._track_type, (0,)).squeeze(1)
        hw_b = self._vt.lookahead_width(pb["idx"], self._track_type, (0,)).squeeze(1)
        off_a = pa["lateral"].abs() > (hw_a + rc.offtrack_margin)
        off_b = pb["lateral"].abs() > (hw_b + rc.offtrack_margin)
        car_dist = torch.linalg.norm(la - lb, dim=1)
        crash = car_dist < rc.car_collision_dist
        # 차간 충돌 책임 분리: 추돌(뒤차) 책임 원칙 — 진행도 s 기준 뒤차에만
        # 종료+페널티. 앞차(무과실)는 term 이 아니므로 done=0 으로 남아
        # final_obs 부트스트랩이 유지된다(train.py 는 차량별 term 만 done 으로 사용).
        # 환경 리셋은 뒤차의 term 만으로도 발동하므로 기존과 동일하게 둘 다 리셋.
        gap_ab = self._wrap_ds(pa["s"] - pb["s"])    # >0 이면 A 가 앞
        crash_a = crash & (gap_ab <= 0)              # A 가 뒤(동률 포함) -> A 책임
        crash_b = crash & (gap_ab >= 0)              # B 가 뒤(동률 포함) -> B 책임
        # 전복/공중부양 종료: 기울기 60° 초과(중력 z 성분 > -0.5) 또는 섀시가
        # 정상 주행 높이(축고 0.05m)를 크게 벗어난 경우. 탄도비행 exploit 차단.
        flip_a = (self.car_a.data.projected_gravity_b[:, 2] > -0.5) | \
                 (self.car_a.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2] > 0.25)
        flip_b = (self.car_b.data.projected_gravity_b[:, 2] > -0.5) | \
                 (self.car_b.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2] > 0.25)
        # 스핀 조기 종료: |헤딩오차| > 100° = 회복 불능 (env_cfg.spin_herr_limit 주석 참조)
        spun_a = wrap_to_pi(ya - pa["psi"]).abs() > rc.spin_herr_limit
        spun_b = wrap_to_pi(yb - pb["psi"]).abs() > rc.spin_herr_limit
        # 장애물 충돌 종료: 차량 등가반경 이내로 footprint 접촉. 스폰 직후 grace 스텝
        # 동안은 면제(리스폰이 우연히 장애물 근처에 떨어져도 즉시-종료 루프 방지).
        self._since_spawn_a += 1
        self._since_spawn_b += 1
        obs_a = self._obstacle_hit(la) & (self._since_spawn_a > rc.obstacle_grace_steps)
        obs_b = self._obstacle_hit(lb) & (self._since_spawn_b > rc.obstacle_grace_steps)
        self._term_a = off_a | crash_a | flip_a | spun_a | obs_a
        self._term_b = off_b | crash_b | flip_b | spun_b | obs_b
        # 진단용 종료 원인 분해 (dual_info -> analyze_issues.py 등)
        self._causes = dict(off_a=off_a, off_b=off_b, crash_a=crash_a, crash_b=crash_b,
                            flip_a=flip_a, flip_b=flip_b, spun_a=spun_a, spun_b=spun_b,
                            obs_a=obs_a, obs_b=obs_b)

        # 차체 동역학: 사이드슬립 β(스핀 전조 벌점) + 요레이트/횡속도(관측)
        vx_a, vy_a, r_a = self._body_dyn(self.car_a, ya)
        vx_b, vy_b, r_b = self._body_dyn(self.car_b, yb)
        beta_a = torch.atan2(vy_a.abs(), vx_a.abs().clamp(min=0.5))
        beta_b = torch.atan2(vy_b.abs(), vx_b.abs().clamp(min=0.5))
        # 횡속도 '관측'만 실차 추정기의 이동평균 지연을 통과시킨다 (env_cfg.vy_obs_ma_steps).
        # β 벌점/종료는 참값(vy_a/vy_b)을 계속 쓴다 — 보상은 물리를 봐야 하고,
        # 지연은 정책이 감당해야 할 관측 결함이기 때문. 스텝당 한 번만 갱신하고
        # (_get_dones 는 스텝당 1회) _get_observations 는 그 결과를 읽기만 한다.
        self._vy_obs_a = self._push_vy(self._vy_buf_a, vy_a)
        self._vy_obs_b = self._push_vy(self._vy_buf_b, vy_b)

        # 실측 종가속도 (행동이 아니라 물리 속도 변화 기준; 급감속->급가속 반복 억제)
        dv_a = (va - self._prev_v_a) / self.step_dt
        dv_b = (vb - self._prev_v_b) / self.step_dt

        # ---- 경쟁 신호: 추월 '완료' 이벤트 + 추돌 위험 근접 페널티 ----
        # 추월 완료: 뒤(gap<0)에 있다가 앞(gap>=0)으로 교차. 교차 폭 검사(<2m)로
        # 랩 경계(±tot/2) 랩어라운드 점프를 이벤트로 오인하지 않게 한다.
        crossed_a = (self._prev_gap_ab < 0) & (gap_ab >= 0) & (gap_ab - self._prev_gap_ab < 2.0)
        crossed_b = (self._prev_gap_ab > 0) & (gap_ab <= 0) & (self._prev_gap_ab - gap_ab < 2.0)
        ot_a = crossed_a & (self._ot_cd_a <= 0)
        ot_b = crossed_b & (self._ot_cd_b <= 0)
        cd_full = torch.full_like(self._ot_cd_a, int(rc.overtake_cooldown))
        self._ot_cd_a = torch.where(ot_a, cd_full, (self._ot_cd_a - 1).clamp(min=0))
        self._ot_cd_b = torch.where(ot_b, cd_full, (self._ot_cd_b - 1).clamp(min=0))
        # 근접 페널티: 뒤차에만, 거리 [collision_dist, prox_dist] 선형 램프 ×
        # 접근속도 게이트(거리 유지 추종은 무벌점, 충돌 코스로 접근 중일 때만).
        # 스폰/리스폰 텔레포트는 간격이 3m 이상이라 closeness=0 으로 자동 차단된다.
        closeness = ((rc.prox_dist - car_dist)
                     / max(rc.prox_dist - rc.car_collision_dist, 1e-6)).clamp(0.0, 1.0)
        closing = (self._prev_car_dist - car_dist) / self.step_dt      # >0: 접근 중
        gate = (closing / max(rc.prox_close_ref, 1e-6)).clamp(0.0, 1.0)
        prox_pen = rc.k_proximity * closeness * gate
        prox_a = torch.where(gap_ab <= 0, prox_pen, torch.zeros_like(prox_pen))
        prox_b = torch.where(gap_ab >= 0, prox_pen, torch.zeros_like(prox_pen))

        # 보상
        def rew(ds, lateral, hw, last_act, prev_act, term, lap, ds_opp, dv, v, beta, ot, prox):
            r = rc.k_progress * ds
            r = r - rc.k_deviation * (lateral / hw) ** 2
            # 벽 근접 shaping: 종료 경계(|lat| = hw + margin) 대비 비율의 고차 램프.
            # 경계 '직전' 스텝들에 회피 그래디언트를 만든다 (env_cfg.k_wall 주석 참조).
            # 비율>1 은 같은 스텝에 term 처리되어 종말 벌점으로 대체되므로 1로 clamp.
            wall_ratio = (lateral.abs() / (hw + rc.offtrack_margin).clamp(min=0.1)).clamp(max=1.0)
            r = r - rc.k_wall * wall_ratio ** rc.wall_pow
            # 행동 스무딩(조향/스로틀 분리) + 조향 크기 벌점 — PWM/포화 조향 억제
            # (env_cfg.k_smooth_steer 주석 참조)
            dact = (last_act - prev_act).abs()
            r = r - rc.k_smooth_steer * dact[:, 0] - rc.k_smooth_thr * dact[:, 1]
            r = r - rc.k_steer_mag * last_act[:, 0].square()
            # 데드존형 가속 벌점: |a|<=데드존 무벌점, 초과분만 2차
            # (충돌/리스폰 스파이크 대비 상한 clamp). 감속측 데드존은 별도로 넓다 —
            # 코너 진입 제동은 무벌점, 브레이크체크급 급감속만 과세 (env_cfg 주석 참조).
            dz = torch.where(dv < 0.0, torch.full_like(dv, rc.decel_deadzone),
                             torch.full_like(dv, rc.accel_deadzone))
            excess = (dv.abs() - dz).clamp(min=0.0)
            r = r - rc.k_accel * (excess / max(rc.accel_ref - rc.accel_deadzone, 1e-6)) \
                .square().clamp(max=4.0)
            # 사이드슬립 데드존 벌점: 정상 코너링(데드존 이내) 무벌점, 슬라이드가
            # 커질수록 2차 — 스핀/드리프트 억제. 계수는 커리큘럼 램프(_k_slip_cur).
            slip_ex = (beta - rc.slip_deadzone).clamp(min=0.0)
            r = r - self._k_slip_cur * (slip_ex / max(rc.slip_deadzone, 1e-6)) \
                .square().clamp(max=rc.slip_clamp)
            r = r + rc.k_overtake * (ds - ds_opp)
            r = r + ot.float() * rc.r_overtake
            r = r - prox
            r = r + lap.float() * rc.r_lap
            # 종말 벌점(속도 의존): 감속이 종말 비용을 직접 줄여 '이탈 직전 감속'에
            # 그래디언트가 생긴다 (env_cfg.r_term_base 주석 참조).
            r = torch.where(term, -(rc.r_term_base + rc.k_term_speed * v * v), r)
            return r

        self._rew_a = rew(ds_a, pa["lateral"], hw_a, self._last_act_a, self._prev_act_a,
                          self._term_a, lap_a, ds_b, dv_a, va, beta_a, ot_a, prox_a)
        self._rew_b = rew(ds_b, pb["lateral"], hw_b, self._last_act_b, self._prev_act_b,
                          self._term_b, lap_b, ds_a, dv_b, vb, beta_b, ot_b, prox_b)

        # 행동/속도/간격 시프트: 다음 스텝 기준 현재값이 prev 가 된다 (리셋은 스폰 로직이 덮음)
        self._prev_act_a = self._last_act_a.clone()
        self._prev_act_b = self._last_act_b.clone()
        # 명령 행동 히스토리 롤(최신이 앞): 지연 창에서 '배송 중'인 명령을 관측에 노출해
        # 비-마르코프성을 제거한다 (env_cfg.act_hist_len 주석 참조).
        self._hist_a = torch.cat([self._cmd_a, self._hist_a[:, :-2]], dim=1)
        self._hist_b = torch.cat([self._cmd_b, self._hist_b[:, :-2]], dim=1)
        self._prev_v_a = va.clone()
        self._prev_v_b = vb.clone()
        self._prev_gap_ab = gap_ab.clone()
        self._prev_car_dist = car_dist.clone()

        # 부트스트랩용 최종 관측(리셋 전 상태). 히스토리는 방금 롤했으므로
        # 최신 항목이 a_t(이번 스텝 명령) -> s_{t+1} 관측과 시점이 일치한다.
        self._final_obs_a = self._observe_car(la, ya, va, self._vy_obs_a, r_a, pa, lb, yb, vb, pb, self._hist_a)
        self._final_obs_b = self._observe_car(lb, yb, vb, self._vy_obs_b, r_b, pb, la, ya, va, pa, self._hist_b)

        truncated = self.episode_length_buf >= self.max_episode_length - 1
        # 차량별 리셋: term 난 차만 개별 리스폰하고 상대는 계속 주행한다.
        # final_obs 계산 "이후"에 수행 — 터미널 관측은 리스폰 전 상태여야 한다.
        # env 단위 리셋은 시간초과(truncated)뿐 — 그건 _reset_idx 가 처리.
        self._respawn_terminated(pa, pb, truncated)
        self._trunc = truncated
        return torch.zeros_like(truncated), truncated

    def _get_rewards(self) -> torch.Tensor:
        # 프레임워크용 트랙 단위 스칼라(로깅). 학습은 dual_info() 의 차량별 보상을 사용.
        return self._rew_a + self._rew_b

    # ------------------------------------------------------------------ #
    # 관측 (리셋 후 상태에서 새로 계산)
    # ------------------------------------------------------------------ #
    def _observe_car(self, local_xy, yaw, speed, vy, yawrate, proj, opp_local_xy, opp_yaw,
                     opp_speed, opp_proj, act_hist):
        rc = self.cfg.racing
        # ---- 관측 기준점을 실차 base_link(뒤축)로 이동 (env_cfg.obs_ref_offset_x) ----
        # 인자로 받은 local_xy/proj 는 물리 루트(=URDF base_link=축거 중점) 기준이고
        # 보상/종료가 그것을 쓴다. 관측만 뒤축으로 옮겨 실차와 같은 점에서 만든다.
        # 재투영 비용은 레이캐스트의 ~2% 수준이라 무시 가능하다.
        ref_xy = local_xy
        if rc.obs_ref_offset_x != 0.0:
            fwd = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)      # (N,2) 차체 전방
            ref_xy = local_xy + rc.obs_ref_offset_x * fwd
            proj = self._vt.project(ref_xy, self._track_type)
        s, lateral, idx = proj["s"], proj["lateral"], proj["idx"]
        herr = wrap_to_pi(yaw - proj["psi"])
        curv = self._vt.lookahead_curvature(idx, self._track_type, self._curv_offsets)
        width = self._vt.lookahead_width(idx, self._track_type, self._width_offsets)  # (N,1+k)
        hw_local = width[:, 0]

        wall_scan, scan = self._scan(ref_xy, yaw, idx, opp_local_xy, opp_yaw)

        # 차체 기준 상대차량 위치 (기준점 = 관측 기준점 = 뒤축, 실차 rl_controller 와 동일:
        # 실차는 자차 pose(base_link)에서 상대차 '차중심'까지의 상대좌표를 만든다)
        rel = opp_local_xy - ref_xy
        cosy, siny = torch.cos(-yaw), torch.sin(-yaw)
        rel_x = rel[:, 0] * cosy - rel[:, 1] * siny
        rel_y = rel[:, 0] * siny + rel[:, 1] * cosy

        # ---- 실차 LiDAR 검출 모델: 상대차의 '감지 박스'(뒷부분 12×12cm)만 잡힌다 ----
        # (1) 거리 초과 (2) 시야각(±scan_fov) 밖 (3) 벽 가림(occlusion) 이면 미검출 ->
        # 상대차 특징 5개 전부 0. 판정 기준점은 차중심이 아니라 '박스 중심'(뒤로
        # det_box_rear, 상대차 헤딩 기준). 참값(노이즈 전) 기준 — 검출은 물리 현상.
        opp_rel_yaw = wrap_to_pi(opp_yaw - yaw)
        bcx = rel_x - rc.det_box_rear * torch.cos(opp_rel_yaw)   # 박스 중심(바디프레임)
        bcy = rel_y - rc.det_box_rear * torch.sin(opp_rel_yaw)
        rng = torch.sqrt(bcx * bcx + bcy * bcy)
        brg = torch.atan2(bcy, bcx)
        spacing = 2.0 * rc.scan_fov / max(rc.n_beams - 1, 1)
        bi = ((brg + rc.scan_fov) / spacing).round().long().clamp(0, rc.n_beams - 1)
        wall_at = wall_scan.gather(1, bi.unsqueeze(1)).squeeze(1)
        box_circ = rc.det_box_size * 0.70711                    # 정사각 외접반경
        visible = ((rng < rc.scan_max_range) & (brg.abs() <= rc.scan_fov)
                   & (wall_at >= rng - box_circ))

        # ---- 도메인 랜덤화 (관측에만 적용; 보상/종료는 참값 사용) ----
        # 스캔은 실차에서 측위와 무관한 직접 측정이므로 자체 노이즈만,
        # lateral/heading 은 측위 결과이므로 측위 노이즈를 모사한다.
        lat_o, herr_o = lateral, herr
        vy_o, r_o = vy, yawrate
        if rc.obs_noise:
            scan = scan + torch.randn_like(scan) * rc.scan_noise_std
            if rc.scan_spike_p > 0:     # 빔별 큰 오차(실측 MAE 꼬리 재현): 정책이 단일 빔 과신 안 하도록
                spike = torch.rand_like(scan) < rc.scan_spike_p
                scan = scan + spike.float() * torch.randn_like(scan) * rc.scan_spike_std
            if rc.scan_dropout_p > 0:
                drop = torch.rand_like(scan) < rc.scan_dropout_p
                scan = torch.where(drop, torch.full_like(scan, rc.scan_max_range), scan)
            scan = scan.clamp(0.0, rc.scan_max_range)
            lat_o = lateral + torch.randn_like(lateral) * rc.pose_noise_lateral
            herr_o = herr + torch.randn_like(herr) * rc.pose_noise_heading
            vy_o = vy + torch.randn_like(vy) * rc.vy_noise_std
            r_o = yawrate + torch.randn_like(yawrate) * rc.gyro_noise_std
            if rc.opp_pos_noise > 0:
                rel_x = rel_x + torch.randn_like(rel_x) * rc.opp_pos_noise
                rel_y = rel_y + torch.randn_like(rel_y) * rc.opp_pos_noise

        scan_norm = (scan / rc.scan_max_range).clamp(0, 1)
        rel_speed = speed - opp_speed
        gap_s = self._wrap_ds(opp_proj["s"] - s)
        opp = torch.stack([
            (rel_x / rc.scan_max_range).clamp(-1, 1),
            (rel_y / rc.scan_max_range).clamp(-1, 1),
            (rel_speed / max(rc.v_max, 1e-6)).clamp(-1, 1),
            (gap_s / 10.0).clamp(-1, 1),
            visible.float(),
        ], dim=1)
        # 미검출(거리/시야각/벽 가림) -> 전부 0
        opp = opp * visible.float().unsqueeze(1)
        if rc.obs_noise and rc.opp_dropout_p > 0:
            # 추가 확률적 검출 실패 모사 (센서/알고리즘 미스)
            keep = (torch.rand(opp.shape[0], device=self.device) >= rc.opp_dropout_p)
            opp = opp * keep.float().unsqueeze(1)

        obs = torch.cat([
            scan_norm,                                              # n_beams
            (speed / max(rc.v_max, 1e-6)).unsqueeze(1),             # 1
            torch.stack([torch.sin(herr_o), torch.cos(herr_o)], 1),  # 2
            (lat_o / hw_local).clamp(-2, 2).unsqueeze(1),           # 1  (±1 = 벽)
            # 곡률 클립: ±1 -> ±2(2026-07-30) -> ±3(2026-08-12, env_cfg.curv_clip).
            # 클립이 실제 코스의 |κ|max 보다 작으면 정책은 '급코너'와 '헤어핀'을
            # 관측상 구분하지 못한다 — 신규 연습 맵(|κ|max 2.78/2.43)에서 ±2 가
            # 정확히 그 상태였고, 실차가 그 코너에서 덜 꺾어 벽에 박았다.
            # (근거/측정은 env_cfg.RacingCfg.curv_clip 주석)
            curv.clamp(-rc.curv_clip, rc.curv_clip),                # k
            (width / self._hw_ref).clamp(0, 1),                     # 1+k (현재/전방 폭)
            act_hist.clamp(-1, 1),                                  # 2*act_hist_len
            opp,                                                    # 5
            # 차체 동역학 (요 감쇠/슬라이드 감지용 — 실차 IMU 요레이트 + 횡속도 추정)
            torch.stack([(r_o / max(rc.yawrate_norm, 1e-6)).clamp(-1, 1),
                         (vy_o / max(rc.vy_norm, 1e-6)).clamp(-1, 1)], 1),  # 2
        ], dim=1).float()
        return obs

    def _scan(self, local_xy, yaw, idx0, opp_local_xy, opp_yaw):
        """벽 레이캐스트 스캔 + 상대차 감지 박스 + 장애물 footprint 오버레이.

        반환: (wall_scan, scan) — wall_scan 은 오버레이 전 벽 전용 스캔으로,
        상대차 가시성(벽 가림) 판정에 사용한다.

        실차(F1TENTH, 원형 덕트 벽 + Livox 3D LiDAR) 대응 파이프라인:
          deskew -> 지면 위 높이 밴드(예: 0.05~덕트 높이) 필터 ->
          차체 기준 ±scan_fov 를 n_beams 개 방위각 섹터로 분할 ->
          섹터별 최소 거리 -> scan_max_range 클립 -> /scan_max_range 정규화.

        ★2026-08-12 '섹터별 최소 거리'를 실제로 모사하도록 수정. 구 코드는 빔 각으로
          레이를 하나만 쏴서, 실차의 섹터 축약(8.71° 폭 안 점들의 min)보다 구조적으로
          멀게 읽었다 — 실제 맵에서 측정한 편향 평균 +0.17m, 빔의 52%가 학습 노이즈
          (0.035m)보다 큰 오차. 빔당 scan_sector_rays 개 레이의 min 으로 바꾸니
          편향 -0.003m, 오차>0.035m 인 빔 0.0% 로 사라진다(S=3 기준, S=5 는 개선 없음).

        ★스캔/측위 지연에 대한 확인(2026-08-12): 실차 EKF(ekf_localizer)는
          predict_frequency 100Hz + extend_state_step 80(=0.8s 버퍼)로 지연 관측을
          updateWithDelay 로 시각 정렬 융합하므로 출력 pose 는 '현재 시각' 추정치이고,
          rl_controller 의 scan_builder 는 점마다 측정 시각 pose 로 되돌린 뒤 현재
          차체 프레임으로 옮긴다(deskew). 따라서 Livox 100ms 스윕 누적은 '정지한 벽'에
          대해서는 기하학적으로 보상된다 -> act_delay_min/max=(0,2) 유지가 맞다.
          (실측 pose 지연 mean 142ms / p95 163ms 는 모두 0.8s 버퍼 안이다.)
        """
        rc = self.cfg.racing
        # 섹터 축약 모사: 빔마다 섹터(폭 2*fov/(B-1))를 가로지르는 S개 레이를 쏘고
        # 최소값을 취한다 — 실차 의사 스캔이 '섹터 안 점들의 최소 거리'이기 때문
        # (env_cfg.scan_sector_rays 주석: 편향 평균 -0.21m). S=1 이면 구 동작.
        # _sector_offsets 는 (S,) 로 __init__ 에서 만들어 둔다.
        ang_b = (self._beam_angles.unsqueeze(1)                     # (B,1)
                 + self._sector_offsets.unsqueeze(0)).reshape(-1)   # (B*S,)
        ang_w = yaw.unsqueeze(1) + ang_b.unsqueeze(0)               # (N,B*S) 월드 기준
        # 각도 지터(sim2real): 섹터 축약으로 못 잡는 잔여 각 불확실성(Livox 자체 오차).
        # 각을 흔들어 raycast 하면 그 요동이 물리적으로 재현된다(평가 시 obs_noise=False).
        if rc.obs_noise and rc.scan_angle_jitter > 0:
            ang_w = ang_w + (torch.rand_like(ang_w) * 2 - 1) * rc.scan_angle_jitter
        try:
            rays = self._raycast_fn(local_xy, ang_w, self._track_type, idx0,
                                    rc.raycast_half_window, rc.scan_max_range)
        except Exception:
            if self._raycast_fn is self._vt.raycast:
                raise
            print("[RacingEnv] compiled raycast failed; falling back to eager.")
            self._raycast_fn = self._vt.raycast
            rays = self._vt.raycast(local_xy, ang_w, self._track_type, idx0,
                                    rc.raycast_half_window, rc.scan_max_range)
        S = self._sector_offsets.shape[0]
        reduce_sector = (lambda r: r.reshape(r.shape[0], rc.n_beams, S).amin(dim=2)) \
            if S > 1 else (lambda r: r)
        # 벽 전용 스캔(상대차 가시성=벽 가림 판정용). 섹터 축약 후 빔 해상도로 돌린다.
        # ★사각지대는 여기 적용하지 않는다 — 이건 '벽이 물리적으로 가리는가'를 보는
        #   기하 판정이고, 사각지대는 자차 반사 제거라는 처리 단계의 산물이기 때문.
        scan = reduce_sector(rays)
        # 장애물 footprint 는 '축약 전' 레이 해상도에서 합성한다 — 실차에서도 장애물은
        # 섹터 안 점군으로 잡혀 그 섹터의 최소거리를 끌어내리기 때문(빔 각에만 얹으면
        # 섹터 폭 안에 있는데도 놓치는 경우가 생긴다).
        # 활성 장애물이 하나도 없으면(무장애 env 또는 평가) 건너뛴다.
        if rc.obstacles_enabled and bool(self._obs_active.any()):
            rays = overlay_obstacles(rays, local_xy, ang_w, self._obs_verts,
                                     self._obs_active, rc.scan_max_range)
        # ---- 실차 자차반사 제거 박스로 생기는 근거리 사각지대 모사 ----
        # (env_cfg.scan_blind_box 주석) 축약 '전' 레이 해상도에서 적용해야 실차와
        # 같다 — 실차도 점 단위로 지운 뒤 섹터 min 을 취하므로, 섹터 안에 박스 밖
        # 점이 하나라도 남아 있으면 그 거리가 살아난다. 빔 해상도에서 지우면
        # 그런 섹터까지 통째로 날아가 과도하게 가려진다.
        if rc.scan_blind_box is not None:
            bx0, bx1, bay = (float(v) for v in rc.scan_blind_box)
            hit_x = rays * torch.cos(ang_b)                 # (N,B*S) 차체 기준 히트 좌표
            hit_y = rays * torch.sin(ang_b)
            blind = (hit_x > bx0) & (hit_x < bx1) & (hit_y.abs() < bay)
            rays = torch.where(blind, torch.full_like(rays, rc.scan_max_range), rays)
        scan_wo = reduce_sector(rays)
        # ---- 상대차 감지 박스 오버레이 (구 원형 opp_radius 대체) ----
        # 실차는 상대차 뒷부분 감지 박스(12×12cm, 10~30cm 높이)만 LiDAR 로 검출한다.
        # 박스 중심(차중심에서 뒤로 det_box_rear)을 등가 원판으로 스캔에 반영하되,
        # 12cm 소형이라 빔 사이로 빠지지 않도록 각폭을 최소 반섹터로 바닥친다
        # (실측 Livox 누적 = 박스가 든 섹터를 반드시 채움).
        rel = opp_local_xy - local_xy
        cosy, siny = torch.cos(-yaw), torch.sin(-yaw)
        bx = rel[:, 0] * cosy - rel[:, 1] * siny            # 상대차 위치(바디프레임)
        by = rel[:, 0] * siny + rel[:, 1] * cosy
        rel_yaw = wrap_to_pi(opp_yaw - yaw)
        box_cx = bx - rc.det_box_rear * torch.cos(rel_yaw)  # 감지박스 중심(뒷부분)
        box_cy = by - rc.det_box_rear * torch.sin(rel_yaw)
        half_sector = rc.scan_fov / max(rc.n_beams - 1, 1)  # = beam_spacing/2
        overlaid = overlay_opponent(scan_wo, self._beam_angles,
                                    torch.stack([box_cx, box_cy], dim=1),
                                    rc.scan_max_range, rc.det_box_size * 0.70711,
                                    min_half_ext=half_sector)
        return scan, overlaid

    def _get_observations(self):
        la, ya, va = self._car_state(self.car_a)
        lb, yb, vb = self._car_state(self.car_b)
        pa = self._vt.project(la, self._track_type)
        pb = self._vt.project(lb, self._track_type)
        _, _, r_a = self._body_dyn(self.car_a, ya)
        _, _, r_b = self._body_dyn(self.car_b, yb)
        # 횡속도는 _get_dones 가 이번 스텝에 갱신해 둔 이동평균값을 그대로 쓴다
        # (여기서 다시 push 하면 스텝당 두 번 밀려 창 길이가 반토막 난다).
        obs_a = self._observe_car(la, ya, va, self._vy_obs_a, r_a, pa, lb, yb, vb, pb, self._hist_a)
        obs_b = self._observe_car(lb, yb, vb, self._vy_obs_b, r_b, pb, la, ya, va, pa, self._hist_b)
        return {"car_a": obs_a, "car_b": obs_b, "policy": torch.cat([obs_a, obs_b], dim=1)}

    # ------------------------------------------------------------------ #
    def _place(self, car: Articulation, env_ids, idx):
        """차량을 중심선 인덱스 idx 지점에 배치(자세=접선 방향, 속도/조인트 0). 스폰 s 반환."""
        g = self._track_type[env_ids]
        origins = self.scene.env_origins[env_ids]
        xy = self._vt.pts[g, idx]                  # (n,2) 중심선 위 스폰 좌표(로컬)
        yaw = self._vt.psi[g, idx]                 # (n,)  접선 방향
        root = car.data.default_root_state[env_ids].clone()
        root[:, 0] = origins[:, 0] + xy[:, 0]
        root[:, 1] = origins[:, 1] + xy[:, 1]
        root[:, 2] = origins[:, 2] + self.cfg.racing.wheel_radius   # 축고 = 바퀴 반경
        root[:, 3] = torch.cos(yaw * 0.5)   # qw
        root[:, 4] = 0.0
        root[:, 5] = 0.0
        root[:, 6] = torch.sin(yaw * 0.5)   # qz
        root[:, 7:] = 0.0                   # 선/각속도 0
        car.write_root_state_to_sim(root, env_ids)
        jp = car.data.default_joint_pos[env_ids]
        jv = car.data.default_joint_vel[env_ids]
        car.write_joint_state_to_sim(jp, jv, None, env_ids)
        return self._vt.s[g, idx]              # 스폰 지점의 진행도 s

    def _rand_gap_idx(self, env_ids, base_idx):
        """base_idx 에서 호길이 [gap_min, gap_max] 만큼 앞/뒤(50/50) 로 떨어진
        랜덤 중심선 인덱스.
        gap_min: 스폰 즉시 충돌 방지. gap_max: 30초 에피소드 내 상호작용
        (추월 시도)이 실제로 일어나도록 가까이 배치."""
        rc = self.cfg.racing
        g = self._track_type[env_ids]
        Lg = self._vt.length[g]
        tot = self._total_s[env_ids]
        n = len(env_ids)
        gap_min = torch.minimum(torch.full_like(tot, rc.spawn_min_gap_s), 0.25 * tot)
        gap_max = torch.minimum(torch.full_like(tot, rc.spawn_max_gap_s), 0.5 * tot)
        gap = gap_min + torch.rand(n, device=self.device) * (gap_max - gap_min).clamp(min=0.0)
        sign = torch.where(torch.rand(n, device=self.device) < 0.5, 1.0, -1.0)
        return (base_idx + (sign * gap / tot * Lg.float()).long()) % Lg

    def _reset_delay_a(self, env_ids):
        """Car A 리스폰: 배송 중 명령 비우기 + 히스토리 초기화 + 지연 재추첨.
        (차량별로 따로 두는 이유: 개별 리스폰 시 상대 차량의 in-flight 명령을
        지우면 안 되기 때문)"""
        self._act_buf_a[:, env_ids] = 0.0
        self._hist_a[env_ids] = 0.0
        self._cmd_a[env_ids] = 0.0
        self._delay_a[env_ids] = self._sample_delay(len(env_ids))

    def _reset_delay_b(self, env_ids):
        self._act_buf_b[:, env_ids] = 0.0
        self._hist_b[env_ids] = 0.0
        self._cmd_b[env_ids] = 0.0
        self._delay_b[env_ids] = self._sample_delay(len(env_ids))

    def _resample_v_cap_a(self, env_ids):
        """Car A 속도 핸디캡 재추첨 (스폰/리스폰마다 — env_cfg.v_cap_a_range 주석 참조)."""
        if self._v_cap_a is None:
            return
        lo, hi = self.cfg.racing.v_cap_a_range
        self._v_cap_a[env_ids] = (torch.rand(len(env_ids), device=self.device)
                                  * (float(hi) - float(lo)) + float(lo))

    def _set_prev_gap(self, env_ids):
        """스폰/리스폰 직후 s 간격 버퍼 재동기화 (텔레포트의 가짜 추월 이벤트 방지)."""
        tot = self._total_s[env_ids]
        g = self._prev_s_a[env_ids] - self._prev_s_b[env_ids]
        g = torch.where(g < -0.5 * tot, g + tot, g)
        g = torch.where(g > 0.5 * tot, g - tot, g)
        self._prev_gap_ab[env_ids] = g

    # ------------------------------------------------------------------ #
    # 장애물 (env_cfg.obstacles_enabled 주석 참조)
    # ------------------------------------------------------------------ #
    def _make_footprint(self, shape, size, yaw, center):
        """모양/크기/자세/중심 -> 2D 볼록 다각형 정점 (n,8,2), CCW, 로컬 좌표.
        box(사각4)/tri(삼각3)/cyl(팔각8). box/tri 는 마지막 정점 반복으로 8 패딩
        (닫힘 edge = 마지막슬롯->0 은 실제 변, 중간 퇴화 edge 는 overlay/충돌에서 무시)."""
        n = size.shape[0]; dev = self.device
        h = (size * 0.5)                                       # (n,) 반경/반변
        # box(정사각): 4 정점 CCW (+,-)(+,+)(-,+)(-,-)
        hh = h[:, None]
        box = torch.stack([
            torch.stack([ hh[:, 0], -hh[:, 0]], -1),
            torch.stack([ hh[:, 0],  hh[:, 0]], -1),
            torch.stack([-hh[:, 0],  hh[:, 0]], -1),
            torch.stack([-hh[:, 0], -hh[:, 0]], -1)], dim=1)   # (n,4,2)
        box = torch.cat([box, box[:, 3:4].expand(n, 4, 2)], dim=1)   # (n,8,2)
        # tri(외접반경 h), 각 90/210/330° CCW
        ta = torch.tensor([90., 210., 330.], device=dev) * (torch.pi / 180.0)
        tri3 = torch.stack([torch.cos(ta), torch.sin(ta)], -1)      # (3,2)
        tri = h[:, None, None] * tri3[None]                        # (n,3,2)
        tri = torch.cat([tri, tri[:, 2:3].expand(n, 5, 2)], dim=1)  # (n,8,2)
        # cyl(팔각, 반경 h)
        ca = torch.arange(8, device=dev) * (2 * torch.pi / 8)
        oct8 = torch.stack([torch.cos(ca), torch.sin(ca)], -1)     # (8,2)
        cyl = h[:, None, None] * oct8[None]                        # (n,8,2)
        base = torch.stack([box, tri, cyl], dim=1)[torch.arange(n, device=dev), shape]  # (n,8,2)
        # yaw 회전 + 중심 이동
        c, s = torch.cos(yaw), torch.sin(yaw)                     # (n,)
        rx = base[..., 0] * c[:, None] - base[..., 1] * s[:, None]
        ry = base[..., 0] * s[:, None] + base[..., 1] * c[:, None]
        return torch.stack([rx, ry], -1) + center[:, None, :]     # (n,8,2)

    def _resample_obstacles(self, env_ids, sa, sb):
        """env 리셋마다 2~3개 장애물을 트랙 위 통과 가능한 랜덤 지점에 재스폰.

        모양/크기/자세 랜덤(≤50cm), 스폰 지점 근처 회피, 양쪽 통과폭 보장(막힘 방지).
        sa/sb: 해당 env 의 스폰 호길이(장애물이 스폰 지점에 겹치지 않도록)."""
        rc = self.cfg.racing
        if not rc.obstacles_enabled or len(env_ids) == 0:
            return
        dev = self.device
        n = len(env_ids)
        self._obs_active[env_ids] = False
        g = self._track_type[env_ids]                             # (n,)
        Lg = self._vt.length[g].clamp(min=1)                      # (n,)
        tots = self._vt.total_s[g]                                # (n,)
        s_all = self._vt.s[g]                                     # (n,L)
        hw_all = self._vt.hw[g]                                   # (n,L)
        # 개수: obstacle_count_weights(P(0/1/2/3))로 뽑아 3개(혼잡) 확률을 낮춘다.
        cw = torch.tensor(rc.obstacle_count_weights, device=dev, dtype=torch.float)
        cnt = torch.multinomial(cw, n, replacement=True)          # (n,) in {0..len(cw)-1}
        has = torch.rand(n, device=dev) < rc.obstacle_p
        cnt = torch.where(has, cnt, torch.zeros_like(cnt))        # (n,) env 별 개수
        R = 24                                                    # 후보 수(거절 샘플링)
        smin, smax = rc.obstacle_size_range
        rows = torch.arange(n, device=dev)
        gi = g[:, None].expand(n, R)                              # 후보 gather 용
        placed_c = torch.zeros(n, self._obs_kmax, 2, device=dev)  # 배치된 장애물 중심(간격 검사)
        placed_v = torch.zeros(n, self._obs_kmax, dtype=torch.bool, device=dev)
        for k in range(self._obs_kmax):
            slot = cnt > k                                        # (n,) 이 슬롯 채울 env
            if not bool(slot.any()):
                continue
            shape = torch.randint(0, 3, (n,), device=dev)         # 0 box/1 tri/2 cyl
            size = torch.rand(n, device=dev) * (smax - smin) + smin
            yaw = torch.rand(n, device=dev) * (2 * torch.pi)
            # 외접반경(배치 폭 필터): box=√2·h, tri/cyl=h
            circ = size * 0.5 * torch.where(shape == 0, torch.full_like(size, 1.41421356),
                                            torch.ones_like(size))
            # 한쪽 벽만 통과폭 보장: 필요 반폭 = circ + (pass_clear + wall_min_clear)/2
            #   (한쪽에 통과 레인 pass_clear, 반대쪽에 최소여유 wall_min_clear 가 들어갈 폭).
            need_hw = circ + 0.5 * (rc.obstacle_pass_clear
                                    + rc.obstacle_wall_min_clear)  # (n,)
            cand = (torch.rand(n, R, device=dev) * Lg[:, None]).long()
            cand = cand.clamp(max=(Lg[:, None] - 1).clamp(min=0))
            cs = torch.gather(s_all, 1, cand)                     # (n,R)
            chw = torch.gather(hw_all, 1, cand)                   # (n,R)
            cand_pt = self._vt.pts[gi, cand]                      # (n,R,2) 후보 중심선점
            d_sa = (cs - sa[:, None]).abs(); d_sa = torch.minimum(d_sa, tots[:, None] - d_sa)
            d_sb = (cs - sb[:, None]).abs(); d_sb = torch.minimum(d_sb, tots[:, None] - d_sb)
            valid = (d_sa > rc.obstacle_spawn_clear_s) & (d_sb > rc.obstacle_spawn_clear_s) \
                    & (chw > need_hw[:, None])                    # (n,R) 스폰회피 + 한쪽 통과폭
            if k > 0:                                            # 이미 배치된 장애물과 최소 간격
                sep = (cand_pt[:, :, None, :] - placed_c[:, None, :k, :]).norm(dim=3)  # (n,R,k)
                far = (sep > rc.obstacle_min_sep) | ~placed_v[:, None, :k]
                valid = valid & far.all(dim=2)
            has_valid = valid.any(dim=1)
            first = torch.argmax(valid.to(torch.int8), dim=1)     # 첫 valid(없으면 0)
            pick = torch.gather(cand, 1, first[:, None]).squeeze(1)   # (n,) 중심선 idx
            fill = slot & has_valid                               # (n,) 실제 채우는 env
            cpt = self._vt.pts[g, pick]                           # (n,2)
            psi = self._vt.psi[g, pick]                           # (n,)
            nrm = torch.stack([-torch.sin(psi), torch.cos(psi)], dim=1)   # (n,2)
            hwp = self._vt.hw[g, pick]                            # (n,)
            # 한쪽 벽에만 pass_clear 통과 레인을 남기고 (선택된) 벽 쪽으로 오프셋.
            #   off_max: 반대벽에 wall_min_clear 만 남기고 벽으로 밀 수 있는 한계.
            #   off_min: 먼 쪽(통과 레인)에 pass_clear 를 열기 위한 최소 밀기(넓은 트랙=0).
            #   sign: 어느 벽에 붙일지 좌/우 랜덤. off_min>0 이면 좁은 구간이라 반드시 한쪽 벽에 붙음.
            off_max = (hwp - circ - rc.obstacle_wall_min_clear).clamp(min=0.0)  # (n,)
            off_min = (rc.obstacle_pass_clear - (hwp - circ)).clamp(min=0.0)    # (n,)
            off = off_min + torch.rand(n, device=dev) * (off_max - off_min).clamp(min=0.0)
            sign = torch.where(torch.rand(n, device=dev) < 0.5,
                               torch.full_like(off, -1.0), torch.full_like(off, 1.0))
            lat = sign * off
            center = cpt + nrm * lat[:, None]                     # (n,2)
            placed_c[rows, k] = center                            # 다음 슬롯 간격 검사용(fill 만 유효)
            placed_v[:, k] = fill
            verts = self._make_footprint(shape, size, yaw, center)   # (n,8,2)
            ii = env_ids[fill]
            self._obs_verts[ii, k] = verts[fill]
            self._obs_active[ii, k] = True
            self._obs_center[ii, k] = center[fill]
            self._obs_rad[ii, k] = circ[fill]
            self._obs_shape[ii, k] = shape[fill]
            self._obs_size[ii, k] = size[fill]
            self._obs_yaw[ii, k] = yaw[fill]

    def _obstacle_hit(self, xy):
        """xy:(N,2) -> (N,) bool: 활성 장애물 footprint 까지 obstacle_hit_margin 이내.
        볼록 다각형: 내부(모든 실변의 왼쪽) 또는 최근접 변까지 거리 < margin 이면 충돌."""
        rc = self.cfg.racing
        if not rc.obstacles_enabled:
            return torch.zeros(xy.shape[0], dtype=torch.bool, device=self.device)
        m = rc.obstacle_hit_margin
        v1 = self._obs_verts                                     # (N,K,8,2)
        e = torch.roll(v1, -1, dims=2) - v1                      # (N,K,8,2) edge 벡터
        w = xy[:, None, None, :] - v1                            # (N,K,8,2) 점-정점
        ee = (e * e).sum(-1)                                     # (N,K,8)
        real = ee > 1e-8
        tt = ((w * e).sum(-1) / ee.clamp_min(1e-9)).clamp(0.0, 1.0)
        proj = v1 + tt[..., None] * e
        d = (xy[:, None, None, :] - proj).norm(dim=-1)           # (N,K,8) 점-변 거리
        d = torch.where(real, d, torch.full_like(d, 1e9))
        dmin = d.amin(dim=2)                                     # (N,K)
        cross = e[..., 0] * w[..., 1] - e[..., 1] * w[..., 0]    # (N,K,8) CCW: 내부면 >0
        inside = ((cross >= -1e-6) | ~real).all(dim=2)          # (N,K)
        hit = self._obs_active & (inside | (dmin < m))          # (N,K)
        return hit.any(dim=1)

    def _spawn_pair(self, env_ids):
        """두 차량 동시 스폰: car_a 랜덤 지점, car_b 는 car_a 에서 최소 gap 유지."""
        g = self._track_type[env_ids]
        Lg = self._vt.length[g]
        idx_a = (torch.rand(len(env_ids), device=self.device) * Lg).long() % Lg
        idx_b = self._rand_gap_idx(env_ids, idx_a)
        sa = self._place(self.car_a, env_ids, idx_a)
        sb = self._place(self.car_b, env_ids, idx_b)
        # 진행도/체인 초기화
        self._travel_a[env_ids] = 0.0
        self._travel_b[env_ids] = 0.0
        self._prev_act_a[env_ids] = 0.0
        self._prev_act_b[env_ids] = 0.0
        self._reset_delay_a(env_ids)
        self._reset_delay_b(env_ids)
        self._prev_v_a[env_ids] = 0.0
        self._prev_v_b[env_ids] = 0.0
        # 스폰은 정지 상태(_place 가 속도 0)이므로 vy 이동평균 창도 비운다.
        # 안 비우면 직전 에피소드의 횡속도가 최대 3스텝 새어 들어온다.
        self._vy_buf_a[env_ids] = 0.0
        self._vy_buf_b[env_ids] = 0.0
        self._prev_s_a[env_ids] = sa
        self._prev_s_b[env_ids] = sb
        self._set_prev_gap(env_ids)
        self._resample_v_cap_a(env_ids)
        # 지면 마찰 재추첨 (env 단위 — 두 차 공유)
        mr = self.cfg.tire.mu_range
        if mr is not None:
            self._mu[env_ids] = (torch.rand(len(env_ids), device=self.device)
                                 * (float(mr[1]) - float(mr[0])) + float(mr[0]))
        # 장애물 재추첨 (env 단위 — 두 차 공유하는 물리 트랙 특징) + grace 리셋
        self._resample_obstacles(env_ids, sa, sb)
        self._since_spawn_a[env_ids] = 0
        self._since_spawn_b[env_ids] = 0
        self._update_obstacle_visuals(env_ids)          # GUI(wall_visuals) 시에만 갱신

    def _respawn_terminated(self, pa, pb, truncated):
        """차량별 리셋: term 난 차만 리스폰(상대 차량은 계속 주행).
        리스폰 위치는 '주행 중인 상대의 현재 위치'에서 최소 gap 을 유지한 중심선 지점.
        시간초과(truncated) env 는 프레임워크 경로(_reset_idx)가 두 차를 함께 스폰."""
        live = ~truncated
        need_a = self._term_a & live
        need_b = self._term_b & live
        both = need_a & need_b
        only_a = need_a & ~both
        only_b = need_b & ~both
        if both.any():                       # 동시 종료(양쪽 과실 등): 쌍 스폰 재사용
            self._spawn_pair(both.nonzero(as_tuple=False).squeeze(-1))
        if only_a.any():
            ids = only_a.nonzero(as_tuple=False).squeeze(-1)
            sa = self._place(self.car_a, ids, self._rand_gap_idx(ids, pb["idx"][ids]))
            self._travel_a[ids] = 0.0
            self._prev_act_a[ids] = 0.0
            self._reset_delay_a(ids)
            self._prev_v_a[ids] = 0.0
            self._vy_buf_a[ids] = 0.0        # vy 이동평균 창 비움(스폰=정지)
            self._prev_s_a[ids] = sa
            self._set_prev_gap(ids)
            self._resample_v_cap_a(ids)
            self._since_spawn_a[ids] = 0     # 장애물 grace(개별 리스폰 — 장애물은 유지)
        if only_b.any():
            ids = only_b.nonzero(as_tuple=False).squeeze(-1)
            sb = self._place(self.car_b, ids, self._rand_gap_idx(ids, pa["idx"][ids]))
            self._travel_b[ids] = 0.0
            self._prev_act_b[ids] = 0.0
            self._reset_delay_b(ids)
            self._prev_v_b[ids] = 0.0
            self._vy_buf_b[ids] = 0.0        # vy 이동평균 창 비움(스폰=정지)
            self._prev_s_b[ids] = sb
            self._set_prev_gap(ids)
            self._since_spawn_b[ids] = 0     # 장애물 grace(개별 리스폰 — 장애물은 유지)

    # ------------------------------------------------------------------ #
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        self._spawn_pair(env_ids)

    # ------------------------------------------------------------------ #
    # 학습 루프가 읽는 차량별 정보
    # ------------------------------------------------------------------ #
    def dual_info(self):
        return {
            "rew_a": self._rew_a, "rew_b": self._rew_b,
            "term_a": self._term_a, "term_b": self._term_b,
            "trunc": self._trunc,
            # 리셋 전 상태에서 계산한 s_{t+1} 관측 (리플레이 버퍼의 next_obs 로 사용할 것)
            "final_obs_a": self._final_obs_a, "final_obs_b": self._final_obs_b,
            # 종료 원인 분해 (진단용; 학습 루프는 사용하지 않음)
            "causes": self._causes,
        }
