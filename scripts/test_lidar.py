#!/usr/bin/env python3
"""학습된 정책을 '실제 3D LiDAR 관측'으로 구동하는 평가 (sim-to-real 리허설).

test.py 와의 차이:
  - test.py       : 학습 때와 같은 해석적 2D 레이캐스트 스캔으로 정책 평가
  - test_lidar.py : 평가 대상 차(--car, 기본 b=Pow=실차 배포 대상)의 스캔 관측
                    32차원을 "실제 벽 메시 + 3D LiDAR(RayCaster) + 실차 가공
                    파이프라인(높이밴드 -> 32섹터)" 출력으로 교체해 정책 평가.
    -> 학습 가중치가 실차 LiDAR 파이프라인 관측에서도 잘 동작하는지 확인.

★ Car A 속도 핸디캡(env_cfg.v_cap_a_range, 추월 연습용)은
  --car a 평가 시 자동 비활성(평가 대상을 핸디캡하면 안 됨),
  --car b 평가 시 유지(A = 느린 트래픽 역할, 추월 리허설).

측정 지표:
  - 평가 차 에피소드 리턴 / 종료율 / 속도 (성능 유지 여부)
  - |Δaction|: 동일 상태에서 LiDAR 관측 행동 vs 해석 관측 행동 차이
    (관측 갭이 정책 출력에 미치는 영향의 직접 측정)
  - 스캔 MAE: LiDAR 파이프라인 스캔 vs 해석 스캔

제약: RayCaster 는 단일 정적 메시만 지원 -> num_envs=1. 상대차는
LiDAR 메시에 없으므로, 실차에서 상대차 검출이 별도 모듈이듯 여기서도
해석적 footprint 오버레이(overlay_opponent)를 LiDAR 벽 스캔 위에 얹는다.

실행:
  conda activate env_isaacsim
  python -u scripts/test_lidar.py --ckpt_dir dacerpp_runs/20260710 --steps 2000 --car b
  # GUI 로 보려면 --headless 없이, 끄려면 --headless
"""
from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, required=True,
                    help="cvar.pt/pow.pt 가 있는 체크포인트 폴더")
parser.add_argument("--steps", type=int, default=10000)
parser.add_argument("--car", type=str, default="b", choices=["a", "b"],
                    help="LiDAR 평가 대상 차 (b=Pow, 실차 배포 대상 — 기본)")
parser.add_argument("--channels", type=int, default=16, help="LiDAR 수직 채널 수")
parser.add_argument("--sector_mode", type=str, default="center", choices=["center", "min"],
                    help="섹터 축약: center=중심각 최근접 레이(권장), min=최소거리(보수적)")
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--tracks", type=str, default="",
                    help="고정 배정할 트랙 이름(콤마 구분, 대소문자 무시). test.py 와 동일. "
                         "num_envs=1 이라 첫 이름만 사용. 미지정 시 랜덤. (test.py --list_tracks 로 확인)")
parser.add_argument("--no_obstacles", action="store_true",
                    help="장애물 스폰 비활성화 (기본: 학습·대회와 동일하게 0~3개 스폰). "
                         "장애물은 LiDAR 스캔에도 반영되고 GUI 에도 표시된다.")
parser.add_argument("--all_tracks", action="store_true",
                    help="f1tenth 실서킷 + 절차 생성 맵도 함께 사용. 기본은 comp 폴더"
                         "(대회 코스+실측 연습맵)만 — 학습(기본 comp-only)과 트랙 분포 일치.")
parser.add_argument("--cvar_eta", type=float, default=0.5)
parser.add_argument("--pow_eta", type=float, default=1.3)
parser.add_argument("--log_every", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

_cvar_ckpt = os.path.join(args.ckpt_dir, "cvar.pt")
_pow_ckpt = os.path.join(args.ckpt_dir, "pow.pt")
if not (os.path.isfile(_cvar_ckpt) and os.path.isfile(_pow_ckpt)):
    parser.error(f"--ckpt_dir: {args.ckpt_dir} 에 cvar.pt/pow.pt 가 없습니다.")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ----------------------------------------------------------------------------
import numpy as np
import torch

from isaaclab.sensors import RayCaster, RayCasterCfg, patterns

from dacerpp_lab.env_cfg import RacingEnvCfg
from dacerpp_lab.racing_env import RacingEnv
from dacerpp_lab.track_field import TrackField
from dacerpp_lab.vectorized_track import overlay_opponent, overlay_obstacles, wrap_to_pi
from dacer_pp import DACERppConfig, DACERpp, conservative_cvar, aggressive_pow

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WALL_PRIM = "/World/TrackWalls"
Z_BAND = (0.03, 0.28)


class LidarRacingEnv(RacingEnv):
    """RacingEnv + 실제 벽 충돌 메시 + 평가 대상 차 3D LiDAR."""

    def _setup_scene(self):
        super()._setup_scene()
        self._spawn_wall_collision_mesh()
        # 장애물/감지박스 시각 프림 (wall_visuals=False 라 base 가 안 만들므로 직접 호출).
        # 벽은 위 충돌 메시가 displayColor 로 이미 렌더된다.
        if self.cfg.racing.obstacles_enabled:
            self._spawn_obstacle_visuals()
        self._spawn_det_box_visuals()
        lidar_cfg = RayCasterCfg(
            prim_path=f"/World/envs/env_0/Car_{args.car.upper()}/base_link",
            mesh_prim_paths=[WALL_PRIM],
            # 실측(2026-07-16): 지면->빔 원점(반구 밑면) 12.2cm.
            # base_link 원점은 차축 높이(=바퀴 반경 0.06m)이므로 오프셋 = 0.122-0.06.
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.062)),
            ray_alignment="base",
            pattern_cfg=patterns.LidarPatternCfg(
                channels=args.channels,
                vertical_fov_range=(-10.0, 10.0),
                horizontal_fov_range=(-135.0, 135.0),
                horizontal_res=1.0,
            ),
            max_distance=float(self.cfg.racing.scan_max_range),
        )
        self._lidar = RayCaster(lidar_cfg)
        self.scene.sensors["lidar"] = self._lidar

    def _spawn_wall_collision_mesh(self):
        import omni.usd
        from pxr import UsdGeom, UsdPhysics

        if not hasattr(self, "_field"):
            self._field = TrackField(self.cfg.track_field)
        g = int(self._field.env_track_type[0])
        cl = self._field.centerlines[g]
        hw = self._field.half_widths[g]
        psi = self._field.features[g]["psi"]
        nrm = np.stack([-np.sin(psi), np.cos(psi)], axis=1)
        h = float(self.cfg.track_field.params.wall_height)
        origin = self._terrain.env_origins[0].detach().cpu().numpy()

        pts, indices = [], []
        for side in (+1.0, -1.0):
            b = cl + side * nrm * hw[:, None]
            n = len(b)
            base = len(pts)
            pts += [(float(x + origin[0]), float(y + origin[1]), float(origin[2])) for x, y in b]
            pts += [(float(x + origin[0]), float(y + origin[1]), float(origin[2] + h)) for x, y in b]
            for i in range(n):
                j = (i + 1) % n
                indices += [base + i, base + j, base + n + j]
                indices += [base + i, base + n + j, base + n + i]

        stage = omni.usd.get_context().get_stage()
        mesh = UsdGeom.Mesh.Define(stage, WALL_PRIM)
        mesh.CreatePointsAttr(pts)
        mesh.CreateFaceVertexCountsAttr([3] * (len(indices) // 3))
        mesh.CreateFaceVertexIndicesAttr(indices)
        mesh.CreateDisplayColorAttr([(0.85, 0.4, 0.1)])
        mesh.CreateDoubleSidedAttr(True)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
        print(f"[LIDAR-EVAL] wall mesh: track={g} ({self._field.track_names[g]}), "
              f"tris={len(indices)//3}")


def lidar_wall_scan(env, car, rc, B, fov, beam_spacing):
    """3D LiDAR -> 높이밴드 -> 32섹터 벽 거리 (실차 가공 파이프라인)."""
    lidar = env.scene.sensors["lidar"]
    hits = lidar.data.ray_hits_w[0]
    spos = lidar.data.pos_w[0]
    _, yaw, _ = env._car_state(car)

    d = hits - spos
    finite = torch.isfinite(d).all(dim=1)
    zok = (hits[:, 2] > Z_BAND[0]) & (hits[:, 2] < Z_BAND[1])
    valid = finite & zok
    dxy = d[valid, :2]
    dist = torch.linalg.norm(dxy, dim=1)
    az = torch.atan2(dxy[:, 1], dxy[:, 0]) - yaw[0]
    az = (az + math.pi) % (2 * math.pi) - math.pi
    bins = torch.round((az + fov) / beam_spacing).long()
    inb = (bins >= 0) & (bins < B) & (az.abs() <= fov + 0.5 * beam_spacing)
    scan = torch.full((B,), rc.scan_max_range, device=env.device)
    if args.sector_mode == "min":
        scan.scatter_reduce_(0, bins[inb], dist[inb], reduce="amin", include_self=True)
    else:  # center: 섹터 중심각에 가장 가까운 레이 (해석 스캔과 정합, 검증 오차 ~3.5cm)
        az_i, dist_i, bins_i = az[inb], dist[inb], bins[inb]
        for b in range(B):
            m = bins_i == b
            if m.any():
                cang = -fov + b * beam_spacing
                scan[b] = dist_i[m][(az_i[m] - cang).abs().argmin()]
    return scan


def main():
    cfg = RacingEnvCfg()
    cfg.scene.num_envs = 1
    cfg.project_dir = PROJECT_DIR
    cfg.track_field.num_envs = 1
    cfg.wall_visuals = False
    cfg.racing.obs_noise = False
    if not args.all_tracks:                       # 기본: comp 폴더만 (학습과 트랙 분포 일치)
        cfg.track_field.num_procedural = 0
        cfg.track_field.use_f1tenth = False
    if args.no_obstacles:                         # 장애물 끄기(기본은 학습·대회처럼 켬)
        cfg.racing.obstacles_enabled = False
    # 평가: 트랙 무작위 1개(실행마다 다름). --tracks 로 고정 지정(num_envs=1 이라 첫 이름).
    # 벽 메시/LiDAR/장애물은 env_track_type[0] 기준으로 생성되므로 자동 반영.
    cfg.track_field.random_tracks = True
    cfg.track_field.pinned_tracks = tuple(
        t.strip() for t in args.tracks.split(",") if t.strip())
    cfg.observation_space = 2 * cfg.racing.obs_dim()
    cfg.scene.env_spacing = TrackField(cfg.track_field).suggested_env_spacing
    # A 속도 핸디캡: A 를 '평가'할 때는 비활성 (핸디캡 걸린 대상 평가는 무의미),
    # B 평가 시에는 유지 (A = 느린 트래픽, 추월 리허설)
    if args.car == "a":
        cfg.racing.v_cap_a_range = None
    print(f"[LIDAR-EVAL] subject=Car_{args.car.upper()} | "
          f"A 속도캡={cfg.racing.v_cap_a_range}")

    env = LidarRacingEnv(cfg)
    rc = cfg.racing
    dev = env.device
    obs_dim, act_dim = rc.obs_dim(), rc.act_dim()
    B = rc.n_beams
    fov = rc.scan_fov
    beam_spacing = 2.0 * fov / (B - 1)
    trk = env._field.track_names[int(env._field.env_track_type[0])]
    print(f"[LIDAR-EVAL] 트랙={trk} | 장애물={'끔' if args.no_obstacles else '켬(0~3개)'}")

    def make_agent(risk, seed):
        c = DACERppConfig(obs_dim=obs_dim, act_dim=act_dim, risk=risk,
                          device=str(dev), use_compile=False, seed=seed)
        return DACERpp(c)

    agent_cvar = make_agent(conservative_cvar(args.cvar_eta), seed=0).load(_cvar_ckpt)
    agent_pow = make_agent(aggressive_pow(args.pow_eta), seed=1).load(_pow_ckpt)
    print(f"[LIDAR-EVAL] ckpt loaded <- {args.ckpt_dir} (sector_mode={args.sector_mode})")

    sub = args.car                                  # 평가 대상 ("a"|"b")
    sub_agent = agent_cvar if sub == "a" else agent_pow
    oth_agent = agent_pow if sub == "a" else agent_cvar
    sub_car = env.car_a if sub == "a" else env.car_b
    oth_car = env.car_b if sub == "a" else env.car_a
    NAME = f"Car_{sub.upper()}"

    obs, _ = env.reset()
    oa, ob = obs["car_a"], obs["car_b"]
    det = not args.stochastic

    ep_ret = torch.zeros(1, device=dev)
    fin, act_gap, scan_mae, vlog, terms = [], [], [], [], 0
    causes = {c: 0 for c in ("off", "crash", "flip", "spun", "obs")}   # 리셋 원인 분해
    for it in range(args.steps):
        # ---- 평가 차 관측의 스캔 32차원을 LiDAR 파이프라인으로 교체 ----
        wall = lidar_wall_scan(env, sub_car, rc, B, fov, beam_spacing)  # (B,)
        ls, ys, vs = env._car_state(sub_car)
        lo, yo, _ = env._car_state(oth_car)                            # 상대차 위치+헤딩
        rel = lo - ls
        cosy, siny = torch.cos(-ys), torch.sin(-ys)
        bx = rel[:, 0] * cosy - rel[:, 1] * siny
        by = rel[:, 0] * siny + rel[:, 1] * cosy
        # 상대차 감지 박스(뒷부분 12×12cm) 중심을 LiDAR 벽 스캔에 오버레이 (train/test 와 동일;
        # 상대차는 움직여 RayCaster 정적메시로 못 잡으므로 해석적 검출. 각폭은 반섹터로 바닥).
        ry = wrap_to_pi(yo - ys)
        box_c = torch.stack([bx - rc.det_box_rear * torch.cos(ry),
                             by - rc.det_box_rear * torch.sin(ry)], dim=1)
        wall = overlay_opponent(wall.unsqueeze(0), env._beam_angles, box_c,
                                rc.scan_max_range, rc.det_box_size * 0.70711,
                                min_half_ext=beam_spacing * 0.5)          # (1,B)
        # ---- 장애물을 LiDAR 스캔에 반영 (학습 해석 스캔과 동일; 안 하면 정책이 장애물을
        #      못 봐서 눈먼 채 부딪혀 리셋된다). 장애물은 정적이라 벽처럼 해석적 오버레이. ----
        if rc.obstacles_enabled and bool(env._obs_active.any()):
            ang_w = ys.unsqueeze(1) + env._beam_angles.unsqueeze(0)      # (1,B) 월드 빔각
            wall = overlay_obstacles(wall, ls, ang_w, env._obs_verts,
                                     env._obs_active, rc.scan_max_range)  # (1,B)
        o_ana = oa if sub == "a" else ob                                # 해석 관측(참조)
        o_lidar = o_ana.clone()
        o_lidar[:, :B] = (wall / rc.scan_max_range).clamp(0, 1)
        scan_mae.append((o_lidar[0, :B] - o_ana[0, :B]).abs().mean().item() * rc.scan_max_range)
        vlog.append(float(vs[0]))

        a_sub = sub_agent.act_batch(o_lidar, deterministic=det)        # LiDAR 관측 행동
        a_ref = sub_agent.act_batch(o_ana, deterministic=det)          # 해석 관측 행동(참조)
        act_gap.append((a_sub - a_ref).abs().mean().item())
        # 비평가 차량은 항상 결정적 — --stochastic 은 평가 대상 차에만 적용
        a_oth = oth_agent.act_batch(ob if sub == "a" else oa, deterministic=True)

        action = (torch.cat([a_sub, a_oth], dim=1) if sub == "a"
                  else torch.cat([a_oth, a_sub], dim=1))
        next_obs, _, terminated, truncated, _ = env.step(action)
        di = env.dual_info()
        oa, ob = next_obs["car_a"], next_obs["car_b"]

        ep_ret += di[f"rew_{sub}"]
        terms += int(di[f"term_{sub}"].sum())
        for c in causes:                              # 리셋 원인 분해(off/crash/flip/spun/obs)
            causes[c] += int(di["causes"][f"{c}_{sub}"].sum())
        done = di[f"term_{sub}"] | di["trunc"]
        if done.any():
            fin.append(float(ep_ret[0]))
            ep_ret[:] = 0.0

        if it % args.log_every == 0 and it > 0:
            r = sum(fin[-50:]) / len(fin[-50:]) if fin else float("nan")
            vm = sum(vlog[-args.log_every:]) / args.log_every
            print(f"[{it}] ep_ret(최근50)={r:.2f} | {NAME} 종료 {terms}회 "
                  f"[{' '.join(f'{c}={causes[c]}' for c in causes)}] | v평균 {vm:.2f}m/s | "
                  f"|Δact|={sum(act_gap[-args.log_every:])/args.log_every:.4f} | "
                  f"scan MAE={sum(scan_mae[-args.log_every:])/args.log_every:.3f}m", flush=True)

    print("=" * 70)
    n = max(len(act_gap), 1)
    print(f"[RESULT] steps={args.steps} 에피소드 {len(fin)}개 | {NAME} term {terms}회 | "
          f"v평균 {sum(vlog)/max(len(vlog),1):.2f}m/s")
    print(f"  리셋 원인 분해: " + "  ".join(f"{c}={causes[c]}" for c in causes)
          + "  (off=이탈, crash=차간충돌, obs=장애물, spun=스핀, flip=전복)")
    if fin:
        print(f"  {NAME} 평균 에피소드 리턴 (LiDAR 관측) = {sum(fin)/len(fin):.2f}")
    print(f"  평균 |Δaction| (LiDAR vs 해석 관측)     = {sum(act_gap)/n:.4f} (행동범위 [-1,1])")
    print(f"  평균 스캔 MAE                           = {sum(scan_mae)/n:.3f} m")
    print("  해석: |Δaction| 이 탐험노이즈(0.15) 보다 충분히 작으면 관측 갭이 정책에 무해.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
