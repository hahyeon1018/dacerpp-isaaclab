#!/usr/bin/env python3
"""3D LiDAR 시뮬레이션 검증: 해석적 2D 레이캐스트 vs 실제 메시 + 3D LiDAR.

학습은 벽 폴리라인에 대한 해석적 2D 레이캐스트(_vt.raycast)를 쓴다.
이 스크립트는 그 근사가 "실제 3D LiDAR + 실제 벽 지오메트리" 파이프라인과
일치하는지 검증한다:

  1) env_0 트랙의 덕트 벽을 실제 USD 삼각 메시(+충돌 API)로 스폰
  2) Car_A 에 3D LiDAR(RayCaster + LidarPatternCfg: 16채널, 수직 ±10°,
     수평 ±135° @1°)를 장착
  3) 실차 가공 파이프라인 그대로: 높이 밴드 필터 -> 32개 방위각 섹터
     최소거리 -> max_range 클립
  4) 매 스텝 해석 스캔과 섹터 스캔의 빔별 오차를 누적, 통계 출력

제약: Isaac Lab RayCaster 는 단일 정적 메시만 지원 -> num_envs=1 고정,
상대차량(car_b)은 메시에 없으므로 "벽 스캔"만 비교한다(상대차 footprint
오버레이는 해석 단계라 검증 대상 아님).

실행:
  conda activate env_isaacsim
  python -u scripts/validate_lidar.py --headless --steps 300
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--channels", type=int, default=16, help="LiDAR 수직 채널 수")
parser.add_argument("--track_id", type=int, default=None,
                    help="검증할 트랙 layout id (기본: env_0 에 배정된 트랙)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ----------------------------------------------------------------------------
import math

import numpy as np
import torch

from isaaclab.sensors import RayCaster, RayCasterCfg, patterns

from dacerpp_lab.env_cfg import RacingEnvCfg
from dacerpp_lab.racing_env import RacingEnv
from dacerpp_lab.track_field import TrackField

import os
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WALL_PRIM = "/World/TrackWalls"
Z_BAND = (0.03, 0.28)          # 높이 밴드(m): 지면 노이즈 제외 ~ 벽 상단(0.3) 미만


class LidarValidationEnv(RacingEnv):
    """RacingEnv + 실제 벽 충돌 메시 + 3D LiDAR 센서."""

    def _setup_scene(self):
        super()._setup_scene()
        self._spawn_wall_collision_mesh()
        # Car_A 루트에 3D LiDAR (base_link 중심에서 z+0.12)
        lidar_cfg = RayCasterCfg(
            prim_path="/World/envs/env_0/Car_A/base_link",
            mesh_prim_paths=[WALL_PRIM],
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.12)),
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
        """env_0 트랙의 좌/우 덕트 벽을 삼각 메시 + CollisionAPI 로 스폰."""
        import omni.usd
        from pxr import UsdGeom, UsdPhysics

        if not hasattr(self, "_field"):
            self._field = TrackField(self.cfg.track_field)
        g = int(self._field.env_track_type[0]) if args.track_id is None else args.track_id
        self._val_track = g
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
            for i in range(n):          # quad -> 삼각형 2개 (양면은 DoubleSided 로)
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
        # 실제 충돌 활성화 (정적 콜라이더, triangle mesh 그대로)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
        print(f"[VAL] wall collision mesh spawned: track={g}, verts={len(pts)}, tris={len(indices)//3}")


def main():
    cfg = RacingEnvCfg()
    cfg.scene.num_envs = 1                 # RayCaster 단일 정적 메시 제약
    cfg.project_dir = PROJECT_DIR
    cfg.track_field.num_envs = 1
    cfg.wall_visuals = False               # 검증용 충돌 메시가 대신함
    cfg.racing.obs_noise = False
    cfg.observation_space = 2 * cfg.racing.obs_dim()
    spacing = TrackField(cfg.track_field).suggested_env_spacing
    cfg.scene.env_spacing = spacing

    env = LidarValidationEnv(cfg)
    rc = cfg.racing
    dev = env.device
    B = rc.n_beams
    fov = rc.scan_fov                       # half-angle (rad), 빔 = ±fov 등간격 B개
    beam_spacing = 2.0 * fov / (B - 1)

    obs, _ = env.reset()
    lidar = env.scene.sensors["lidar"]

    errs, agree = {}, {}
    for step in range(args.steps):
        # 스크립트 주행: 완만한 사인 조향 + 저속 (v = (act+1)/2 * v_max)
        steer = 0.3 * math.sin(step / 40.0)
        thr = -0.6                          # ~= 0.2 * v_max
        action = torch.tensor([[steer, thr, 0.0, -1.0]], device=dev)  # car_b 정지
        env.step(action)

        # ---- 3D LiDAR -> 높이밴드 -> 32섹터 최소거리 (실차 가공 파이프라인) ----
        hits = lidar.data.ray_hits_w[0]                     # (R,3) 월드 히트
        spos = lidar.data.pos_w[0]                          # 센서 월드 위치
        la, yaw, _ = env._car_state(env.car_a)
        yaw0 = yaw[0]

        d = hits - spos
        finite = torch.isfinite(d).all(dim=1)
        zok = (hits[:, 2] > Z_BAND[0]) & (hits[:, 2] < Z_BAND[1])
        valid = finite & zok
        dxy = d[valid, :2]
        dist = torch.linalg.norm(dxy, dim=1)
        az = torch.atan2(dxy[:, 1], dxy[:, 0]) - yaw0
        az = (az + math.pi) % (2 * math.pi) - math.pi
        bins = torch.round((az + fov) / beam_spacing).long()
        inb = (bins >= 0) & (bins < B) & (az.abs() <= fov + 0.5 * beam_spacing)
        sector = torch.full((B,), rc.scan_max_range, device=dev)
        sector.scatter_reduce_(0, bins[inb], dist[inb], reduce="amin", include_self=True)

        # 중심빔 근사: 수평에 가까운 채널만 남기고 섹터 중심각에 가장 가까운 레이 거리
        elev = torch.atan2(d[valid, 2], dist.clamp(min=1e-6))
        central = elev.abs() <= math.radians(0.8)
        center_ray = torch.full((B,), rc.scan_max_range, device=dev)
        az_c, dist_c, bins_c = az[central], dist[central], bins[central]
        for b in range(B):
            m = (bins_c == b)
            if m.any():
                cang = -fov + b * beam_spacing
                k = (az_c[m] - cang).abs().argmin()
                center_ray[b] = dist_c[m][k]

        # ---- 해석적 2D 레이캐스트: 학습 윈도(±64) vs 전체 윈도 ----
        ang_w = yaw.unsqueeze(1) + env._beam_angles.unsqueeze(0)
        proj = env._vt.project(la, env._track_type)
        full_w = env._vt.length.max().item() // 2      # 전 트랙 커버
        analytic64 = env._vt.raycast(la, ang_w, env._track_type, proj["idx"],
                                     rc.raycast_half_window, rc.scan_max_range)[0]
        analyticF = env._vt.raycast(la, ang_w, env._track_type, proj["idx"],
                                    int(full_w), rc.scan_max_range)[0]

        def acc(key, lhs, rhs):
            hit = (lhs < rc.scan_max_range * 0.99) & (rhs < rc.scan_max_range * 0.99)
            miss = (lhs >= rc.scan_max_range * 0.99) & (rhs >= rc.scan_max_range * 0.99)
            errs.setdefault(key, []).append((lhs[hit] - rhs[hit]).abs())
            agree.setdefault(key, [0, 0])
            agree[key][0] += int((hit | miss).sum()); agree[key][1] += B

        acc("A. 섹터min vs 해석(윈도64) [학습 obs 와의 실제 갭]", sector, analytic64)
        acc("B. 섹터min vs 해석(전체윈도) [윈도 효과 제거]", sector, analyticF)
        acc("C. 중심빔   vs 해석(전체윈도) [순수 지오메트리]", center_ray, analyticF)

        if step % 50 == 0 and errs:
            e = torch.cat(errs["C. 중심빔   vs 해석(전체윈도) [순수 지오메트리]"])
            print(f"[{step}] C(지오메트리) mean|err|={e.mean():.4f}m p95={torch.quantile(e, 0.95):.4f}m")

    print("=" * 74)
    print(f"[RESULT] track={env._val_track} steps={args.steps}")
    for key in sorted(errs):
        e = torch.cat(errs[key])
        a = agree[key]
        print(f"  {key}")
        print(f"     mean|err|={e.mean():.4f}m  p95={torch.quantile(e, 0.95):.4f}m "
              f"max={e.max():.4f}m  히트/미스일치={a[0]/max(a[1],1)*100:.1f}%  (빔 {len(e)}개)")
    eC = torch.cat(errs["C. 중심빔   vs 해석(전체윈도) [순수 지오메트리]"])
    ok = eC.mean() < 0.05
    print(f"  판정(C 기준): {'PASS — 해석 모델이 실제 3D LiDAR 지오메트리와 일치' if ok else 'FAIL — 지오메트리 모델 자체에 편차'}")
    print("  해석: A-B 차이=탐색 윈도 부족分, B-C 차이=섹터 min 처리 편향分")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
