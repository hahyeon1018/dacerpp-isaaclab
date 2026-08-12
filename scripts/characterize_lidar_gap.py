#!/usr/bin/env python3
"""해석 스캔(학습/test.py) vs 3D LiDAR 파이프라인 스캔(test_lidar/실차) 차이 구조 측정.

test_lidar 조향 진동의 원인이 '스캔 파이프라인 갭'인지, 그 갭의 크기/구조(빔별,
시간 상관)를 정량화해 학습 노이즈 모델 보정에 쓴다. 정책 불필요(전진 주행으로 관측만).
"""
from __future__ import annotations
import argparse, math, os
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--channels", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from dacerpp_lab.env_cfg import RacingEnvCfg
from dacerpp_lab.racing_env import RacingEnv
from dacerpp_lab.track_field import TrackField

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WALL_PRIM = "/World/TrackWalls"
Z_BAND = (0.03, 0.28)
RESULT = os.path.join(PROJECT_DIR, "generated", "lidar_gap.txt")
_L = []
def out(s):
    _L.append(s); open(RESULT, "w").write("\n".join(_L) + "\n")


class LidarEnv(RacingEnv):
    def _setup_scene(self):
        super()._setup_scene()
        import omni.usd
        from pxr import UsdGeom, UsdPhysics
        if not hasattr(self, "_field"):
            self._field = TrackField(self.cfg.track_field)
        g = int(self._field.env_track_type[0])
        cl = self._field.centerlines[g]; hw = self._field.half_widths[g]
        psi = self._field.features[g]["psi"]
        nrm = np.stack([-np.sin(psi), np.cos(psi)], 1)
        h = float(self.cfg.track_field.params.wall_height)
        o = self._terrain.env_origins[0].detach().cpu().numpy()
        pts, idx = [], []
        for side in (+1., -1.):
            b = cl + side * nrm * hw[:, None]; n = len(b); base = len(pts)
            pts += [(float(x+o[0]), float(y+o[1]), float(o[2])) for x, y in b]
            pts += [(float(x+o[0]), float(y+o[1]), float(o[2]+h)) for x, y in b]
            for i in range(n):
                j = (i+1) % n
                idx += [base+i, base+j, base+n+j, base+i, base+n+j, base+n+i]
        stage = omni.usd.get_context().get_stage()
        m = UsdGeom.Mesh.Define(stage, WALL_PRIM)
        m.CreatePointsAttr(pts); m.CreateFaceVertexCountsAttr([3]*(len(idx)//3))
        m.CreateFaceVertexIndicesAttr(idx); m.CreateDoubleSidedAttr(True)
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("none")
        lc = RayCasterCfg(
            prim_path="/World/envs/env_0/Car_B/base_link", mesh_prim_paths=[WALL_PRIM],
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.062)), ray_alignment="base",
            pattern_cfg=patterns.LidarPatternCfg(channels=args.channels,
                vertical_fov_range=(-10., 10.), horizontal_fov_range=(-135., 135.),
                horizontal_res=1.0), max_distance=float(self.cfg.racing.scan_max_range))
        self._lidar = RayCaster(lc); self.scene.sensors["lidar"] = self._lidar


def lidar_scan(env, car, rc, B, fov, bs):
    lidar = env.scene.sensors["lidar"]
    hits = lidar.data.ray_hits_w[0]; spos = lidar.data.pos_w[0]
    _, yaw, _ = env._car_state(car)
    d = hits - spos
    valid = torch.isfinite(d).all(1) & (hits[:,2] > Z_BAND[0]) & (hits[:,2] < Z_BAND[1])
    dxy = d[valid,:2]; dist = torch.linalg.norm(dxy, dim=1)
    az = torch.atan2(dxy[:,1], dxy[:,0]) - yaw[0]
    az = (az+math.pi) % (2*math.pi) - math.pi
    bins = torch.round((az+fov)/bs).long()
    inb = (bins>=0)&(bins<B)&(az.abs()<=fov+0.5*bs)
    scan = torch.full((B,), rc.scan_max_range, device=env.device)
    az_i, dist_i, bins_i = az[inb], dist[inb], bins[inb]
    for b in range(B):
        m = bins_i==b
        if m.any():
            cang = -fov + b*bs
            scan[b] = dist_i[m][(az_i[m]-cang).abs().argmin()]
    return scan


def main():
    cfg = RacingEnvCfg()
    cfg.scene.num_envs = 1; cfg.project_dir = PROJECT_DIR; cfg.track_field.num_envs = 1
    cfg.wall_visuals = False; cfg.racing.obs_noise = False
    cfg.racing.obstacles_enabled = False        # 순수 벽 스캔 갭만 측정
    # 실측 대회 코스에서 측정 (comp_lobby)
    cfg.track_field.pinned_tracks = ("comp_lobby_0730",)
    cfg.track_field.num_procedural = 0; cfg.track_field.use_f1tenth = False
    cfg.observation_space = 2*cfg.racing.obs_dim()
    cfg.scene.env_spacing = TrackField(cfg.track_field).suggested_env_spacing
    env = LidarEnv(cfg); dev = env.device; rc = cfg.racing
    B = rc.n_beams; fov = rc.scan_fov; bs = 2.0*fov/(B-1)

    env.reset()
    sdiff = np.full((args.steps, B), np.nan)     # 프레임별 부호있는 갭(무효=nan)
    aa = torch.zeros(1,2,device=dev)
    for it in range(args.steps):
        aa[:,0] = 0.4; aa[:,1] = 0.15*math.sin(it*0.1)   # 완만한 전진+조향
        env.step(torch.cat([aa,aa],1))
        lb, yb, _ = env._car_state(env.car_b)
        pb = env._vt.project(lb, env._track_type)
        ang = yb.unsqueeze(1) + env._beam_angles.unsqueeze(0)
        ana = env._vt.raycast(lb, ang, env._track_type, pb["idx"],
                              rc.raycast_half_window, rc.scan_max_range)[0]  # (B,)
        lid = lidar_scan(env, env.car_b, rc, B, fov, bs)                     # (B,)
        both = (ana < rc.scan_max_range-0.1) & (lid < rc.scan_max_range-0.1)
        dd = (lid - ana).cpu().numpy(); bth = both.cpu().numpy()
        sdiff[it, bth] = dd[bth]
    absd = np.abs(sdiff); alld = absd[np.isfinite(absd)]
    out(f"[측정] {args.steps}스텝 (comp_lobby_0730), 유효 빔-샘플 {len(alld)}개")
    out(f"[스캔 갭 |LiDAR-해석|] 평균 {alld.mean():.3f}m  중앙 {np.median(alld):.3f}m  "
        f"p90 {np.percentile(alld,90):.3f}m  p99 {np.percentile(alld,99):.3f}m  최대 {alld.max():.3f}m")
    out(f"[비율] >5cm {np.mean(alld>0.05)*100:.0f}%  >10cm {np.mean(alld>0.10)*100:.0f}%  "
        f">20cm {np.mean(alld>0.20)*100:.0f}%")
    # 시간 상관(lag-1): 프레임 간 갭이 구조적(상관 큼)인지 백색(≈0)인지 -> 스무딩 대상 판별
    ac = []
    for b in range(B):
        col = sdiff[:, b]; m = np.isfinite(col[1:]) & np.isfinite(col[:-1])
        if m.sum() > 20:
            x, y = col[1:][m], col[:-1][m]
            if x.std() > 1e-6 and y.std() > 1e-6:
                ac.append(np.corrcoef(x, y)[0, 1])
    out(f"[시간상관 lag-1] 평균 {np.nanmean(ac):.2f} (1=완전구조적/느린드리프트, 0=프레임간 백색점프)")
    # 프레임 간 변화(고주파 성분): |Δgap|/프레임 = 정책이 스텝마다 겪는 스캔 요동
    fd = np.abs(np.diff(sdiff, axis=0)); fdv = fd[np.isfinite(fd)]
    out(f"[프레임간 |Δ갭|] 평균 {fdv.mean():.3f}m  p90 {np.percentile(fdv,90):.3f}m "
        f"(이게 크면 스텝마다 스캔이 튀어 조향이 진동)")
    out(f"[대조] 학습 백색노이즈 scan_noise_std={rc.scan_noise_std}m -> 프레임간 |Δ|≈{rc.scan_noise_std*1.13:.3f}m")
    env.close()

main()
simulation_app.close()
