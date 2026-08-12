#!/usr/bin/env python3
"""조향 진동 원인 직접 규명: 같은 정책/트랙에서 해석관측 vs LiDAR관측 조향 시계열 비교.

test_lidar 진동이 (a)스캔 파이프라인 갭 때문인지 (b)정책 고유 뱅뱅(트랙폭 가시성)인지
가른다. 매 스텝 두 관측으로 각각 조향을 뽑아 std/부호반전율/자기상관을 비교한다.
"""
from __future__ import annotations
import argparse, math, os
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, default="dacerpp_runs/20260731")
parser.add_argument("--track", type=str, default="comp_lobby_0730")
parser.add_argument("--steps", type=int, default=800)
parser.add_argument("--channels", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np, torch
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from dacerpp_lab.env_cfg import RacingEnvCfg
from dacerpp_lab.racing_env import RacingEnv
from dacerpp_lab.track_field import TrackField
from dacerpp_lab.vectorized_track import overlay_opponent
from dacer_pp import DACERppConfig, DACERpp, aggressive_pow

PD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WALL = "/World/TrackWalls"; Z = (0.03, 0.28)
RES = os.path.join(PD, "generated", "steer_osc.txt"); _L = []
def out(s): _L.append(s); open(RES, "w").write("\n".join(_L) + "\n")


class LEnv(RacingEnv):
    def _setup_scene(self):
        super()._setup_scene()
        import omni.usd
        from pxr import UsdGeom, UsdPhysics
        if not hasattr(self, "_field"): self._field = TrackField(self.cfg.track_field)
        g = int(self._field.env_track_type[0])
        cl = self._field.centerlines[g]; hw = self._field.half_widths[g]
        psi = self._field.features[g]["psi"]; nrm = np.stack([-np.sin(psi), np.cos(psi)], 1)
        h = float(self.cfg.track_field.params.wall_height)
        o = self._terrain.env_origins[0].detach().cpu().numpy()
        pts, idx = [], []
        for side in (+1., -1.):
            b = cl + side*nrm*hw[:,None]; n=len(b); base=len(pts)
            pts += [(float(x+o[0]), float(y+o[1]), float(o[2])) for x,y in b]
            pts += [(float(x+o[0]), float(y+o[1]), float(o[2]+h)) for x,y in b]
            for i in range(n):
                j=(i+1)%n; idx += [base+i,base+j,base+n+j, base+i,base+n+j,base+n+i]
        st = omni.usd.get_context().get_stage()
        m = UsdGeom.Mesh.Define(st, WALL); m.CreatePointsAttr(pts)
        m.CreateFaceVertexCountsAttr([3]*(len(idx)//3)); m.CreateFaceVertexIndicesAttr(idx)
        m.CreateDoubleSidedAttr(True); UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("none")
        lc = RayCasterCfg(prim_path="/World/envs/env_0/Car_B/base_link", mesh_prim_paths=[WALL],
            offset=RayCasterCfg.OffsetCfg(pos=(0.,0.,0.062)), ray_alignment="base",
            pattern_cfg=patterns.LidarPatternCfg(channels=args.channels,
                vertical_fov_range=(-10.,10.), horizontal_fov_range=(-135.,135.), horizontal_res=1.0),
            max_distance=float(self.cfg.racing.scan_max_range))
        self._lidar = RayCaster(lc); self.scene.sensors["lidar"] = self._lidar


def lidar_scan(env, car, rc, B, fov, bs):
    ld = env.scene.sensors["lidar"]; hits = ld.data.ray_hits_w[0]; sp = ld.data.pos_w[0]
    _, yaw, _ = env._car_state(car); d = hits - sp
    v = torch.isfinite(d).all(1) & (hits[:,2]>Z[0]) & (hits[:,2]<Z[1])
    dxy = d[v,:2]; dist = torch.linalg.norm(dxy, dim=1)
    az = torch.atan2(dxy[:,1], dxy[:,0]) - yaw[0]; az = (az+math.pi)%(2*math.pi)-math.pi
    bins = torch.round((az+fov)/bs).long(); inb = (bins>=0)&(bins<B)&(az.abs()<=fov+0.5*bs)
    scan = torch.full((B,), rc.scan_max_range, device=env.device)
    ai, di, bi = az[inb], dist[inb], bins[inb]
    for b in range(B):
        m = bi==b
        if m.any(): scan[b] = di[m][(ai[m]-(-fov+b*bs)).abs().argmin()]
    return scan


def sig_stats(s, dt):
    s = np.asarray(s); rev = np.sum(np.diff(np.sign(s)) != 0) / (len(s)*dt)
    d = np.diff(s)
    return dict(std=s.std(), rev_hz=rev, dstd=d.std(),
                ac1=np.corrcoef(s[1:], s[:-1])[0,1] if s.std()>1e-6 else 0.0)


def main():
    cfg = RacingEnvCfg()
    cfg.scene.num_envs = 1; cfg.project_dir = PD; cfg.track_field.num_envs = 1
    cfg.wall_visuals = False; cfg.racing.obs_noise = False
    cfg.racing.obstacles_enabled = False
    cfg.track_field.pinned_tracks = (args.track,)
    cfg.track_field.num_procedural = 0; cfg.track_field.use_f1tenth = False
    cfg.observation_space = 2*cfg.racing.obs_dim()
    cfg.scene.env_spacing = TrackField(cfg.track_field).suggested_env_spacing
    env = LEnv(cfg); dev = env.device; rc = cfg.racing
    B = rc.n_beams; fov = rc.scan_fov; bs = 2.0*fov/(B-1)
    c = DACERppConfig(obs_dim=rc.obs_dim(), act_dim=2, risk=aggressive_pow(1.3),
                      device=str(dev), use_compile=False, seed=1)
    ag = DACERpp(c).load(os.path.join(PD, args.ckpt_dir, "pow.pt"))
    obs, _ = env.reset(); oa, ob = obs["car_a"], obs["car_b"]
    st_lidar, st_ana, mae = [], [], []
    for it in range(args.steps):
        wall = lidar_scan(env, env.car_b, rc, B, fov, bs)
        ls, ys, _ = env._car_state(env.car_b); lo,_,_ = env._car_state(env.car_a)
        rel = lo-ls; cy,sy = torch.cos(-ys), torch.sin(-ys)
        bx = rel[:,0]*cy-rel[:,1]*sy; by = rel[:,0]*sy+rel[:,1]*cy
        wall = overlay_opponent(wall.unsqueeze(0), env._beam_angles,
                                torch.stack([bx,by],1), rc.scan_max_range, rc.opp_radius)
        o_lidar = ob.clone(); o_lidar[:, :B] = (wall/rc.scan_max_range).clamp(0,1)
        a_lidar = ag.act_batch(o_lidar, deterministic=True)   # LiDAR 관측 조향
        a_ana = ag.act_batch(ob, deterministic=True)          # 해석 관측 조향
        st_lidar.append(float(a_lidar[0,1])); st_ana.append(float(a_ana[0,1]))
        mae.append(float((o_lidar[0,:B]-ob[0,:B]).abs().mean())*rc.scan_max_range)
        a_a = ag.act_batch(oa, deterministic=True)            # Car A(cvar 아님-여기선 pow로 무방, 상대역)
        nobs,_,_,_,_ = env.step(torch.cat([a_a, a_lidar],1))  # LiDAR 조향으로 실제 주행
        oa, ob = nobs["car_a"], nobs["car_b"]
    dt = env.step_dt
    L = sig_stats(st_lidar, dt); A = sig_stats(st_ana, dt)
    out(f"[트랙] {args.track}  스텝 {args.steps}  dt={dt*1000:.0f}ms  scan MAE {np.mean(mae):.3f}m")
    out(f"[조향 std]        LiDAR관측 {L['std']:.3f}  해석관측 {A['std']:.3f}  (클수록 큰 조향폭)")
    out(f"[부호반전 Hz]     LiDAR관측 {L['rev_hz']:.1f}  해석관측 {A['rev_hz']:.1f}  (진동 주파수)")
    out(f"[스텝간 조향변화]  LiDAR관측 {L['dstd']:.3f}  해석관측 {A['dstd']:.3f}  (고주파성분=진동)")
    out(f"[조향 자기상관]   LiDAR관측 {L['ac1']:.2f}  해석관측 {A['ac1']:.2f}  (낮을수록 뱅뱅)")
    ratio = L['dstd']/max(A['dstd'],1e-6)
    out(f"\n[판정] 스텝간 조향변화 비율(LiDAR/해석) = {ratio:.2f}")
    out("  >1.5 = LiDAR 스캔이 진동 유발(스캔갭 원인) / ≈1 = 정책 고유 뱅뱅(트랙폭 가시성)")
    env.close()

main()
simulation_app.close()
