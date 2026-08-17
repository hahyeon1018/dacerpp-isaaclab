#!/usr/bin/env python3
"""감지 박스 통합 헤드리스 스모크: 상대차 박스 검출/가시성/시각프림/NaN/obs_dim."""
from __future__ import annotations
import argparse, os
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=60)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import omni.usd
from pxr import UsdGeom
from dacerpp_lab.env_cfg import RacingEnvCfg
from dacerpp_lab.racing_env import RacingEnv
from dacerpp_lab.track_field import TrackField

RES = os.path.join(os.path.dirname(__file__), "..", "generated", "smoke_detbox.txt")
_L = []
def out(s):
    _L.append(s); open(RES, "w").write("\n".join(_L) + "\n")


def main():
    pd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = RacingEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.project_dir = pd
    cfg.track_field.num_envs = args.num_envs
    cfg.track_field.num_procedural = 0; cfg.track_field.use_f1tenth = False  # comp-only
    cfg.wall_visuals = True                 # 시각 프림 생성 검사
    cfg.racing.obstacles_enabled = False    # 감지박스만 격리 검사
    cfg.racing.obs_noise = False
    cfg.episode_length_s = 1.0e9
    cfg.observation_space = 2 * cfg.racing.obs_dim()
    cfg.scene.env_spacing = TrackField(cfg.track_field).suggested_env_spacing
    env = RacingEnv(cfg); dev = env.device; rc = cfg.racing
    N = env.num_envs; B = rc.n_beams
    obs_dim = rc.obs_dim()
    # 감지박스는 스캔/가시성 '내용'만 바꾸므로 차원은 obs_dim() 공식과 일치해야 한다.
    # 상수(58)로 박아두면 curv_lookahead 를 늘릴 때마다 이 스모크가 헛돈다.
    dim_expect = (rc.n_beams + 1 + 2 + 1 + len(rc.curv_lookahead)
                  + (1 + len(rc.curv_lookahead)) + 2 * rc.act_hist_len
                  + rc.n_opp_feats + 2)
    out(f"[obs_dim] {obs_dim} (감지박스는 차원 불변 — 구성식 기대 {dim_expect})")

    # 시각 프림 확인
    stage = omni.usd.get_context().get_stage()
    npr = 0; sz = None
    for i in range(N):
        for car in ("A", "B"):
            pr = stage.GetPrimAtPath(f"/World/envs/env_{i}/Car_{car}/DetBox")
            if pr.IsValid():
                npr += 1
                if sz is None:
                    from pxr import UsdGeom as UG
                    xf = UG.Xformable(pr).GetOrderedXformOps()
                    sz = [tuple(op.Get()) for op in xf if "scale" in op.GetName()]
    out(f"[시각프림] DetBox {npr}개 생성 (기대 {2*N}=차량2×env{N}), scale 샘플 {sz}")

    obs, _ = env.reset()
    vis_any = 0; nan = False
    opp_i = B + 1 + 2 + 1 + len(rc.curv_lookahead) + (1 + len(rc.curv_lookahead)) + 2 * rc.act_hist_len
    aa = torch.zeros(N, 2, device=dev); ab = torch.zeros(N, 2, device=dev)
    for it in range(args.steps):
        aa[:, 0] = 0.5; ab[:, 0] = 0.5     # 전진(상대와 조우 유도)
        obs, _, _, _, _ = env.step(torch.cat([aa, ab], 1))
        if torch.isnan(obs["car_a"]).any() or torch.isnan(obs["car_b"]).any():
            nan = True; break
        # 가시성 플래그 = opp 특징의 5번째 (index opp_i+4)
        va = obs["car_a"][:, opp_i + 4]; vb = obs["car_b"][:, opp_i + 4]
        vis_any += int((va > 0.5).sum() + (vb > 0.5).sum())
    out(f"[가시성] opp 특징 index={opp_i}, {args.steps}스텝 누적 '상대 검출(visible=1)' {vis_any}회 "
        f"(>0 = 감지박스로 상대 검출 작동)")
    # 검출 시 스캔에 박스 반영: 상대가 보이는 env 에서 근접반사(<3.5m) 존재?
    scan_a = obs["car_a"][:, :B]
    out(f"[스캔] 관측 스캔 [{float(scan_a.min()):.2f},{float(scan_a.max()):.2f}] (0~1), "
        f"근접반사(<0.35) env {int((scan_a < 0.35).any(1).sum())}/{N}")
    out(f"[NaN] {'발생' if nan else '없음'}")
    ok = (not nan) and (obs_dim == dim_expect) and (npr == 2 * N) and (vis_any > 0)
    out(f"\n{'✅ 감지박스 스모크 통과' if ok else '❌ 확인 필요'}")
    env.close()

main()
simulation_app.close()
