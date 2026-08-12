#!/usr/bin/env python3
"""장애물 통합 헤드리스 스모크: 배치/스캔반영/충돌종료/NaN 검증."""
from __future__ import annotations
import argparse, os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=250)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from dacerpp_lab.env_cfg import RacingEnvCfg
from dacerpp_lab.racing_env import RacingEnv

RESULT = os.path.join(os.path.dirname(__file__), "..", "generated", "smoke_obstacles.txt")
_lines = []
def out(s):
    _lines.append(s)
    with open(RESULT, "w") as f:
        f.write("\n".join(_lines) + "\n")


def main():
    pd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = RacingEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.project_dir = pd
    cfg.track_field.num_envs = args.num_envs
    cfg.wall_visuals = False
    cfg.episode_length_s = 1.0e9
    cfg.observation_space = 2 * cfg.racing.obs_dim()
    from dacerpp_lab.track_field import TrackField
    cfg.scene.env_spacing = TrackField(cfg.track_field).suggested_env_spacing
    env = RacingEnv(cfg)
    dev = env.device
    N = env.num_envs
    rc = cfg.racing

    obs, _ = env.reset()
    # 배치 통계 — 개수 분포(0~3 균등이어야)
    act = env._obs_active
    n_per = act.sum(1).long()
    dist = [int((n_per == c).sum()) for c in range(rc.obstacle_max + 1)]
    frac = [d / N for d in dist]
    out(f"[배치] 개수 분포(0/1/2/3 %) = {[f'{f*100:.0f}' for f in frac]}  평균 {float(n_per.float().mean()):.2f}  "
        f"(목표 가중치 {[f'{w*100:.0f}' for w in rc.obstacle_count_weights]}, 3개 확률 {frac[3]*100:.0f}%)")
    # 활성 장애물 배치 통과성: 중심 횡오프셋 + 외접반경이 반폭 안에 들어오는가 샘플 확인
    sizes = env._obs_size[act]
    out(f"[크기] {float(sizes.min())*100:.0f}~{float(sizes.max())*100:.0f}cm "
          f"(≤50cm), 모양분포 box/tri/cyl = "
          f"{[int((env._obs_shape[act]==s).sum()) for s in (0,1,2)]}")
    # 간격 검증: 장애물↔장애물 중심거리 최소, 차↔장애물 최소
    cen = env._obs_center; a = env._obs_active                    # (N,K,2),(N,K)
    oo_min = 999.0
    for i in range(N):
        idx = a[i].nonzero(as_tuple=False).squeeze(-1)
        if len(idx) >= 2:
            c = cen[i, idx]
            dm = torch.cdist(c, c) + torch.eye(len(idx), device=dev) * 999
            oo_min = min(oo_min, float(dm.min()))
    la, _, _ = env._car_state(env.car_a); lb, _, _ = env._car_state(env.car_b)
    co_min = 999.0
    for car in (la, lb):
        for i in range(N):
            idx = a[i].nonzero(as_tuple=False).squeeze(-1)
            if len(idx):
                d = (cen[i, idx] - car[i]).norm(dim=1).min()
                co_min = min(co_min, float(d))
    out(f"[간격] 장애물↔장애물 중심 최소거리 {oo_min:.2f}m (설정 {rc.obstacle_min_sep}m 이상) | "
        f"차↔장애물 스폰시 최소 {co_min:.2f}m (충돌margin {rc.obstacle_hit_margin}m 초과여야)")
    # 장애물-벽 간격(한쪽 통과 보장): 중심선 투영 -> 근접벽/반대벽(통과 레인) 간격
    ai, ak = a.nonzero(as_tuple=True)
    ocen = env._obs_center[ai, ak]; orad = env._obs_rad[ai, ak]; ott = env._track_type[ai]
    pr = env._vt.project(ocen, ott)
    hwl = env._vt.lookahead_width(pr["idx"], ott, (0,)).squeeze(1)
    lat_abs = pr["lateral"].abs()
    near_gap = hwl - lat_abs - orad          # 장애물이 기댄(근접) 벽 간격 (외접반경 기준, 보수적)
    far_gap = hwl + lat_abs - orad           # 반대(통과 레인) 벽 간격 — 여기로 차가 통과
    pc = rc.obstacle_pass_clear; wmc = rc.obstacle_wall_min_clear
    tol = 0.04                               # 곡선부 국소법선 slack(~4cm) 여유
    n_act = int(len(orad))
    hug = int((near_gap < pc - 1e-3).sum())  # 근접벽<pass_clear = 예전 '양쪽 보장'이면 불가한 '벽붙임'
    ok_far = float(far_gap.min()) >= pc - tol
    ok_near = float(near_gap.min()) >= wmc - tol
    out(f"[장애물-벽 간격] 통과레인(far) 최소 {float(far_gap.min()):.2f}m (>= pass_clear {pc}) "
        f"{'OK' if ok_far else '부족'} | 근접벽(near) 최소 {float(near_gap.min()):.2f}m "
        f"(>= wall_min_clear {wmc}) {'OK' if ok_near else '부족'}")
    out(f"[한쪽 보장] '벽붙임'(근접벽<pass_clear) 장애물 {hug}/{n_act}개 "
        f"({(hug/max(n_act,1))*100:.0f}%) — >0 = 새 한쪽-보장 배치 동작(예전 로직이면 0)")

    # 전진 주행(장애물/벽 조우 유도): throttle+, steer 작게 랜덤
    obs_term_a = obs_term_b = 0
    scan_hit_frames = 0
    nan = False
    aa = torch.zeros(N, 2, device=dev); ab = torch.zeros(N, 2, device=dev)
    for it in range(args.steps):
        aa[:, 0] = 0.5 + 0.2 * torch.randn(N, device=dev)   # throttle
        aa[:, 1] = 0.3 * torch.randn(N, device=dev)         # steer
        ab[:, 0] = 0.5 + 0.2 * torch.randn(N, device=dev)
        ab[:, 1] = 0.3 * torch.randn(N, device=dev)
        obs, _, _, _, _ = env.step(torch.cat([aa, ab], dim=1))
        di = env.dual_info()
        if torch.isnan(obs["car_a"]).any() or torch.isnan(obs["car_b"]).any():
            nan = True; break
        obs_term_a += int(di["causes"]["obs_a"].sum())
        obs_term_b += int(di["causes"]["obs_b"].sum())
        # 스캔에 장애물 반영 확인: 장애물 보유 env 에서 벽전용 대비 정책스캔이 더 짧은 빔 존재?
        # (관측 첫 n_beams 가 정규화 스캔) — 근사적으로 매우 짧은 빔 비율로 대체 확인
    scan_a = obs["car_a"][:, :rc.n_beams]
    very_short = (scan_a < 0.35).any(dim=1)   # <3.5m 반사 있는 env
    out(f"[스캔] 관측 스캔 정상 범위 [{float(scan_a.min()):.2f},{float(scan_a.max()):.2f}] "
          f"(0~1 정규화), 근접반사 보유 env {int(very_short.sum())}/{N}")
    out(f"[충돌종료] {args.steps}스텝 누적 obs_a={obs_term_a} obs_b={obs_term_b} "
          f"(0 초과 = 장애물 충돌 종료 작동)")
    out(f"[NaN] {'발생' if nan else '없음'}")
    # 재스폰 시 장애물 재추첨 확인: 강제 env 리셋 후 배치가 달라지는가
    before = env._obs_center.clone()
    env._reset_idx(torch.arange(N, device=dev))
    changed = (env._obs_center - before).abs().sum() > 1e-3
    out(f"[재스폰] 리셋 후 장애물 배치 변경: {bool(changed)} (도메인 랜덤화 작동)")
    ok = (not nan) and (obs_term_a + obs_term_b > 0) and bool(changed) \
         and ok_far and ok_near and (hug > 0)
    out(f"\n{'✅ 스모크 통과' if ok else '❌ 확인 필요'}")
    env.close()


main()
simulation_app.close()
