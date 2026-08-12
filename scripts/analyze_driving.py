#!/usr/bin/env python3
"""학습 정책의 주행 거동 계측 분석 (headless).

test.py 와 동일한 환경/물리에서 정책을 돌리며 스텝 단위로
조향(명령/실제 조인트각/실측 요레이트), 속도, 곡률, 횡위치, 감속을 기록해
언더스티어·과감속·라인 선택·제동 시점을 정량화한다.

실행:
  python -u scripts/analyze_driving.py --ckpt_dir dacerpp_runs/20260709_1 --steps 3000 --headless
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=16, help="16 = 트랙 16종 각 1개")
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--cvar_eta", type=float, default=0.5)
parser.add_argument("--pow_eta", type=float, default=1.2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

_cvar = os.path.join(args.ckpt_dir, "cvar.pt")
_pow = os.path.join(args.ckpt_dir, "pow.pt")
assert os.path.isfile(_cvar) and os.path.isfile(_pow), args.ckpt_dir

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from dacerpp_lab.env_cfg import RacingEnvCfg
from dacerpp_lab.racing_env import RacingEnv
from dacer_pp import DACERppConfig, DACERpp, conservative_cvar, aggressive_pow

MU_G = 0.65 * 9.81   # main() 에서 cfg.tire.mu 로 갱신
L_WB = 0.33   # 실측 축거


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = RacingEnvCfg()
    global MU_G
    MU_G = cfg.tire.mu * 9.81   # 마찰 한계 횡가속 (env_cfg.TireModelCfg.mu 기준)
    cfg.scene.num_envs = args.num_envs
    cfg.project_dir = project_dir
    cfg.track_field.num_envs = args.num_envs
    cfg.wall_visuals = False
    cfg.racing.obs_noise = False
    cfg.episode_length_s = 1.0e9
    cfg.observation_space = 2 * cfg.racing.obs_dim()
    from dacerpp_lab.track_field import TrackField
    cfg.scene.env_spacing = TrackField(cfg.track_field).suggested_env_spacing

    env = RacingEnv(cfg)
    rc = cfg.racing
    dev = env.device
    N = env.num_envs
    obs_dim, act_dim = rc.obs_dim(), rc.act_dim()

    def make_agent(risk, seed):
        c = DACERppConfig(obs_dim=obs_dim, act_dim=act_dim, risk=risk,
                          device=str(dev), use_compile=False, seed=seed)
        return DACERpp(c)

    ag_a = make_agent(conservative_cvar(args.cvar_eta), 0).load(_cvar)
    ag_b = make_agent(aggressive_pow(args.pow_eta), 1).load(_pow)
    print(f"[ckpt] {args.ckpt_dir}: cvar step={ag_a.step} alpha={ag_a.log_alpha.exp().item():.3f} | "
          f"pow step={ag_b.step} alpha={ag_b.log_alpha.exp().item():.3f}")

    obs, _ = env.reset()
    oa, ob = obs["car_a"], obs["car_b"]

    # ---- 로그 버퍼 (car_a=CVaR / car_b=Pow 분리) ----
    T = args.steps
    L = {k: torch.zeros(T, N, device=dev) for k in
         ["v_a", "v_b", "cmd_st_a", "cmd_st_b", "cmd_th_a", "cmd_th_b",
          "joint_a", "joint_b", "yr_a", "yr_b", "lat_a", "lat_b", "hw_a", "hw_b",
          "k0_a", "k0_b", "k15_a", "k15_b", "k60_a", "k60_b", "k90_a", "k90_b",
          "term_a", "term_b"]}

    prev_term_a = torch.zeros(N, dtype=torch.bool, device=dev)
    prev_term_b = torch.zeros(N, dtype=torch.bool, device=dev)
    fresh_a = torch.ones(N, dtype=torch.bool, device=dev)   # 리스폰 직후(계측 제외용)
    fresh_b = torch.ones(N, dtype=torch.bool, device=dev)

    for t in range(T):
        aa = ag_a.act_batch(oa, deterministic=True)
        ab = ag_b.act_batch(ob, deterministic=True)

        # 물리 스텝 "이전" 상태에서 곡률/위치 기록 (행동과 같은 시점)
        la, ya, va = env._car_state(env.car_a)
        lb, yb, vb = env._car_state(env.car_b)
        pa = env._vt.project(la, env._track_type)
        pb = env._vt.project(lb, env._track_type)
        for tag, p in (("a", pa), ("b", pb)):
            k = env._vt.lookahead_curvature(p["idx"], env._track_type, (0, 15, 60, 90))
            L[f"k0_{tag}"][t], L[f"k15_{tag}"][t] = k[:, 0], k[:, 1]
            L[f"k60_{tag}"][t], L[f"k90_{tag}"][t] = k[:, 2], k[:, 3]
            hw = env._vt.lookahead_width(p["idx"], env._track_type, (0,)).squeeze(1)
            L[f"hw_{tag}"][t] = hw
            L[f"lat_{tag}"][t] = p["lateral"]
        L["v_a"][t], L["v_b"][t] = va, vb
        L["cmd_st_a"][t], L["cmd_th_a"][t] = aa[:, 0], aa[:, 1]
        L["cmd_st_b"][t], L["cmd_th_b"][t] = ab[:, 0], ab[:, 1]
        L["joint_a"][t] = env.car_a.data.joint_pos[:, env._steer_ids_a].mean(dim=1)
        L["joint_b"][t] = env.car_b.data.joint_pos[:, env._steer_ids_b].mean(dim=1)
        L["yr_a"][t] = env.car_a.data.root_ang_vel_w[:, 2]
        L["yr_b"][t] = env.car_b.data.root_ang_vel_w[:, 2]
        L["term_a"][t] = fresh_a.float()   # 임시: 아래서 실제 term 으로 덮음
        L["term_b"][t] = fresh_b.float()

        next_obs, _, _, _, _ = env.step(torch.cat([aa, ab], dim=1))
        di = env.dual_info()
        oa, ob = next_obs["car_a"], next_obs["car_b"]
        L["term_a"][t] = di["term_a"].float() + 2.0 * fresh_a.float()  # 1=term, 2=fresh, 3=둘다
        L["term_b"][t] = di["term_b"].float() + 2.0 * fresh_b.float()
        fresh_a = di["term_a"].clone()
        fresh_b = di["term_b"].clone()

    # ================= 분석 =================
    def report(tag, name):
        v = L[f"v_{tag}"]; cmd = L[f"cmd_st_{tag}"]; th = L[f"cmd_th_{tag}"]
        joint = L[f"joint_{tag}"]; yr = L[f"yr_{tag}"]
        lat = L[f"lat_{tag}"]; hw = L[f"hw_{tag}"]
        k0 = L[f"k0_{tag}"]; k15 = L[f"k15_{tag}"]; k60 = L[f"k60_{tag}"]; k90 = L[f"k90_{tag}"]
        tm = L[f"term_{tag}"]
        ok = (tm < 0.5)                                   # term/리스폰 직후 스텝 제외
        ok[1:] &= (tm[:-1] < 0.5)                          # term 다음 스텝(텔레포트)도 제외
        dt = env.step_dt
        dv = torch.zeros_like(v); dv[1:] = (v[1:] - v[:-1]) / dt
        dv_ok = ok.clone(); dv_ok[0] = False

        print(f"\n{'='*72}\n[{name}] 유효 스텝 {int(ok.sum())} / term {int((tm==1).sum() + (tm==3).sum())}회")

        # 1) 속도-곡률 프로파일 vs 물리 한계속
        print("  --- 속도 vs 현재 곡률 (v_lim = sqrt(mu*g/|k|), mu*g=11.8) ---")
        for lo, hi, lbl in [(0.0, 0.05, "직선(|k|<0.05)"), (0.05, 0.15, "완만(0.05~0.15)"),
                            (0.15, 0.25, "중간(0.15~0.25)"), (0.25, 1.0, "급코너(>0.25)")]:
            m = ok & (k0.abs() >= lo) & (k0.abs() < hi)
            if m.sum() < 10: continue
            vm = v[m]
            klim = (MU_G / k0.abs().clamp(min=1e-3))[m].sqrt().clamp(max=12)
            print(f"    {lbl:18s} v={vm.mean():5.2f}±{vm.std():4.2f} (p95 {vm.quantile(0.95):4.1f}) "
                  f"| 한계속 평균 {klim.mean():4.1f} | 여유율 v/v_lim={((vm/klim).mean()):.2f}")

        # 2) 그립 사용률 (a_lat = v * yaw_rate)
        alat = (v * yr).abs()
        m = ok & (v > 3.0)
        print(f"  --- 그립 사용률 |a_lat|/mu*g (v>3): p50={alat[m].quantile(0.5)/MU_G:.2f} "
              f"p90={alat[m].quantile(0.9)/MU_G:.2f} p99={alat[m].quantile(0.99)/MU_G:.2f} "
              f"(1.0 = 마찰 한계) | 실측 최대 a_lat={alat[m].quantile(0.999):.1f} m/s²")

        # 3) 언더스티어: 실측 요레이트 / 기하학적 기대치 (조인트각 기준)
        print("  --- 언더스티어 (실측 yaw rate / bicycle 기대치, |joint|>0.1rad) ---")
        expect = v / L_WB * torch.tan(joint)
        for lo, hi in [(2, 4), (4, 6), (6, 8), (8, 11)]:
            m = ok & (v >= lo) & (v < hi) & (joint.abs() > 0.1) & (expect.abs() > 0.5)
            if m.sum() < 20: continue
            ratio = (yr[m] / expect[m]).clamp(-2, 2)
            print(f"    v {lo}~{hi} m/s: ratio p50={ratio.quantile(0.5):.2f} "
                  f"(1.0=완전 추종, 낮을수록 앞바퀴 미끄러짐) n={int(m.sum())}")

        # 4) 조향 명령 vs 실제 조인트: 액추에이터 추종
        m = ok
        cmd_rad = cmd * rc.max_steering_angle
        print(f"  --- 조향: |명령|평균 {cmd.abs()[m].mean():.2f} (직선구간 "
              f"{cmd.abs()[ok & (k0.abs()<0.05)].mean():.2f}) | "
              f"조인트 추종오차 {((joint-cmd_rad).abs()[m]).mean():.3f} rad")

        # 5) 라인 선택: sign(k)*lat/hw (>0 인코스, <0 아웃코스)
        line = (torch.sign(k0) * lat / hw)
        for lo, hi, lbl in [(0.15, 1.0, "코너(|k|>0.15)"), (0.0, 0.05, "직선")]:
            m = ok & (k0.abs() >= lo) & (k0.abs() < hi)
            if m.sum() < 10: continue
            print(f"  --- 라인 {lbl}: sign(k)·lat/hw 평균 {line[m].mean():+.2f} "
                  f"(+1=인코스 벽, -1=아웃코스 벽) | |lat|/hw p90={((lat.abs()/hw)[m]).quantile(0.9):.2f}")

        # 6) 제동: 접근 단계별 감속량
        print("  --- 제동 프로파일 (dv, m/s²; 음수=감속) ---")
        appr90 = dv_ok & (k90.abs() > 0.25) & (k0.abs() < 0.1) & (v > 6)   # 코너 13.6m 전
        appr60 = dv_ok & (k60.abs() > 0.25) & (k0.abs() < 0.1) & (v > 6)   # 9m 전
        entry = dv_ok & (k15.abs() > 0.25) & (v > 4)                        # 2.2m 전
        incorner = dv_ok & (k0.abs() > 0.25)
        for m, lbl in [(appr90, "13.6m 전(급코너)"), (appr60, "9m 전"), (entry, "2.2m 전"),
                       (incorner, "코너 내부")]:
            if m.sum() < 10:
                print(f"    {lbl:16s} (표본 부족)"); continue
            print(f"    {lbl:16s} dv 평균 {dv[m].mean():+5.2f}  p10 {dv[m].quantile(0.10):+5.2f}")
        print(f"    전체 최대 감속(p1) = {dv[dv_ok].quantile(0.01):+.2f} m/s² "
              f"(후륜 제동 이론 한계 ≈ -5)")

        # 7) term 상황
        tmask = (tm == 1) | (tm == 3)
        if tmask.sum() > 0:
            vt_ = v[tmask]
            side = (torch.sign(k0) * lat / hw)[tmask]
            print(f"  --- term {int(tmask.sum())}회: 속도 평균 {vt_.mean():.1f} m/s | "
                  f"급코너 비율 {(k0.abs()[tmask]>0.2).float().mean():.0%} | "
                  f"아웃코스측 이탈 비율 {(side<0).float().mean():.0%}")

    report("a", "CVaR (car_a)")
    report("b", "Pow (car_b)")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
