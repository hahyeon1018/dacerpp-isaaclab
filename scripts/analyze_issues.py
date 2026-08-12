#!/usr/bin/env python3
"""Car B 잔여 문제 계측 분석 (headless).

문제1 — 코너 드리프트/슬립: 슬립 이벤트의 위치·속도·지속시간, 드리프트 중
        그립 사용률(타이어 모델의 피크 이후 손실 여부), 그리고 보상 손익
        (진행 보상 vs 슬립 벌점)을 정량화해 "드리프트가 합리적 선택인지" 판정.
문제2 — 추월 상황 벽 충돌: B 벽/스핀 종료 중 'A 근접' 상황의 비율, 직전 1초의
        제동/조향 반응, 추종 vs 충돌-리스폰의 보상률 비교(충돌 도피 인센티브).

실행:
  python -u scripts/analyze_issues.py --ckpt_dir dacerpp_runs/20260710_1 --steps 4000 --headless
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=4000)
parser.add_argument("--cvar_eta", type=float, default=0.5)
parser.add_argument("--pow_eta", type=float, default=1.2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

_cvar = os.path.join(args.ckpt_dir, "cvar.pt")
_pow = os.path.join(args.ckpt_dir, "pow.pt")
assert os.path.isfile(_cvar) and os.path.isfile(_pow), args.ckpt_dir

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch

from dacerpp_lab.env_cfg import RacingEnvCfg
from dacerpp_lab.racing_env import RacingEnv
from dacerpp_lab.track_field import TrackField
from dacer_pp import DACERppConfig, DACERpp, conservative_cvar, aggressive_pow


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = RacingEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.project_dir = project_dir
    cfg.track_field.num_envs = args.num_envs
    cfg.wall_visuals = False
    cfg.racing.obs_noise = False
    cfg.episode_length_s = 1.0e9          # 개별 리스폰만 (test.py 와 동일)
    cfg.observation_space = 2 * cfg.racing.obs_dim()
    cfg.scene.env_spacing = TrackField(cfg.track_field).suggested_env_spacing

    env = RacingEnv(cfg)
    rc = cfg.racing
    tc = cfg.tire
    dev = env.device
    N = env.num_envs
    T = args.steps
    dt = env.step_dt
    MU_G = tc.mu * 9.81

    def make_agent(risk, seed):
        c = DACERppConfig(obs_dim=rc.obs_dim(), act_dim=rc.act_dim(), risk=risk,
                          device=str(dev), use_compile=False, seed=seed)
        return DACERpp(c)

    ag_a = make_agent(conservative_cvar(args.cvar_eta), 0).load(_cvar)
    ag_b = make_agent(aggressive_pow(args.pow_eta), 1).load(_pow)
    print(f"[ckpt] {args.ckpt_dir}: cvar step={ag_a.step} pow step={ag_b.step} | "
          f"envs={N} steps={T} | A 속도캡={rc.v_cap_a_range}")

    obs, _ = env.reset()
    oa, ob = obs["car_a"], obs["car_b"]

    keys = ["v_a", "v_b", "beta_b", "beta_sgn_b", "yr_b", "st_b", "th_b", "joint_b",
            "k0_b", "lat_b", "hw_b", "s_b", "gap_ab", "dist", "herr_b",
            "term_b", "off_b", "crash_b", "spun_b", "flip_b", "term_a",
            "rew_b", "fresh_b"]
    L = {k: torch.zeros(T, N, device=dev) for k in keys}
    fresh_b = torch.ones(N, dtype=torch.bool, device=dev)

    for t in range(T):
        aa = ag_a.act_batch(oa, deterministic=True)
        ab = ag_b.act_batch(ob, deterministic=True)

        # ---- 결정 시점(물리 전) 상태 기록 ----
        la, ya, va = env._car_state(env.car_a)
        lb, yb, vb = env._car_state(env.car_b)
        pa = env._vt.project(la, env._track_type)
        pb = env._vt.project(lb, env._track_type)
        vel = env.car_b.data.root_lin_vel_w
        cosy, siny = torch.cos(yb), torch.sin(yb)
        vx = vel[:, 0] * cosy + vel[:, 1] * siny
        vy = -vel[:, 0] * siny + vel[:, 1] * cosy
        L["v_a"][t], L["v_b"][t] = va, vb
        L["beta_b"][t] = torch.atan2(vy.abs(), vx.abs().clamp(min=0.5))  # env 벌점과 동일 정의
        L["beta_sgn_b"][t] = torch.sign(vy)
        L["yr_b"][t] = env.car_b.data.root_ang_vel_w[:, 2]
        L["st_b"][t], L["th_b"][t] = ab[:, 0], ab[:, 1]
        L["joint_b"][t] = env.car_b.data.joint_pos[:, env._steer_ids_b].mean(dim=1)
        L["k0_b"][t] = env._vt.lookahead_curvature(pb["idx"], env._track_type, (0,)).squeeze(1)
        L["hw_b"][t] = env._vt.lookahead_width(pb["idx"], env._track_type, (0,)).squeeze(1)
        L["lat_b"][t] = pb["lateral"]
        L["s_b"][t] = pb["s"]
        L["herr_b"][t] = ((yb - pb["psi"]) + np.pi) % (2 * np.pi) - np.pi
        L["gap_ab"][t] = env._wrap_ds(pa["s"] - pb["s"])   # >0: A 가 앞
        L["dist"][t] = torch.linalg.norm(la - lb, dim=1)
        L["fresh_b"][t] = fresh_b.float()

        next_obs, _, _, _, _ = env.step(torch.cat([aa, ab], dim=1))
        di = env.dual_info()
        oa, ob = next_obs["car_a"], next_obs["car_b"]
        L["term_b"][t] = di["term_b"].float()
        L["term_a"][t] = di["term_a"].float()
        L["rew_b"][t] = di["rew_b"]
        for c in ("off_b", "crash_b", "spun_b", "flip_b"):
            L[c][t] = di["causes"][c].float()
        fresh_b = di["term_b"].clone()

    # ================= 분석 (CPU numpy) =================
    D = {k: v.cpu().numpy() for k, v in L.items()}
    tt = env._track_type.cpu().numpy()
    tot_s = env._total_s.cpu().numpy()

    v, beta, k0, hw = D["v_b"], D["beta_b"], D["k0_b"], D["hw_b"]
    term = D["term_b"] > 0.5
    fresh = D["fresh_b"] > 0.5
    ok = ~fresh
    ok[1:] &= ~(term[:-1])                     # 리스폰 직후 스텝 제외
    dv = np.zeros_like(v); dv[1:] = (v[1:] - v[:-1]) / dt
    dv_ok = ok.copy(); dv_ok[0] = False; dv_ok[1:] &= ok[:-1]
    ds = np.zeros_like(v)
    raw = D["s_b"][1:] - D["s_b"][:-1]
    raw = np.where(raw < -0.5 * tot_s, raw + tot_s, raw)
    raw = np.where(raw > 0.5 * tot_s, raw - tot_s, raw)
    ds[1:] = raw
    alat = np.abs(v * D["yr_b"])

    n_term = int(term.sum())
    print(f"\n{'='*78}\n[Car B] 유효 스텝 {int(ok.sum())} | term {n_term}회 "
          f"(off {int(D['off_b'].sum())}, crash {int(D['crash_b'].sum())}, "
          f"spun {int(D['spun_b'].sum())}, flip {int(D['flip_b'].sum())}) | "
          f"v 평균 {v[ok].mean():.2f} m/s")

    # ---------------- 문제 1: 드리프트/슬립 ----------------
    print(f"\n{'-'*78}\n[문제1] 코너 드리프트/슬립 (slip_deadzone={rc.slip_deadzone}rad="
          f"{np.degrees(rc.slip_deadzone):.0f}°, k_slip={rc.k_slip})")
    corner = ok & (np.abs(k0) > 0.15)
    drift = ok & (beta > rc.slip_deadzone) & (v > 3.0)
    print(f"  β 분위 (코너 스텝): p50={np.quantile(beta[corner],0.5):.3f} "
          f"p90={np.quantile(beta[corner],0.9):.3f} p99={np.quantile(beta[corner],0.99):.3f} rad")
    print(f"  드리프트 스텝 비율: 전체 {drift.sum()/max(ok.sum(),1)*100:.1f}% | "
          f"코너 내 {((drift & corner).sum()/max(corner.sum(),1))*100:.1f}%")

    # 그립 사용률: 드리프트 중에도 그립이 유지되는가 (tanh 모델 = 피크 이후 손실 0)
    m_dr = drift & (v > 4); m_cl = corner & ~drift & (v > 4)
    if m_dr.sum() > 20 and m_cl.sum() > 20:
        print(f"  |a_lat|/μg: 드리프트 중 p50={np.quantile(alat[m_dr],0.5)/MU_G:.2f} "
              f"p90={np.quantile(alat[m_dr],0.9)/MU_G:.2f} | "
              f"클린 코너링 p50={np.quantile(alat[m_cl],0.5)/MU_G:.2f} "
              f"p90={np.quantile(alat[m_cl],0.9)/MU_G:.2f}")
        print(f"  속도: 드리프트 중 평균 {v[m_dr].mean():.2f} m/s | 클린 코너 {v[m_cl].mean():.2f} m/s")

    # 보상 손익: 드리프트 스텝의 진행 보상 vs 슬립 벌점 (env 수식 재현)
    slip_pen = rc.k_slip * np.clip(((beta - rc.slip_deadzone).clip(0) / rc.slip_deadzone) ** 2,
                                   0, rc.slip_clamp)
    m = drift.copy(); m[0] = False; m[1:] &= ok[:-1]     # ds 유효 스텝만
    if m.sum() > 20:
        print(f"  드리프트 스텝 보상 수지: 진행 +{ds[m].mean():.3f}/스텝 vs "
              f"슬립벌점 -{slip_pen[m].mean():.3f}/스텝 "
              f"(벌점/진행 = {slip_pen[m].mean()/max(ds[m].mean(),1e-9)*100:.0f}%)")
        satur = (slip_pen[m] >= 0.399).mean()
        print(f"  슬립벌점 상한(0.4) 포화 비율: {satur*100:.0f}% | "
              f"드리프트 중 β 평균 {beta[m].mean():.2f} rad ({np.degrees(beta[m].mean()):.0f}°)")

    # 드리프트 이벤트(연속 구간) 추출: 어디서, 얼마나, 결말은
    ev_dur, ev_v0, ev_beta, ev_thr, ev_term, ev_loc = [], [], [], [], [], []
    for i in range(N):
        t0 = None
        for t in range(T):
            if drift[t, i] and t0 is None:
                t0 = t
            elif (not drift[t, i]) and t0 is not None:
                if t - t0 >= 3:                        # 0.1초 이상 지속만 이벤트
                    ev_dur.append((t - t0) * dt)
                    ev_v0.append(v[t0, i])
                    ev_beta.append(beta[t0:t, i].max())
                    ev_thr.append(D["th_b"][t0:t, i].mean())
                    ev_term.append(bool(term[t0:min(t + 5, T), i].any()))
                    ev_loc.append((tt[i], D["s_b"][t0, i]))
                t0 = None
    if ev_dur:
        ev_dur, ev_v0, ev_beta, ev_thr = map(np.array, (ev_dur, ev_v0, ev_beta, ev_thr))
        print(f"  드리프트 이벤트 {len(ev_dur)}개: 지속 p50={np.quantile(ev_dur,0.5):.2f}s "
              f"p90={np.quantile(ev_dur,0.9):.2f}s | 진입속도 평균 {ev_v0.mean():.1f} m/s | "
              f"β최대 평균 {np.degrees(np.mean(ev_beta)):.0f}° | "
              f"이벤트 중 스로틀 평균 {np.mean(ev_thr):+.2f} | "
              f"5스텝 내 term 으로 끝난 비율 {np.mean(ev_term)*100:.0f}%")
        # 특정 코너 집중도: (트랙타입, s 5m 구간) 상위
        cell = {}
        for (g, s0) in ev_loc:
            key = (int(g), int(s0 // 5))
            cell[key] = cell.get(key, 0) + 1
        top = sorted(cell.items(), key=lambda x: -x[1])[:5]
        print("  드리프트 다발 구간 (트랙타입, s구간): " +
              ", ".join(f"T{g}@{s5*5}-{s5*5+5}m x{c}" for (g, s5), c in top))

    # ---------------- 문제 2: 추월 상황 벽 충돌 ----------------
    print(f"\n{'-'*78}\n[문제2] A 근접 상황의 벽/스핀 충돌")
    W = 30                                              # 직전 1초 윈도
    wall_term = (D["off_b"] > 0.5) | (D["spun_b"] > 0.5)
    events = []                                         # (i, t, A근접 여부)
    for i in range(N):
        for t in np.nonzero(wall_term[:, i])[0]:
            lo = max(0, t - W)
            near = ((D["gap_ab"][lo:t + 1, i] > 0) & (D["gap_ab"][lo:t + 1, i] < 8)
                    & (D["dist"][lo:t + 1, i] < 5.0)).any()
            events.append((i, int(t), bool(near)))
    n_wall = len(events)
    n_near = sum(1 for *_, nr in events if nr)
    print(f"  B 벽/스핀 term {n_wall}회 중 A 근접(≤5m, 전방 8m 내) 동반 {n_near}회 "
          f"({n_near/max(n_wall,1)*100:.0f}%)")

    def window_stats(evs, label):
        if not evs:
            print(f"  {label}: (이벤트 없음)"); return
        th_w, st_w, dv_w, v_t, hw_t, dist_m, thr_glob = [], [], [], [], [], [], D["th_b"][ok].mean()
        for i, t, _ in evs:
            lo = max(1, t - W)
            th_w.append(D["th_b"][lo:t + 1, i].mean())
            st_w.append(np.abs(D["st_b"][lo:t + 1, i]).mean())
            dv_w.append(dv[lo:t + 1, i].min())
            v_t.append(v[t, i]); hw_t.append(hw[t, i])
            dist_m.append(D["dist"][lo:t + 1, i].min())
        print(f"  {label} (n={len(evs)}): term 시 v={np.mean(v_t):.1f} m/s, "
              f"폭 2hw={2*np.mean(hw_t):.1f}m | 직전1초: 스로틀 {np.mean(th_w):+.2f} "
              f"(전체평균 {thr_glob:+.2f}), |조향| {np.mean(st_w):.2f}, "
              f"최대감속 {np.mean(dv_w):+.1f} m/s², 최소차간 {np.mean(dist_m):.2f}m")

    window_stats([e for e in events if e[2]], "A 근접 벽충돌")
    window_stats([e for e in events if not e[2]], "단독 벽충돌  ")

    # 접근 encounter 의 결말 분류: (car_dist<3 & A 전방) 시작 후 3초
    enc_out = {"추월": 0, "벽/스핀": 0, "추돌": 0, "추종 지속": 0}
    enc_active = np.zeros(N, dtype=bool)
    for i in range(N):
        t = 1
        while t < T:
            close = (D["gap_ab"][t, i] > 0) and (D["dist"][t, i] < 3.0) and ok[t, i]
            if close and not enc_active[i]:
                enc_active[i] = True
                hi = min(t + 90, T)
                seg_gap = D["gap_ab"][t:hi, i]
                seg_wall = wall_term[t:hi, i]
                seg_crash = D["crash_b"][t:hi, i] > 0.5
                if seg_crash.any() and (not seg_wall.any()
                                        or np.argmax(seg_crash) < np.argmax(seg_wall)):
                    enc_out["추돌"] += 1
                elif seg_wall.any() and ((seg_gap < 0).any()
                                         and np.argmax(seg_wall) < np.argmax(seg_gap < 0)
                                         or not (seg_gap < 0).any()):
                    enc_out["벽/스핀"] += 1
                elif (seg_gap < 0).any():
                    enc_out["추월"] += 1
                else:
                    enc_out["추종 지속"] += 1
            elif not close:
                enc_active[i] = False
            t += 1
    tot_enc = max(sum(enc_out.values()), 1)
    print("  A 접근(차간<3m) encounter 결말 (3초 내): " +
          ", ".join(f"{k} {v}회({v/tot_enc*100:.0f}%)" for k, v in enc_out.items()))

    # 충돌-도피 인센티브: 추종 중 보상률 vs 자유 주행 보상률, 종말 비용 -> 손익분기
    follow = ok & (D["gap_ab"] > 0) & (D["gap_ab"] < 4) & (D["dist"] < 2.5) & ~term
    free = ok & ((D["gap_ab"] < 0) | (D["dist"] > 8)) & ~term
    if follow.sum() > 50:
        r_fol, r_fre = D["rew_b"][follow].mean(), D["rew_b"][free].mean()
        r_term_mean = D["rew_b"][term].mean() if term.any() else float("nan")
        v_fol = v[follow].mean()
        print(f"  보상률: A 추종 중 {r_fol:+.3f}/스텝 (v={v_fol:.1f}) vs 자유 주행 {r_fre:+.3f}/스텝")
        print(f"  종말 벌점 평균 {r_term_mean:+.1f} -> 손익분기 "
              f"{abs(r_term_mean)/max(r_fre-r_fol,1e-9):.0f}스텝 "
              f"(={abs(r_term_mean)/max(r_fre-r_fol,1e-9)*dt:.1f}초 이상 추종하면 충돌-리스폰이 이득)")

    env.close()
    simulation_app.close()
    print("ANALYZE_ISSUES DONE")


if __name__ == "__main__":
    main()
