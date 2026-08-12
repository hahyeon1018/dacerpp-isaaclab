"""Car B 잔여 문제 계측: (1) 드리프트 코너링 (2) A 근접 시 벽 충돌.

학습과 동일 조건(A 속도캡 활성)으로 주행하며 로그 수집 후:
  1) 드리프트 이벤트(|β|>slip_deadzone 지속 3스텝+) 추출:
     위치(곡률), 속도 vs 그립한계속, 지속시간, 슬립벌점 vs 진행보상 경제성
  2) B 종료 원인 분해(벽/추돌/스핀/기타) + 'A 근접 상황' 교차분석:
     당시 폭, B 속도, 접근속도, 제동 가능성(TTC vs 제동한계)
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, default="dacerpp_runs/20260710_1")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=6000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import torch
from dacerpp_lab.env_cfg import RacingEnvCfg
from dacerpp_lab.racing_env import RacingEnv
from dacerpp_lab.vectorized_track import wrap_to_pi
from dacer_pp import DACERppConfig, DACERpp, conservative_cvar, aggressive_pow

project_dir = "/home/sscc/Desktop/hahyeon/dacerpp_isaaclab"
cfg = RacingEnvCfg()
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
MU_G_EFF = 10.3   # characterize_car 실측 횡그립 한계 (a_lat_max)

def make(risk, seed):
    c = DACERppConfig(obs_dim=rc.obs_dim(), act_dim=2, risk=risk,
                      device=str(dev), use_compile=False, seed=seed)
    return DACERpp(c)
ag_a = make(conservative_cvar(0.5), 0).load(os.path.join(args.ckpt_dir, "cvar.pt"))
ag_b = make(aggressive_pow(1.2), 1).load(os.path.join(args.ckpt_dir, "pow.pt"))

obs, _ = env.reset()
oa, ob = obs["car_a"], obs["car_b"]
T = args.steps
K = ["v_b", "v_a", "beta_b", "herr_b", "k0_b", "k15_b", "hw_b", "lat_b",
     "st_b", "th_b", "term_b", "car_dist", "gap_ab", "ds_b", "rew_b"]
L = {k: torch.zeros(T, N, device=dev) for k in K}

prev_s_b = None
for t in range(T):
    aa = ag_a.act_batch(oa, deterministic=True)
    ab = ag_b.act_batch(ob, deterministic=True)
    la, ya, va = env._car_state(env.car_a)
    lb, yb, vb = env._car_state(env.car_b)
    pa = env._vt.project(la, env._track_type)
    pb = env._vt.project(lb, env._track_type)
    # 차체 좌표 사이드슬립
    vel = env.car_b.data.root_lin_vel_w
    cosy, siny = torch.cos(yb), torch.sin(yb)
    vx = vel[:, 0] * cosy + vel[:, 1] * siny
    vy = -vel[:, 0] * siny + vel[:, 1] * cosy
    L["beta_b"][t] = torch.atan2(vy.abs(), vx.abs().clamp(min=0.5))
    L["herr_b"][t] = wrap_to_pi(yb - pb["psi"])
    k = env._vt.lookahead_curvature(pb["idx"], env._track_type, (0, 15))
    L["k0_b"][t], L["k15_b"][t] = k[:, 0], k[:, 1]
    L["hw_b"][t] = env._vt.lookahead_width(pb["idx"], env._track_type, (0,)).squeeze(1)
    L["lat_b"][t] = pb["lateral"]
    L["v_b"][t], L["v_a"][t] = vb, va
    L["st_b"][t], L["th_b"][t] = ab[:, 0], ab[:, 1]
    L["car_dist"][t] = torch.linalg.norm(la - lb, dim=1)
    gap = pa["s"] - pb["s"]
    tot = env._total_s
    gap = torch.where(gap < -0.5 * tot, gap + tot, gap)
    gap = torch.where(gap > 0.5 * tot, gap - tot, gap)
    L["gap_ab"][t] = gap                            # >0 : A 가 B 앞
    if prev_s_b is not None:
        ds = pb["s"] - prev_s_b
        ds = torch.where(ds < -0.5 * tot, ds + tot, ds)
        ds = torch.where(ds > 0.5 * tot, ds - tot, ds)
        L["ds_b"][t] = ds
    prev_s_b = pb["s"].clone()

    next_obs, _, _, _, _ = env.step(torch.cat([aa, ab], dim=1))
    di = env.dual_info()
    oa, ob = next_obs["car_a"], next_obs["car_b"]
    L["term_b"][t] = di["term_b"].float()
    L["rew_b"][t] = di["rew_b"]

dt = env.step_dt
ok = torch.ones(T, N, dtype=torch.bool, device=dev)
ok[1:] = L["term_b"][:-1] < 0.5        # 리스폰 직후 스텝 제외 (ds 텔레포트 오염)
print(f"\n{'='*74}\n[개요] {T}스텝 x {N}대 | B 평균속도 {L['v_b'][ok].mean():.2f} m/s | "
      f"B term {int(L['term_b'].sum())}회")

# ================= 1. 드리프트 이벤트 분석 =================
beta, v, k0 = L["beta_b"], L["v_b"], L["k0_b"]
drift = (beta > rc.slip_deadzone) & ok
print(f"\n---- 1. 드리프트 (|β| > {rc.slip_deadzone:.2f}rad={rc.slip_deadzone*57.3:.0f}°) ----")
print(f"  전체 스텝 중 드리프트 비율: {drift.float().mean():.3%} | "
      f"β p95={beta[ok].quantile(0.95):.3f} p99={beta[ok].quantile(0.99):.3f} rad")
# 이벤트 추출 (env 별 연속 구간, 3스텝 이상)
events = []
for n in range(N):
    d = drift[:, n]
    t0 = None
    for t in range(T):
        if d[t] and t0 is None:
            t0 = t
        elif (not d[t]) and t0 is not None:
            if t - t0 >= 3:
                events.append((n, t0, t))
            t0 = None
if events:
    import statistics as S
    e_beta, e_v, e_k, e_dur, e_vlim, e_pen, e_prog, e_term = [], [], [], [], [], [], [], 0
    for (n, t0, t1) in events:
        sl = slice(t0, t1)
        e_beta.append(float(beta[sl, n].max()))
        e_v.append(float(v[sl, n].mean()))
        kmax = float(k0[sl, n].abs().max().clamp(min=1e-3))
        e_k.append(kmax)
        e_vlim.append((MU_G_EFF / kmax) ** 0.5)
        e_dur.append((t1 - t0) * dt)
        ex = (beta[sl, n] - rc.slip_deadzone).clamp(min=0)
        pen = rc.k_slip * (ex / rc.slip_deadzone).square().clamp(max=4.0)
        e_pen.append(float(pen.sum()))
        e_prog.append(float(L["ds_b"][sl, n].sum()))
        if L["term_b"][t1:min(t1+3, T), n].sum() > 0:
            e_term += 1
    n_ev = len(events)
    print(f"  드리프트 이벤트: {n_ev}건 (지속>=0.1s) | 평균 지속 {S.mean(e_dur):.2f}s | "
          f"직후 종료로 이어진 비율 {e_term/n_ev:.0%}")
    print(f"  이벤트 중 maxβ 분포: p50={S.median(e_beta):.2f} rad "
          f"({S.median(e_beta)*57.3:.0f}°) | 평균속도 {S.mean(e_v):.2f} m/s")
    print(f"  코너 강도 |k|max 평균 {S.mean(e_k):.2f} (그립 한계속 평균 {S.mean(e_vlim):.1f} m/s) "
          f"-> 속도 초과율 v/v_lim 평균 {S.mean([a/b for a,b in zip(e_v,e_vlim)]):.2f}")
    print(f"  [경제성] 이벤트당 슬립벌점 합 평균 {S.mean(e_pen):.3f} vs "
          f"진행보상(k_progress*ds) 합 평균 {S.mean(e_prog):.3f}")
    # 곡률 구간별 드리프트 발생률
    for lo, hi, lbl in [(0.15, 0.25, "중간코너"), (0.25, 0.4, "급코너"), (0.4, 1.0, "최급코너")]:
        m = ok & (k0.abs() >= lo) & (k0.abs() < hi)
        if m.sum() > 50:
            print(f"    {lbl}(|k| {lo}~{hi}): 드리프트 비율 {drift[m].float().mean():.1%} "
                  f"| 평균 v={v[m].mean():.2f} (그립한계 {((MU_G_EFF)/((lo+hi)/2))**0.5:.1f})")

# ================= 2. B 종료 원인 분해 + A 근접 교차분석 =================
print(f"\n---- 2. B 종료 원인 분해 ----")
tm = L["term_b"] > 0.5
tidx = tm.nonzero(as_tuple=False)
n_off = n_crash = n_spin = n_other = 0
wall_near_a = []    # (v, hw, gap, closing, k0)
for (t, n) in tidx:
    t, n = int(t), int(n)
    lat, hw = float(L["lat_b"][t, n]), float(L["hw_b"][t, n])
    herr = float(L["herr_b"][t, n])
    cd, gap = float(L["car_dist"][t, n]), float(L["gap_ab"][t, n])
    is_off = abs(lat) > hw + rc.offtrack_margin - 0.03   # 여유 포함 근사
    is_crash = cd < rc.car_collision_dist + 0.02 and gap >= 0
    is_spin = abs(herr) > rc.spin_herr_limit - 0.05
    if is_crash:
        n_crash += 1
    elif is_spin and not is_off:
        n_spin += 1
    elif is_off:
        n_off += 1
        # A 근접 벽충돌: A 가 앞 0~6m + 직선거리 4m 이내
        if 0 <= gap < 6.0 and cd < 4.0:
            t0 = max(t - 15, 0)
            closing = (L["car_dist"][t0, n] - cd) / ((t - t0) * dt + 1e-6)
            wall_near_a.append((float(L["v_b"][t, n]), hw, gap, float(closing),
                                float(L["k0_b"][t, n])))
    else:
        n_other += 1
n_t = len(tidx)
if n_t:
    print(f"  총 {n_t}회: 벽/이탈 {n_off} ({n_off/n_t:.0%}) | A 추돌 {n_crash} ({n_crash/n_t:.0%}) "
          f"| 스핀 {n_spin} ({n_spin/n_t:.0%}) | 기타 {n_other}")
    print(f"  벽/이탈 중 'A 근접(앞 6m 이내)' 상황: {len(wall_near_a)}건 "
          f"({len(wall_near_a)/max(n_off,1):.0%})")
    if wall_near_a:
        import statistics as S
        vs = [w[0] for w in wall_near_a]; hws = [w[1] for w in wall_near_a]
        gaps = [w[2] for w in wall_near_a]; cls = [w[3] for w in wall_near_a]
        print(f"    당시 B 속도 평균 {S.mean(vs):.1f} m/s | 지역 반폭 평균 {S.mean(hws):.2f}m "
              f"(전체 폭 평균 {L['hw_b'][ok].mean():.2f}) | A 와 s 간격 평균 {S.mean(gaps):.1f}m")
        print(f"    직전 0.5s 접근속도 평균 {S.mean(cls):.1f} m/s "
              f"-> 제동 필요거리(4.6m/s²) 평균 {S.mean([c*c/(2*4.6) for c in cls if c>0] or [0]):.1f}m")
    # 폭 조건: 전체 벽 이탈의 폭 분포
    hw_at_off = [float(L["hw_b"][int(t), int(n)]) for (t, n) in tidx]
    print(f"  종료 시점 반폭 p50={sorted(hw_at_off)[len(hw_at_off)//2]:.2f}m "
          f"(주행 전체 p50={L['hw_b'][ok].quantile(0.5):.2f}m) -> 좁은 구간 편중 여부")

# ---- 추종 행동: A 뒤 0~3m 에서 B 가 속도를 맞추는가 ----
foll = ok & (L["gap_ab"] > 0) & (L["gap_ab"] < 3.0)
if foll.sum() > 100:
    dvrel = (L["v_b"] - L["v_a"])[foll]
    print(f"\n---- 3. A 추종(앞 0~3m) 시 상대속도: 평균 {dvrel.mean():+.2f} m/s "
          f"p90 {dvrel.quantile(0.9):+.2f} (양수=계속 접근) | 해당 상황 {int(foll.sum())}스텝 ----")

env.close()
simulation_app.close()
