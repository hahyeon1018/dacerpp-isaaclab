#!/usr/bin/env python3
"""DACER++ 듀얼 에이전트 학습 (Isaac Lab).

각 환경에 2대: car_a = CVaR(보수), car_b = Pow(공격).
2048대 car_a 는 하나의 CVaR 파라미터를 공유 학습, 2048대 car_b 는 하나의 Pow 파라미터를 공유 학습.

실행:
  conda activate env_isaaclab
  python scripts/train.py --num_envs 2048 --headless
  # 이어서 학습 (예: 20260706 폴더의 가중치에서 재개):
  python scripts/train.py --num_envs 2048 --headless --resume_dir dacerpp_runs/20260706
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

# ----------------------------------------------------------------------------
# 1) Isaac Lab 앱을 가장 먼저 띄운다 (다른 isaaclab/omni import 보다 선행).
# ----------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=10_000_000)
parser.add_argument("--warmup", type=int, default=5_000)
parser.add_argument("--utd", type=int, default=2, help="환경스텝당 그래디언트 업데이트 수")
parser.add_argument("--batch_size", type=int, default=2048)
parser.add_argument("--buffer_size", type=int, default=6_000_000)
parser.add_argument("--save_every", type=int, default=50_000)
parser.add_argument("--save_dir", type=str, default=os.path.expanduser("~/Desktop/hahyeon/dacerpp_isaaclab/dacerpp_runs"),
                    help="이 아래에 실행마다 날짜명 폴더(예: 20260706)를 만들어 저장")
parser.add_argument("--resume_dir", type=str, default=None,
                    help="이어서 학습할 체크포인트 폴더(cvar.pt/pow.pt 위치). 미지정 시 처음부터 학습")
parser.add_argument("--num_procedural", type=int, default=None,
                    help="절차 생성 트랙 수 (미지정 시 TrackFieldCfg 기본값). "
                         "총 트랙 = 이 값 + f1tenth 27종. num_envs 를 넘으면 초과분은 미배정")
parser.add_argument("--track_seed", type=int, default=None,
                    help="절차 트랙 생성 시드 고정 (미지정 시 실행마다 새 세트)")
parser.add_argument("--all_tracks", action="store_true",
                    help="f1tenth 실서킷 + 절차 생성 맵도 함께 학습(일반화). 기본은 comp "
                         "폴더(대회 코스+실측 연습맵)만 — 코스 형상이 알려진 대회 집중(2026-07-31).")
parser.add_argument("--cvar_eta", type=float, default=0.5)
parser.add_argument("--pow_eta", type=float, default=1.3)
parser.add_argument("--k_overtake", type=float, default=None,
                    help="(구) 연속 접근 보상 계수 — 기본 0(비활성). 추월은 r_overtake 이벤트 보상 사용")
parser.add_argument("--k_accel", type=float, default=None,
                    help="종가속도 벌점 계수 (미지정 시 env_cfg 기본값 0.05)")
parser.add_argument("--no_compile", action="store_true",
                    help="torch.compile/CUDA Graphs 비활성화 (디버깅용)")
parser.add_argument("--wall_visuals", action="store_true",
                    help="덕트 벽 렌더 전용 시각화 (GUI + 작은 num_envs 로 확인용)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Isaac Sim 기동 전에 체크포인트 존재를 확인 (기동에 수십 초 걸리므로 fail-fast)
if args.resume_dir and not (os.path.isfile(os.path.join(args.resume_dir, "cvar.pt"))
                            and os.path.isfile(os.path.join(args.resume_dir, "pow.pt"))):
    parser.error(f"--resume_dir: {args.resume_dir} 에 cvar.pt/pow.pt 가 없습니다.")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ----------------------------------------------------------------------------
# 2) 앱 기동 후에 무거운 모듈 import
# ----------------------------------------------------------------------------
import torch

from dacerpp_lab.env_cfg import RacingEnvCfg
from dacerpp_lab.racing_env import RacingEnv
from dacer_pp import (DACERppConfig, DACERpp, TensorReplayBuffer,
                      conservative_cvar, aggressive_pow)


@torch.no_grad()
def collapse_metrics(agent, obs, n_states=32, n_samples=64):
    """정책/크리틱 붕괴 조기 감지 지표 (20260708 런 사후분석에서 도출).

    pstd    : 같은 상태에서 뽑은 디퓨전 행동 n_samples개의 표준편차(상태 평균, 차원별).
              조향(pstd[0])이 0 에 붙으면 정책이 상태별 단일점(뱅뱅 포화)으로 붕괴한 것.
              (붕괴 런 실측 0.02 vs 건강한 확률 정책 0.1~0.5)
    qspread : 9x5 행동 그리드에 대한 min-double-Q 변화폭의 상태 평균.
              0 근처에 계속 머물면 크리틱이 행동 조건부 가치를 잃은 것 —
              '조향 없이 직진' 붕괴의 선행 신호. (붕괴 런 실측 0.08,
              같은 시점 상태간 Q 변화폭 1.5 의 5%에 불과했음)
    """
    o = obs[:n_states]
    n, od = o.shape
    rep = o.unsqueeze(1).expand(n, n_samples, od).reshape(-1, od)
    acts = agent.select_action(rep, with_noise=False).reshape(n, n_samples, -1)
    pstd = acts.std(dim=1).mean(dim=0)                                 # (act_dim,)
    steers = torch.linspace(-1, 1, 9, device=o.device)
    thrs = torch.linspace(-1, 1, 5, device=o.device)
    grid = torch.cartesian_prod(steers, thrs)                          # (45,2)
    og = o.unsqueeze(1).expand(n, grid.shape[0], od).reshape(-1, od)
    ga = grid.unsqueeze(0).expand(n, -1, -1).reshape(-1, 2)
    # 컴파일된 QVN 은 학습 배치 크기에 고정 캡처되므로 eager 모듈로 평가
    q = 0
    for _ in range(4):                                                 # tau 표본 평균으로 분산 축소
        q = q + torch.min(agent.q_beta(agent.qvn1, og, ga),
                          agent.q_beta(agent.qvn2, og, ga))
    q = (q / 4).reshape(n, -1)
    return pstd, (q.max(dim=1).values - q.min(dim=1).values).mean()


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # TF32 행렬곱 허용 (Ampere+ 에서 무료 성능 향상, RL 학습 정밀도 영향 없음)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # ---- 환경 ----
    cfg = RacingEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.project_dir = project_dir
    cfg.track_field.num_envs = args.num_envs
    if args.all_tracks:                      # f1tenth + 절차 포함(일반화)
        cfg.track_field.use_f1tenth = True
        if args.num_procedural is not None:
            cfg.track_field.num_procedural = args.num_procedural
    else:                                    # 기본: comp 폴더만 (f1tenth·절차 제거)
        cfg.track_field.num_procedural = 0
        cfg.track_field.use_f1tenth = False
    if args.track_seed is not None:
        cfg.track_field.procedural_seed = args.track_seed
    cfg.wall_visuals = args.wall_visuals
    if args.k_overtake is not None:
        cfg.racing.k_overtake = args.k_overtake
    if args.k_accel is not None:
        cfg.racing.k_accel = args.k_accel
    # gym 공간 정합 (dict 관측의 "policy" 키 = 두 차량 관측 concat)
    cfg.observation_space = 2 * cfg.racing.obs_dim()
    # 트랙이 서로 안 겹치도록 그리드 간격을 트랙 크기에 맞춤
    from dacerpp_lab.track_field import TrackField
    spacing = TrackField(cfg.track_field).suggested_env_spacing
    cfg.scene.env_spacing = spacing

    env = RacingEnv(cfg)
    device = str(env.device)
    rc = cfg.racing
    obs_dim, act_dim = rc.obs_dim(), rc.act_dim()
    N = env.num_envs
    print(f"[INFO] envs={N} obs_dim={obs_dim} act_dim={act_dim} spacing={spacing:.1f}m "
          f"device={device} compile={not args.no_compile}")

    # ---- 두 에이전트 (파라미터 분리, 각자 2048대 공유) ----
    def make_agent(risk, seed):
        c = DACERppConfig(obs_dim=obs_dim, act_dim=act_dim, risk=risk,
                          batch_size=args.batch_size, device=device,
                          use_compile=not args.no_compile, seed=seed)
        return DACERpp(c)

    agent_cvar = make_agent(conservative_cvar(args.cvar_eta), seed=0)   # car_a
    agent_pow = make_agent(aggressive_pow(args.pow_eta), seed=1)        # car_b

    buf_cvar = TensorReplayBuffer(obs_dim, act_dim, args.buffer_size, device=device, seed=0)
    buf_pow = TensorReplayBuffer(obs_dim, act_dim, args.buffer_size, device=device, seed=1)

    # ---- 저장 폴더: 실행마다 당일 날짜명 하위 폴더 생성 (예: dacerpp_runs/20260706) ----
    date_str = datetime.now().strftime("%Y%m%d")
    run_dir = os.path.join(args.save_dir, date_str)
    n = 1
    while os.path.exists(run_dir):  # 같은 날 재실행 시 20260706_1, _2 ... 로 분리
        run_dir = os.path.join(args.save_dir, f"{date_str}_{n}")
        n += 1
    os.makedirs(run_dir)
    cvar_path = os.path.join(run_dir, "cvar.pt")
    pow_path = os.path.join(run_dir, "pow.pt")
    meta_path = os.path.join(run_dir, "train_meta.json")
    # ---- 붕괴 지표 파일 로그 (2026-07-30 실차 분석 [9].3) ----
    # 구 런(20260726)은 collapse_metrics 를 stdout 에만 찍어, 사후에 '언제 붕괴했나'를
    # 알 수 없어 붕괴 직전 체크포인트로 되돌릴 수 없었다. 매 로깅 시점(200 it)마다
    # CSV 로 append 한다 (resume 시에도 이어 붙임). alpha/entropy 도 함께 남겨
    # 새 엔트로피 처방(정책 자체 다양성 반응)이 작동하는지 추적한다.
    collapse_csv = os.path.join(run_dir, "collapse_log.csv")
    if not os.path.exists(collapse_csv):
        with open(collapse_csv, "w") as f:
            f.write("it,alpha_cvar,alpha_pow,ent_cvar,ent_pow,"
                    "pstd_a_steer,pstd_a_thr,qspread_a,"
                    "pstd_b_steer,pstd_b_thr,qspread_b,"
                    "v_cap,k_slip,mu_min,mu_max,r_a,r_b,term\n")
    print(f"[INFO] checkpoint dir: {run_dir}")

    # ---- 재개(resume): --resume_dir 폴더의 체크포인트 로드 ----
    # 가중치(policy/Q-net/alpha/step)는 복원되지만 리플레이 버퍼는 저장되지 않으므로
    # 재개 직후 warmup 만큼 버퍼를 다시 채운 뒤 업데이트가 시작된다.
    start_it = 0
    if args.resume_dir:
        agent_cvar.load(os.path.join(args.resume_dir, "cvar.pt"))
        agent_pow.load(os.path.join(args.resume_dir, "pow.pt"))
        load_meta = os.path.join(args.resume_dir, "train_meta.json")
        if os.path.exists(load_meta):
            with open(load_meta) as f:
                start_it = int(json.load(f)["it"])
            print(f"[RESUME] {args.resume_dir} 로드 완료 -> it={start_it} 부터 재개")
        else:
            # meta 가 없는 구(舊) 체크포인트: step 으로 it 추정 (step ≈ it*utd)
            start_it = int(agent_cvar.step) // max(1, args.utd)
            print(f"[RESUME] {args.resume_dir} 로드 완료(meta 없음) -> step 기준 추정 it≈{start_it} 부터 재개")
        print("[RESUME] 리플레이 버퍼는 비어 있으므로 warmup 만큼 다시 채웁니다.")

    # ---- 초기 관측 ----
    obs, _ = env.reset()
    oa, ob = obs["car_a"], obs["car_b"]

    for it in range(start_it, args.iterations):
        env.set_curriculum_progress(it)   # 속도 커리큘럼 (v0 -> v_max 램프)
        aa = agent_cvar.act_batch(oa, deterministic=False)
        ab = agent_pow.act_batch(ob, deterministic=False)
        action = torch.cat([aa, ab], dim=1)

        next_obs, _, terminated, truncated, _ = env.step(action)
        di = env.dual_info()

        # 차량별 done(부트스트랩): terminated=true terminal(이탈/충돌)만 done 으로.
        # next_obs 는 리셋 "이전" 상태의 최종 관측(final_obs)을 사용한다 --
        # 리셋된 환경의 리셋 후 관측으로 부트스트랩하면 가치 추정이 오염된다.
        done_a = di["term_a"].float()
        done_b = di["term_b"].float()
        buf_cvar.add_batch(oa, aa, di["rew_a"], di["final_obs_a"], done_a)
        buf_pow.add_batch(ob, ab, di["rew_b"], di["final_obs_b"], done_b)
        oa, ob = next_obs["car_a"], next_obs["car_b"]

        # ---- 업데이트 ----
        if len(buf_cvar) > args.warmup:
            for _ in range(args.utd):
                m_c = agent_cvar.update(buf_cvar.sample(args.batch_size))
        if len(buf_pow) > args.warmup:
            for _ in range(args.utd):
                m_p = agent_pow.update(buf_pow.sample(args.batch_size))

        if it % 200 == 0 and len(buf_cvar) > args.warmup:
            # 지표는 0-d 텐서(지연 동기화) -> 로깅 시점에만 float() 변환
            print(f"[{it}] CVaR q1={float(m_c.get('q1_loss', 0)):.3f} a={float(m_c['alpha']):.2f} "
                  f"r_a={di['rew_a'].mean().item():.3f} | "
                  f"Pow q1={float(m_p.get('q1_loss', 0)):.3f} a={float(m_p['alpha']):.2f} "
                  f"r_b={di['rew_b'].mean().item():.3f} | "
                  f"term={(di['term_a'] | di['term_b']).float().mean().item():.3f}")
            # 붕괴 감지: pstd[0](조향 다양성)->0 이면서 qspread 정체면
            # '평평한 Q + 포화 정책' 붕괴 재발 신호 (collapse_metrics docstring 참조)
            sa_m, qa_m = collapse_metrics(agent_cvar, oa)
            sb_m, qb_m = collapse_metrics(agent_pow, ob)
            print(f"      [collapse] CVaR pstd=({float(sa_m[0]):.3f},{float(sa_m[1]):.3f}) "
                  f"qspread={float(qa_m):.3f} | "
                  f"Pow pstd=({float(sb_m[0]):.3f},{float(sb_m[1]):.3f}) "
                  f"qspread={float(qb_m):.3f} | v_cap={env._v_cmd_max:.1f}m/s "
                  f"k_slip={env._k_slip_cur:.2f} "
                  f"mu=[{env._mu.min().item():.2f},{env._mu.max().item():.2f}]")
            with open(collapse_csv, "a") as f:
                f.write(
                    f"{it},{float(m_c['alpha']):.4f},{float(m_p['alpha']):.4f},"
                    f"{float(m_c.get('entropy', 0)):.4f},{float(m_p.get('entropy', 0)):.4f},"
                    f"{float(sa_m[0]):.4f},{float(sa_m[1]):.4f},{float(qa_m):.4f},"
                    f"{float(sb_m[0]):.4f},{float(sb_m[1]):.4f},{float(qb_m):.4f},"
                    f"{env._v_cmd_max:.2f},{env._k_slip_cur:.3f},"
                    f"{env._mu.min().item():.3f},{env._mu.max().item():.3f},"
                    f"{di['rew_a'].mean().item():.4f},{di['rew_b'].mean().item():.4f},"
                    f"{(di['term_a'] | di['term_b']).float().mean().item():.4f}\n")

        if it > 0 and it % args.save_every == 0:
            agent_cvar.save(cvar_path)
            agent_pow.save(pow_path)
            with open(meta_path, "w") as f:
                json.dump({"it": it}, f)
            print(f"[{it}] checkpoint saved -> {run_dir}")

    agent_cvar.save(cvar_path)
    agent_pow.save(pow_path)
    with open(meta_path, "w") as f:
        json.dump({"it": args.iterations}, f)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
