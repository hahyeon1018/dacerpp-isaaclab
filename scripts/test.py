#!/usr/bin/env python3
"""DACER++ 듀얼 에이전트 추론/시각화 (Isaac Sim GUI).

학습된 cvar.pt(car_a) / pow.pt(car_b) 를 불러와 결정적(deterministic) 행동으로
주행을 재생한다. 학습/리플레이 버퍼 없이 순수 추론만 수행하므로,
GUI + 소규모 num_envs 로 띄워 눈으로 확인하는 용도.

실행:
  conda activate env_isaacsim
  python scripts/test.py --ckpt_dir dacerpp_runs/20260706
  # 예: 환경 9개, 2000 스텝만 재생
  python scripts/test.py --ckpt_dir dacerpp_runs/20260706 --num_envs 9 --steps 2000
"""
from __future__ import annotations

import argparse
import os

# ----------------------------------------------------------------------------
# 1) Isaac Lab 앱을 가장 먼저 띄운다 (다른 isaaclab/omni import 보다 선행).
#    train.py 와 달리 기본이 GUI (--headless 를 주지 않으면 창이 뜬다).
# ----------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_dir", type=str, default=None,
                    help="cvar.pt/pow.pt 가 있는 체크포인트 폴더 (예: dacerpp_runs/20260706). "
                         "--list_tracks 외에는 필수")
parser.add_argument("--num_envs", type=int, default=4,
                    help="시각화용 환경 수 (GUI 렌더 부하를 고려해 소규모 권장)")
parser.add_argument("--steps", type=int, default=0,
                    help="재생할 스텝 수 (0 = 창을 닫을 때까지 무한)")
parser.add_argument("--stochastic", action="store_true",
                    help="Car B 에만 탐험 노이즈 포함 행동 (기본: deterministic). "
                         "Car A 는 테스트에서 항상 노이즈 없이 주행한다.")
parser.add_argument("--cvar_eta", type=float, default=0.5,
                    help="학습 때 사용한 값과 동일해야 함 (행동 후보 선택에 반영)")
parser.add_argument("--pow_eta", type=float, default=1.3,
                    help="학습 때 사용한 값과 동일해야 함 (행동 후보 선택에 반영)")
parser.add_argument("--no_wall_visuals", action="store_true",
                    help="덕트 벽 렌더 비활성화 (기본: 켜짐)")
parser.add_argument("--tracks", type=str, default="",
                    help="고정 배정할 트랙 이름(콤마 구분, 대소문자 무시). "
                         "예: --tracks hall,teras -> env0=hall, env1=teras, "
                         "남는 env 는 '지정한 것을 제외한' 나머지 트랙에서 랜덤. "
                         "미지정 시 전부 랜덤. (--list_tracks 로 이름 확인)")
parser.add_argument("--list_tracks", action="store_true",
                    help="사용 가능한 트랙 이름만 출력하고 종료 (Isaac Sim 기동 없음)")
parser.add_argument("--all_tracks", action="store_true",
                    help="f1tenth 실서킷 + 절차 생성 맵도 함께 사용. 기본은 comp 폴더"
                         "(대회 코스+실측 연습맵)만 — 학습(기본 comp-only)과 트랙 분포 일치.")
parser.add_argument("--log_every", type=int, default=100)
parser.add_argument("--obs_noise", action="store_true",
                    help="평가 중에도 관측 도메인 랜덤화(스캔/측위 노이즈)를 켠다. "
                         "★실차 배포용 체크포인트를 고를 때는 반드시 이 옵션을 쓸 것 — "
                         "노이즈를 끈 평가는 '깨끗한 관측에서만 잘하는' 취약한 정책을 "
                         "가장 좋게 평가한다(실차 실패의 전형적 경로). 기본은 구 동작(끔).")
parser.add_argument("--mu_range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="평가 지면 마찰 범위 덮어쓰기. 학습과 다른 그립에서의 강건성을 "
                         "보려면 지정 (예: 학습 0.85~1.25, 평가 --mu_range 0.5 0.7)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# --list_tracks: Isaac Sim 없이 트랙 목록만 출력하고 종료 (순수 numpy 경로)
if args.list_tracks:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dacerpp_lab.track_field import TrackField, TrackFieldCfg
    _tfc = TrackFieldCfg(num_envs=1)
    if not args.all_tracks:      # 기본: comp 폴더만
        _tfc.num_procedural = 0
        _tfc.use_f1tenth = False
    for i, nm in enumerate(TrackField(_tfc).track_names):
        print(f"  [{i:2d}] {nm}")
    raise SystemExit(0)

# Isaac Sim 기동 전에 체크포인트 존재를 확인 (기동에 수십 초 걸리므로 fail-fast)
if not args.ckpt_dir:
    parser.error("--ckpt_dir 는 필수입니다 (--list_tracks 제외).")
_cvar_ckpt = os.path.join(args.ckpt_dir, "cvar.pt")
_pow_ckpt = os.path.join(args.ckpt_dir, "pow.pt")
if not (os.path.isfile(_cvar_ckpt) and os.path.isfile(_pow_ckpt)):
    parser.error(f"--ckpt_dir: {args.ckpt_dir} 에 cvar.pt/pow.pt 가 없습니다.")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ----------------------------------------------------------------------------
# 2) 앱 기동 후에 무거운 모듈 import
# ----------------------------------------------------------------------------
import torch

from dacerpp_lab.env_cfg import RacingEnvCfg
from dacerpp_lab.racing_env import RacingEnv
from dacer_pp import DACERppConfig, DACERpp, conservative_cvar, aggressive_pow


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ---- 환경 (train.py 와 동일한 구성) ----
    cfg = RacingEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.project_dir = project_dir
    cfg.track_field.num_envs = args.num_envs
    cfg.wall_visuals = not args.no_wall_visuals   # 시각화가 목적이므로 기본 켜짐
    # 평가 관측 노이즈: 기본은 끔(시각화/디버깅용). 배포 체크포인트 선정에는 --obs_noise 로 켤 것.
    cfg.racing.obs_noise = bool(args.obs_noise)
    if args.mu_range is not None:                 # 학습과 다른 그립에서의 강건성 평가
        cfg.tire.mu_range = (float(args.mu_range[0]), float(args.mu_range[1]))
    if not args.all_tracks:                       # 기본: comp 폴더만 (학습과 트랙 분포 일치)
        cfg.track_field.num_procedural = 0
        cfg.track_field.use_f1tenth = False
    # 평가: 트랙 중 env 마다 무작위로 뽑는다 (실행마다 다름; 기본 comp 만, --all_tracks 로 전체).
    # --tracks 로 이름을 주면 그 트랙들을 앞쪽 env 에 고정하고 나머지만 랜덤 충원.
    cfg.track_field.random_tracks = True
    cfg.track_field.pinned_tracks = tuple(
        t.strip() for t in args.tracks.split(",") if t.strip())
    # 시각화: 학습용 시간초과(에피소드 컷) 제거. 학습에서는 30초(=max_episode_length)
    # 마다 truncated 로 두 차가 동시에 리셋되지만, 순수 추론에는 인위적 컷이 불필요하다.
    # 이제 차량은 이탈/충돌/전복 시에만 개별 리스폰하고 그 외엔 창을 닫을 때까지 계속 주행한다.
    cfg.episode_length_s = 1.0e9
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
          f"device={device} deterministic={not args.stochastic}")
    picked = [env._field.track_names[g] for g in env._field.env_track_type]
    n_pin = min(len(cfg.track_field.pinned_tracks), N)
    if n_pin:
        print(f"[INFO] 트랙 배정: 지정 {picked[:n_pin]} + 랜덤 {picked[n_pin:]}")
    else:
        print(f"[INFO] 트랙 배정(랜덤): {picked}")

    # ---- 두 에이전트: 추론 전용 ----
    # use_compile=False: 소규모 추론에는 컴파일 워밍업(수십 초~수 분)이 손해.
    # risk 파라미터는 deterministic 행동 후보 선택(q_beta)에도 쓰이므로 학습과 동일하게.
    def make_agent(risk, seed):
        c = DACERppConfig(obs_dim=obs_dim, act_dim=act_dim, risk=risk,
                          device=device, use_compile=False, seed=seed)
        return DACERpp(c)

    agent_cvar = make_agent(conservative_cvar(args.cvar_eta), seed=0)   # car_a
    agent_pow = make_agent(aggressive_pow(args.pow_eta), seed=1)        # car_b
    agent_cvar.load(_cvar_ckpt)
    agent_pow.load(_pow_ckpt)
    print(f"[INFO] checkpoint loaded <- {args.ckpt_dir} "
          f"(cvar step={agent_cvar.step}, pow step={agent_pow.step})")

    # ---- 재생 루프 ----
    obs, _ = env.reset()
    oa, ob = obs["car_a"], obs["car_b"]
    deterministic = not args.stochastic

    # 에피소드 리턴 집계 (차량별 종료 = 이탈/충돌/전복 시점에 완료 처리)
    ep_ret_a = torch.zeros(N, device=env.device)
    ep_ret_b = torch.zeros(N, device=env.device)
    fin_a: list[float] = []   # 완료된 에피소드 리턴 기록
    fin_b: list[float] = []

    it = 0
    while simulation_app.is_running() and (args.steps == 0 or it < args.steps):
        # Car A(상대역)는 항상 결정적 — --stochastic 은 Car B 에만 적용
        aa = agent_cvar.act_batch(oa, deterministic=True)
        ab = agent_pow.act_batch(ob, deterministic=deterministic)
        action = torch.cat([aa, ab], dim=1)

        next_obs, _, _, _, _ = env.step(action)   # env 단위 done 미사용 (timeout 제거)
        di = env.dual_info()
        oa, ob = next_obs["car_a"], next_obs["car_b"]

        # 차량별 리스폰(이탈/충돌/전복) 시점을 각 차의 에피소드 경계로 삼는다.
        ep_ret_a += di["rew_a"]
        ep_ret_b += di["rew_b"]
        if di["term_a"].any():
            ids = di["term_a"].nonzero(as_tuple=False).squeeze(-1)
            fin_a.extend(ep_ret_a[ids].tolist())
            ep_ret_a[ids] = 0.0
        if di["term_b"].any():
            ids = di["term_b"].nonzero(as_tuple=False).squeeze(-1)
            fin_b.extend(ep_ret_b[ids].tolist())
            ep_ret_b[ids] = 0.0

        if it % args.log_every == 0:
            ra = sum(fin_a[-100:]) / len(fin_a[-100:]) if fin_a else float("nan")
            rb = sum(fin_b[-100:]) / len(fin_b[-100:]) if fin_b else float("nan")
            print(f"[{it}] ep_ret(최근100) CVaR={ra:.2f} Pow={rb:.2f} | "
                  f"r_a={di['rew_a'].mean().item():.3f} r_b={di['rew_b'].mean().item():.3f} | "
                  f"term={(di['term_a'] | di['term_b']).float().mean().item():.3f} "
                  f"(에피소드 {len(fin_a)}개 완료)")
        it += 1

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
