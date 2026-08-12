"""DACER++ 하이퍼파라미터 정의. (업로드본 + 편의 프리셋)"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class RiskConfig:
    """Distortion risk measure beta(.) 설정 (논문 Eq.11/Eq.12)."""
    measure: str = "neutral"      # "neutral" | "cvar" | "pow"
    eta: float = 1.0
    num_quantiles: int = 32       # Q_beta 추정용 표본 수 K


@dataclass
class DACERppConfig:
    # ---- 차원 ----
    obs_dim: int = 0
    act_dim: int = 0
    # ---- 네트워크 ----
    hidden_sizes: Sequence[int] = field(default_factory=lambda: (256, 256))
    diffusion_hidden_sizes: Sequence[int] = field(default_factory=lambda: (256, 256))
    activation: str = "relu"
    embedding_dim: int = 256
    num_cosines: int = 64
    time_dim: int = 16
    # ---- 분포형 critic(QVN) ----
    num_quantiles: int = 32
    num_target_quantiles: int = 32
    huber_kappa: float = 1.0
    # ---- diffusion 정책 ----
    num_timesteps: int = 10        # DACER2 권장(효율). 병렬 환경에서 throughput에 직결. 원본 20
    # ---- 학습 일반 ----
    # 0.99(지평~3.3s)는 코너 탈출 속도가 다음 직선에서 회수하는 이득(2~3s 뒤)을
    # 과도하게 할인해 근시안적 코너링을 유발 -> 0.995(지평 ~6.6s)로 상향.
    gamma: float = 0.995
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    alpha_lr: float = 3e-2
    batch_size: int = 1024        # 병렬환경: 큰 배치 권장
    reward_scale: float = 0.2
    # ---- 엔트로피 조절 ----
    delay_alpha_update: int = 2000
    delay_update: int = 2
    # alpha 상한 = 탐험 노이즈 std 상한(= alpha*explore_noise_scale). 엔트로피를
    # 정책 자체 다양성으로 추정하게 바꾼 뒤(agent._maybe_update_alpha), diffusion
    # 정책의 고유 엔트로피가 구조적으로 타깃보다 훨씬 낮아 alpha 는 '항상' 이 상한에
    # 붙는다 = 상한이 곧 탐험 세기. 0.405(alpha 1.5, 노이즈 0.225)로 두니 상시 과탐험이
    # 되어, 새 급코너 트랙에서 '비관적 CVaR 크리틱'이 하위꼬리(크래시)에 지배당해
    # 1.35M 부근에서 발산했다(20260731, Car A 완전붕괴 / 낙관적 Pow 는 생존). 붕괴 없이
    # 크리틱이 살아 있던 20260726 의 안전 탐험(alpha 0.537, 노이즈 0.081)으로 되돌린다.
    # ★ 구 취지(고속에서 노이즈 std 0.45→과세로 Q 행동무관화 방지)는 그대로, 상한만 더 낮춤.
    init_log_alpha: float = 0.0           # alpha=1.0 (첫 alpha 갱신 때 상한으로 클램프)
    max_log_alpha: float = -0.62          # ln(0.538) -> 노이즈 std 0.081 (20260726 안전값)
    min_log_alpha: float = -4.6           # ln(0.01): 수치 폭주 방지
    target_entropy_scale: float = 0.9     # target_entropy=None 일 때만 사용 (-act_dim*scale)
    # ★2026-08-12 target_entropy 명시 지정.
    # 문제: 구 기본값 -act_dim*0.9 = -1.8 은 이 정책 파라미터화로 '도달 불가능'하다.
    #   20260808~20260810 네 런 전체 로그에서 추정 엔트로피는 -5.8 ~ -7.1 이었고,
    #   그 결과 alpha 손실이 항상 같은 방향으로만 밀려 alpha 가 모든 로그 시점에서
    #   정확히 exp(max_log_alpha)=0.5379 에 고정됐다 = 적응 루프가 상수로 퇴화.
    #   (config 구 주석도 "alpha 는 항상 상한에 붙는다"고 적고 있었다.)
    # 측정된 대응관계: 조향 다양성 pstd 0.030(초기, 건강) ↔ H ≈ -6.3,
    #                  pstd 0.0105(20260810 배포, 붕괴) ↔ H ≈ -7.1.
    # -> -6.3 으로 두면 붕괴 쪽(H<-6.3)에서는 alpha 를 올려 탐험을 유지하고,
    #    다양성이 회복되면(H>-6.3) alpha 를 내릴 '하향 권한'이 처음으로 생긴다.
    # ※ 이것만으로 붕괴가 막히지는 않는다 — 액터 손실(-Q.mean())에 엔트로피 항이
    #    없어 정책이 argmax 로 굳는 것은 DACER 구조상 정상이다. 붕괴 대응의 본체는
    #    train.py 의 pstd 기반 사전붕괴 체크포인트 보존이다.
    target_entropy: float | None = -6.3
    entropy_num_samples: int = 64     # 원본 200
    entropy_num_components: int = 3
    entropy_obs_batch: int = 256      # GMM 엔트로피 추정에 쓸 상태 수 (GPU EM, entropy.estimate_entropy_torch)
    explore_noise_scale: float = 0.15
    # ---- torch.compile / CUDA Graphs ----
    use_compile: bool = True          # 네트워크/샘플러 torch.compile 사용
    compile_mode: str = "reduce-overhead"  # no_grad 행동 샘플링용(CUDA Graphs 포함).
                                           # 그래디언트가 흐르는 경로는 안전하게 default 모드로 컴파일.
    # ---- 리스크 민감 정책 ----
    risk: RiskConfig = field(default_factory=RiskConfig)
    num_action_candidates: int = 1
    seed: int = 0
    device: str = "cuda:0"


# ----------------------------------------------------------------------------
# 두 주행 스타일 프리셋 (요구사항: 보수적=CVaR, 공격적=Pow)
# ----------------------------------------------------------------------------
def conservative_cvar(eta: float = 0.5) -> RiskConfig:
    """CVaR(eta): 0<eta<=1, 작을수록 하위 분위수에 집중 -> 보수적 주행."""
    return RiskConfig(measure="cvar", eta=eta)


def aggressive_pow(eta: float = 1.3) -> RiskConfig:
    """Pow(eta): eta>0 이면 상위 결과를 낙관 -> 공격적 주행.

    1.5 -> 1.2 (2026-07-12): 실차 배포 대상(Car B)의 낙관 왜곡 완화 —
    tau^(1/(1+eta)) 가 상위 분위수(추월 성공 시나리오)를 과대평가해
    통과 불가능한 좁은 폭(반폭 ~0.77m)에서도 파고들다 벽에 스치는 사고를
    유발했다. 공격성은 소폭 줄지만 실차 안전 우선."""
    return RiskConfig(measure="pow", eta=eta)
