"""GMM 기반 diffusion 정책 엔트로피 추정.

diffusion 정책은 명시적 밀도가 없으므로, 상태별로 행동 표본을 여러 개 뽑아
가우시안 혼합(GMM)을 적합한 뒤 그 미분 엔트로피를 추정한다.
이 추정 엔트로피 H_hat 으로 온도 alpha 를 적응 갱신한다(논문 Eq.18).

estimate_entropy_torch : 배치 EM 을 GPU 에서 한 번에 수행 (기본 경로).
  - sklearn 의존 없음, CPU 왕복 없음. 상태 B개를 병렬로 적합.
  - 엔트로피는 sklearn 판과 동일한 혼합 상한 근사:
      H ≈ -sum_k w_k log w_k + sum_k w_k * H(N(mu_k, Sigma_k))
estimate_entropy      : 구 sklearn 경로 (numpy 입력, 호환용으로 유지).
"""
from __future__ import annotations

import math

import numpy as np
import torch


def _safe_cholesky(cov: torch.Tensor, fallback: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,K,d,d) 배치 cholesky. 비-PD 컴포넌트는 fallback(등방 대각) 공분산으로 복구.

    수치 오차(특히 근사-특이 공분산)로 개별 컴포넌트가 PD 를 벗어나도
    전체 추정을 죽이지 않는다. 반환: (L, 복구된 cov).
    """
    L, info = torch.linalg.cholesky_ex(cov)
    if bool((info > 0).any()):
        bad = (info > 0).unsqueeze(-1).unsqueeze(-1)               # (B,K,1,1)
        cov = torch.where(bad, fallback, cov)
        L, info = torch.linalg.cholesky_ex(cov)
        if bool((info > 0).any()):  # 이론상 도달 불가(등방 대각은 항상 PD)
            bad = (info > 0).unsqueeze(-1).unsqueeze(-1)
            eye = torch.eye(cov.shape[-1], dtype=cov.dtype, device=cov.device)
            cov = torch.where(bad, eye.expand_as(cov), cov)
            L = torch.linalg.cholesky(cov)
    return L, cov


@torch.no_grad()
def estimate_entropy_torch(
    actions: torch.Tensor,
    num_components: int = 3,
    num_iters: int = 30,
    reg_covar: float = 1e-6,
) -> float:
    """actions: (B, S, d) torch 텐서 -> 상태 평균 미분 엔트로피(float).

    B개 상태 각각에 K-컴포넌트 full-covariance GMM 을 배치 EM 으로 적합한다.

    EM 은 float64 로 수행한다(sklearn 과 동일 정밀도). float32 로 하면 TF32 가
    켜진 환경(train.py)에서 공분산 einsum 이 TF32 행렬곱(상대오차 ~1e-3)으로
    실행되어, 행동이 clamp 경계에 포화된 근사-특이 분포에서 공분산이 비-PD 가
    되어 cholesky 가 실패한다. float64 에는 TF32 가 적용되지 않는다.
    """
    # 정책 발산(NaN) 시에도 추정기가 프로세스를 죽이지 않도록 소독
    X = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).double()
    B, S, d = X.shape
    K = min(num_components, max(1, S // 2))
    dev = X.device
    eyed = torch.eye(d, dtype=X.dtype, device=dev)

    # ---- 초기화: 등간격 표본을 평균으로, 공분산은 데이터 분산 * I ----
    stride = max(1, S // K)
    mu = X[:, : K * stride : stride, :].clone()                    # (B,K,d)
    var = X.var(dim=1, unbiased=False).mean(dim=-1)                # (B,)
    var = var.clamp_min(reg_covar).view(B, 1, 1, 1)
    cov = (eyed * var).expand(B, K, d, d).contiguous()             # (B,K,d,d)
    fallback = (eyed * (var + reg_covar)).expand(B, K, d, d)       # 복구용 등방 공분산
    log_w = torch.full((B, K), -math.log(K), dtype=X.dtype, device=dev)

    log2pi = d * math.log(2.0 * math.pi)
    reg = eyed * reg_covar

    for _ in range(num_iters):
        # ---- E-step: log responsibility ----
        L, cov = _safe_cholesky(cov + reg, fallback)               # (B,K,d,d)
        diff = X.unsqueeze(1) - mu.unsqueeze(2)                    # (B,K,S,d)
        y = torch.linalg.solve_triangular(L.unsqueeze(2), diff.unsqueeze(-1), upper=False)
        maha = y.squeeze(-1).pow(2).sum(-1)                        # (B,K,S)
        logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)  # (B,K)
        logpdf = -0.5 * (log2pi + logdet.unsqueeze(-1) + maha)     # (B,K,S)
        log_r = log_w.unsqueeze(-1) + logpdf
        log_r = log_r - torch.logsumexp(log_r, dim=1, keepdim=True)
        r = log_r.exp()                                            # (B,K,S)

        # ---- M-step ----
        Nk = r.sum(-1).clamp_min(1e-8)                             # (B,K)
        log_w = torch.log(Nk / S)
        mu = torch.einsum("bks,bsd->bkd", r, X) / Nk.unsqueeze(-1)
        diff = X.unsqueeze(1) - mu.unsqueeze(2)                    # (B,K,S,d)
        cov = torch.einsum("bks,bksd,bkse->bkde", r, diff, diff) / Nk.unsqueeze(-1).unsqueeze(-1)
        cov = 0.5 * (cov + cov.transpose(-1, -2))                  # 수치 대칭화
        cov = cov + reg

    # ---- 혼합 엔트로피 (sklearn 판과 동일한 상한 근사) ----
    w = log_w.exp()
    L, _ = _safe_cholesky(cov + reg, fallback)
    logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)      # (B,K)
    comp_ent = 0.5 * d * (1.0 + math.log(2.0 * math.pi)) + 0.5 * logdet       # (B,K)
    ent = -(w * torch.log(w + 1e-12)).sum(-1) + (w * comp_ent).sum(-1)        # (B,)
    ent = torch.nan_to_num(ent, nan=0.0, posinf=0.0, neginf=0.0)
    return float(ent.mean().item())


def estimate_entropy(actions: np.ndarray, num_components: int = 3) -> float:
    """구 sklearn 경로: actions (batch, num_samples, act_dim) numpy -> float. 호환용."""
    from sklearn.mixture import GaussianMixture

    total = []
    for action in actions:  # (num_samples, act_dim)
        # 표본이 너무 적거나 컴포넌트보다 적으면 안전하게 처리
        n_comp = min(num_components, max(1, action.shape[0] // 2))
        try:
            gmm = GaussianMixture(n_components=n_comp, covariance_type="full", random_state=42)
            gmm.fit(action)
        except Exception:
            total.append(0.0)
            continue
        weights = gmm.weights_
        entropies = []
        for i in range(gmm.n_components):
            cov = gmm.covariances_[i]
            d = cov.shape[0]
            ent = 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * np.linalg.slogdet(cov)[1]
            entropies.append(ent)
        ent = -np.sum(weights * np.log(weights + 1e-12)) + np.sum(weights * np.array(entropies))
        total.append(float(ent))
    return float(sum(total) / len(total)) if total else 0.0
