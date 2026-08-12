"""Quantile regression Huber loss (논문 Eq.5 / Eq.9).

rho^kappa_tau(delta) = | tau - 1{delta<0} | * L_kappa(delta) / kappa
  - L_kappa : Huber loss (임계값 kappa)
  - delta_ij = (target Z_{tau'_j}) - (current Z_{tau_i})  의 쌍별 TD 오차

손실 = (1/N') sum_i sum_j rho^kappa_{tau_i}(delta_ij)
     = sum over i (current quantile),  mean over j (target quantile),  mean over batch.
"""
from __future__ import annotations

import torch


def quantile_huber_loss(
    current_z: torch.Tensor,   # (B, N)   Z_{tau_i}(s,a)
    target_z: torch.Tensor,    # (B, N')  Z_{tau'_j}(s',a*) (이미 detach 및 r+gamma 적용)
    taus: torch.Tensor,        # (B, N)   tau_i
    kappa: float = 1.0,
) -> torch.Tensor:
    # 쌍별 TD 오차 delta_ij = target_j - current_i  -> (B, N, N')
    delta = target_z.unsqueeze(1) - current_z.unsqueeze(2)

    abs_delta = delta.abs()
    huber = torch.where(
        abs_delta <= kappa,
        0.5 * delta.pow(2),
        kappa * (abs_delta - 0.5 * kappa),
    )  # L_kappa(delta_ij)

    # | tau_i - 1{delta_ij < 0} |
    indicator = (delta.detach() < 0).float()                 # (B, N, N')
    weight = (taus.unsqueeze(2) - indicator).abs()           # (B, N, N')

    rho = weight * huber / kappa                             # (B, N, N')

    # target 분위수(j)에 대해 평균, current 분위수(i)에 대해 합, 배치 평균
    loss = rho.mean(dim=2).sum(dim=1).mean()
    return loss
