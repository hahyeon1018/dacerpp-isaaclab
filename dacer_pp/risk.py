"""Distortion risk measure beta: [0,1] -> [0,1].

논문 3-B절:
  - 항등사상  beta(tau)=tau           : risk-neutral
  - Pow(eta, tau) (Eq.11):
        eta >= 0 : tau ** (1/(1+|eta|))            (공격적/낙관적)
        eta <  0 : 1 - (1-tau) ** (1/(1+|eta|))    (보수적/비관적)
  - CVaR(eta, tau) (Eq.12):
        eta * tau                                  (eta<1 이면 하위 분위수에 집중 -> 보수적)

beta 는 uniform 으로 뽑은 tau 를 왜곡하여,
  Q_beta(s,a) = E_{tau~U[0,1]}[ Z_{beta(tau)}(s,a) ]   (Eq.7)
의 기대값 추정에 사용한다.
"""
from __future__ import annotations

import torch

from .config import RiskConfig


def neutral(tau: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
    return tau


def cvar(tau: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
    # CVaR(eta, tau) = eta * tau, 0 < eta <= 1
    return eta * tau


def pow_distortion(tau: torch.Tensor, eta: float = 0.0) -> torch.Tensor:
    p = 1.0 / (1.0 + abs(eta))
    if eta >= 0:
        return torch.clamp(tau, 0.0, 1.0) ** p
    else:
        return 1.0 - torch.clamp(1.0 - tau, 0.0, 1.0) ** p


_MEASURES = {
    "neutral": neutral,
    "cvar": cvar,
    "pow": pow_distortion,
}


def distort(tau: torch.Tensor, cfg: RiskConfig) -> torch.Tensor:
    """RiskConfig 에 따라 tau 를 왜곡해 반환."""
    fn = _MEASURES.get(cfg.measure)
    if fn is None:
        raise ValueError(f"Unknown risk measure: {cfg.measure}")
    return fn(tau, cfg.eta)
