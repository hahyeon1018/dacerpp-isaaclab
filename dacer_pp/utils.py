"""소소한 유틸리티."""
from __future__ import annotations

import torch.nn as nn


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """target <- tau*source + (1-tau)*target  (논문: phi' <- rho*phi' + (1-rho)*phi)."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(tau * sp.data)
    for tb, sb in zip(target.buffers(), source.buffers()):
        tb.data.copy_(sb.data)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())
