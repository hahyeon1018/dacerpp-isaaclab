"""Gaussian diffusion (DDPM) — JAX 레퍼런스(diffusion.py) 의 PyTorch 포팅.

DACER 정책은 역확산 샘플링 p_sample 으로 행동을 생성하며,
정책 갱신 시 이 샘플링 체인 전체를 통해 그래디언트가 흐른다(미분 가능).
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch


def vp_beta_schedule(timesteps: int) -> np.ndarray:
    t = np.arange(1, timesteps + 1)
    T = timesteps
    b_max, b_min = 10.0, 0.1
    alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T**2)
    return 1.0 - alpha


class GaussianDiffusion:
    """num_timesteps 스텝의 VP-schedule DDPM."""

    def __init__(self, num_timesteps: int, device: torch.device | str = "cpu"):
        self.num_timesteps = num_timesteps
        betas = vp_beta_schedule(num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        def t(x):
            return torch.as_tensor(x, dtype=torch.float32, device=device)

        self.betas = t(betas)
        self.alphas_cumprod = t(alphas_cumprod)
        self.sqrt_alphas_cumprod = t(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = t(np.sqrt(1.0 - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = t(np.sqrt(1.0 / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = t(np.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_log_variance_clipped = t(np.log(np.maximum(posterior_variance, 1e-20)))
        self.posterior_std = t(np.sqrt(np.maximum(posterior_variance, 1e-20)))
        self.posterior_mean_coef1 = t(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_coef2 = t((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))
        # 스텝 인덱스 텐서(핫루프에서 host->device 복사 제거)
        self.ts = torch.arange(num_timesteps, device=device)
        self.device = torch.device(device)

    def to(self, device):
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        self.device = torch.device(device)
        return self

    def p_mean_variance(self, t: int, x: torch.Tensor, noise_pred: torch.Tensor):
        x_recon = x * self.sqrt_recip_alphas_cumprod[t] - noise_pred * self.sqrt_recipm1_alphas_cumprod[t]
        x_recon = torch.clamp(x_recon, -1.0, 1.0)
        model_mean = x_recon * self.posterior_mean_coef1[t] + x * self.posterior_mean_coef2[t]
        model_log_variance = self.posterior_log_variance_clipped[t]
        return model_mean, model_log_variance

    def p_sample(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: Tuple[int, ...],
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """noise -> action. model(t, x) 는 예측 노이즈를 반환.

        이 함수는 미분 가능(정책 갱신 시 그래디언트가 통과)하다.
        """
        x = torch.randn(shape, device=self.device, generator=generator)
        for ti in reversed(range(self.num_timesteps)):
            noise_pred = model(self.ts[ti], x)
            model_mean, _ = self.p_mean_variance(ti, x, noise_pred)
            if ti > 0:
                noise = torch.randn(shape, device=self.device, generator=generator)
                x = model_mean + self.posterior_std[ti] * noise
            else:
                x = model_mean
        return x

    def q_sample(self, t: torch.Tensor, x_start: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # t: (B,) 정수 인덱스
        sa = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        so = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        return sa * x_start + so * noise

    def p_loss(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        t: torch.Tensor,
        x_start: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """확산 노이즈 예측 손실(behavior cloning 식 정책 사전학습용, 선택적)."""
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(t, x_start, noise)
        noise_pred = model(t, x_noisy)
        se = 0.5 * (noise_pred - noise) ** 2
        if weights is not None:
            se = weights.reshape(-1, 1) * se
        return se.mean()
