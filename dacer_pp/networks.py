"""DACER++ 신경망 (PyTorch).

핵심:
  - QuantileValueNetwork : 멀티모달 가치 분포를 IQN 방식으로 표현하는 QVN.
        세 개의 신경망 psi, phi, f 로 구성된다(논문이 가치망을 psi, f, phi 로 표기).
            psi : (s,a) -> 특징벡터 psi(s,a) in R^E
            phi : tau  -> 분위수 임베딩 phi(tau) in R^E (코사인 기저 -> Linear -> ReLU)
            f   : (psi(s,a) ⊙ phi(tau)) -> 분위수 값 Z_tau(s,a) in R
        => Z_tau(s,a) = f( psi(s,a) ⊙ phi(tau) ),  tau ~ U[0,1].
  - DiffusionPolicyNet : DACER 의 diffusion 정책(노이즈 예측망). 시간 t 의 정현파 임베딩 사용.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name: str):
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "elu": nn.ELU}[name]


def mlp(in_dim: int, hidden_sizes: Sequence[int], out_dim: int, activation: str) -> nn.Sequential:
    act = _activation(activation)
    layers = []
    last = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(last, h), act()]
        last = h
    layers += [nn.Linear(last, out_dim)]
    return nn.Sequential(*layers)


class QuantileValueNetwork(nn.Module):
    """IQN 스타일 Quantile Value Network (QVN).

    psi, phi, f 세 개의 하위 신경망으로 멀티모달 상태-행동 리턴 분포를 표현한다.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        embedding_dim: int = 256,
        num_cosines: int = 64,
        activation: str = "relu",
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_cosines = num_cosines
        act = _activation(activation)

        # psi : (s,a) -> R^E  (상태-행동 특징)
        self.psi = mlp(obs_dim + act_dim, hidden_sizes, embedding_dim, activation)

        # phi : tau -> R^E   (코사인 기저 임베딩 후 선형사상 + ReLU)
        self.phi = nn.Sequential(nn.Linear(num_cosines, embedding_dim), act())

        # f : R^E -> R       (분위수 값 출력)
        self.f = mlp(embedding_dim, hidden_sizes, 1, activation)

        # 코사인 기저 인덱스 i = 1..num_cosines (버퍼로 보관)
        self.register_buffer(
            "cos_idx", torch.arange(1, num_cosines + 1, dtype=torch.float32) * math.pi
        )

    def feature(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """psi(s,a) : (B, E)"""
        return self.psi(torch.cat([obs, act], dim=-1))

    def quantile_embedding(self, taus: torch.Tensor) -> torch.Tensor:
        """phi(tau) : (..., N, E). taus shape (..., N)."""
        # cos(pi * i * tau)
        cos = torch.cos(taus.unsqueeze(-1) * self.cos_idx)  # (..., N, num_cosines)
        return self.phi(cos)  # (..., N, E)

    def forward(self, obs: torch.Tensor, act: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        """Z_{tau}(s,a) 계산.

        Args:
            obs:  (B, obs_dim)
            act:  (B, act_dim)
            taus: (B, N) 분위수 비율 in [0,1]
        Returns:
            (B, N) 분위수 값
        """
        psi = self.feature(obs, act)            # (B, E)
        phi = self.quantile_embedding(taus)     # (B, N, E)
        x = psi.unsqueeze(1) * phi              # (B, N, E)  Hadamard 결합
        z = self.f(x).squeeze(-1)               # (B, N)
        return z


class DiffusionPolicyNet(nn.Module):
    """DACER diffusion 정책 (역확산 노이즈 예측망).

    입력: obs, 잡음 섞인 action x_t, 확산 시간 t -> 예측 노이즈(또는 epsilon).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        time_dim: int = 16,
        activation: str = "relu",
    ):
        super().__init__()
        assert time_dim % 2 == 0
        self.act_dim = act_dim
        self.time_dim = time_dim
        act = _activation(activation)

        # 시간 임베딩 MLP (DACERPolicyNet 와 동일 구조)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2), act(), nn.Linear(time_dim * 2, time_dim)
        )
        self.net = mlp(obs_dim + act_dim + time_dim, hidden_sizes, act_dim, activation)

        half = time_dim // 2
        freq = torch.arange(half, dtype=torch.float32) / half
        self.register_buffer("inv_freq", 10000.0 ** (-freq))
        self.time_scale = 1.0 / math.sqrt(time_dim)

    def _time_embed(self, t: torch.Tensor, batch_shape) -> torch.Tensor:
        # t: scalar 또는 (B,) -> (B, time_dim)
        t = t.float().reshape(-1)  # (B,) or (1,)
        emb = t.unsqueeze(-1) * self.inv_freq  # (B, half)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1) * self.time_scale
        if emb.shape[0] == 1 and batch_shape[0] != 1:
            emb = emb.expand(batch_shape[0], -1)
        return self.time_mlp(emb)

    def forward(self, obs: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self._time_embed(t, obs.shape)              # (B, time_dim)
        inp = torch.cat([obs, x, te], dim=-1)
        return self.net(inp)
