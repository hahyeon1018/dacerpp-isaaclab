"""GPU 텐서 원형(ring) 리플레이 버퍼.

업로드본(numpy, 단일 transition add)을 Isaac Lab 다중환경용으로 재작성.
- 모든 버퍼를 device(GPU) 위에 둔다 (CPU<->GPU 복사 병목 제거).
- add_batch 로 N개 transition 을 한 번에 저장한다(병렬 환경 1스텝 = N개).
- sample 은 device 위에서 인덱싱하여 Batch 를 그대로 반환한다.
"""
from __future__ import annotations
from typing import NamedTuple
import torch


class Batch(NamedTuple):
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor


class TensorReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int,
                 device: str = "cuda:0", seed: int = 0):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.obs = torch.zeros((self.capacity, obs_dim), device=self.device)
        self.action = torch.zeros((self.capacity, act_dim), device=self.device)
        self.reward = torch.zeros((self.capacity,), device=self.device)
        self.next_obs = torch.zeros((self.capacity, obs_dim), device=self.device)
        self.done = torch.zeros((self.capacity,), device=self.device)
        self.ptr = 0
        self.size = 0
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

    @torch.no_grad()
    def add_batch(self, obs, action, reward, next_obs, done):
        """obs/next_obs:(N,obs_dim) action:(N,act_dim) reward/done:(N,)."""
        n = obs.shape[0]
        idx = (torch.arange(n, device=self.device) + self.ptr) % self.capacity
        self.obs[idx] = obs
        self.action[idx] = action
        self.reward[idx] = reward.reshape(-1)
        self.next_obs[idx] = next_obs
        self.done[idx] = done.reshape(-1).float()
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = int(min(self.size + n, self.capacity))

    @torch.no_grad()
    def sample(self, batch_size: int) -> Batch:
        idx = torch.randint(0, self.size, (batch_size,), generator=self.gen, device=self.device)
        return Batch(self.obs[idx], self.action[idx], self.reward[idx],
                     self.next_obs[idx], self.done[idx])

    def __len__(self):
        return self.size
