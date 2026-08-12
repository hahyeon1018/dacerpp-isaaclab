"""DACER++ 에이전트 (PyTorch).

업로드본을 그대로 유지하되, Isaac Lab 다중환경 루프를 위해
torch 텐서를 직접 받는 act_batch() 를 추가했다(numpy 왕복 제거).

성능:
  - 역확산 샘플링 루프(_build_sampler)를 torch.compile 로 컴파일한다.
      * no_grad 경로(행동 선택/타깃 행동): cfg.compile_mode(기본 "reduce-overhead",
        CUDA Graphs 포함)로 커널 런치 오버헤드를 제거.
      * 그래디언트가 흐르는 경로(액터 손실): cudagraph 없는 default 모드로 컴파일
        (backward + eager 옵티마이저와의 상호작용을 안전하게 유지).
  - QVN forward 도 default 모드로 컴파일(fusion). 옵티마이저/체크포인트는
    원본 모듈을 그대로 사용하므로 저장 포맷은 변하지 않는다.
  - update() 가 반환하는 지표는 0-d 텐서(detach)로 두어 매 스텝 GPU 동기화를 피한다.
    로깅 시점에 float() 로 변환할 것.
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn

from .config import DACERppConfig
from .diffusion import GaussianDiffusion
from .entropy import estimate_entropy_torch
from .losses import quantile_huber_loss
from .networks import DiffusionPolicyNet, QuantileValueNetwork
from .replay_buffer import Batch
from .risk import distort
from .utils import hard_update, soft_update


def _compile_available(device: torch.device) -> bool:
    """torch.compile 이 이 환경에서 실제로 동작하는지 1회 프로브."""
    if not hasattr(torch, "compile"):
        return False
    try:
        fn = torch.compile(lambda x: x * 2.0, dynamic=False)
        fn(torch.ones(2, device=device))
        return True
    except Exception:
        return False


class DACERpp:
    def __init__(self, cfg: DACERppConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.step = 0
        self._entropy = 0.0
        # 명시 지정(cfg.target_entropy)이 있으면 그것을, 없으면 구 규칙을 쓴다.
        # 구 규칙(-act_dim*0.9=-1.8)은 이 정책이 도달할 수 없는 값이라 alpha 를
        # 상한에 고정시킨다 — config.DACERppConfig.target_entropy 주석 참조.
        self.target_entropy = (float(cfg.target_entropy) if cfg.target_entropy is not None
                               else -cfg.act_dim * cfg.target_entropy_scale)
        torch.manual_seed(cfg.seed)

        def make_qvn():
            return QuantileValueNetwork(
                cfg.obs_dim, cfg.act_dim, cfg.hidden_sizes,
                embedding_dim=cfg.embedding_dim, num_cosines=cfg.num_cosines,
                activation=cfg.activation).to(self.device)

        self.qvn1, self.qvn2 = make_qvn(), make_qvn()
        self.target_qvn1, self.target_qvn2 = make_qvn(), make_qvn()
        hard_update(self.target_qvn1, self.qvn1)
        hard_update(self.target_qvn2, self.qvn2)

        self.policy = DiffusionPolicyNet(
            cfg.obs_dim, cfg.act_dim, cfg.diffusion_hidden_sizes,
            time_dim=cfg.time_dim, activation=cfg.activation).to(self.device)

        self.log_alpha = nn.Parameter(torch.tensor(float(cfg.init_log_alpha), device=self.device))
        self.diffusion = GaussianDiffusion(cfg.num_timesteps, device=self.device)

        self.critic1_opt = torch.optim.Adam(self.qvn1.parameters(), lr=cfg.critic_lr)
        self.critic2_opt = torch.optim.Adam(self.qvn2.parameters(), lr=cfg.critic_lr)
        self.actor_opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.actor_lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.alpha_lr)

        # ---- torch.compile / CUDA Graphs ----
        self._sample_eager = self._build_sampler()
        self._sample_nograd = self._sample_eager   # no_grad 행동 샘플링 (CUDA Graphs)
        self._sample_grad = self._sample_eager     # 액터 손실용 (미분 가능)
        self._q1, self._q2 = self.qvn1, self.qvn2
        self._tq1, self._tq2 = self.target_qvn1, self.target_qvn2
        if cfg.use_compile and _compile_available(self.device):
            self._sample_nograd = torch.compile(self._sample_eager, mode=cfg.compile_mode, dynamic=False)
            self._sample_grad = torch.compile(self._sample_eager, dynamic=False)
            self._q1 = torch.compile(self.qvn1, dynamic=False)
            self._q2 = torch.compile(self.qvn2, dynamic=False)
            self._tq1 = torch.compile(self.target_qvn1, dynamic=False)
            self._tq2 = torch.compile(self.target_qvn2, dynamic=False)

    # ----------------------------- 행동 생성 ----------------------------- #
    def _build_sampler(self):
        """역확산 샘플링 체인을 하나의 함수로 인라인(컴파일 대상).

        p_sample 과 동일한 수식. T 가 파이썬 상수라 컴파일 시 루프가 풀려
        전체 체인이 단일 그래프(CUDA Graph 캡처 가능)가 된다.
        """
        policy = self.policy
        d = self.diffusion
        T = self.cfg.num_timesteps
        act_dim = self.cfg.act_dim

        def sample(obs: torch.Tensor) -> torch.Tensor:
            x = torch.randn(obs.shape[0], act_dim, device=obs.device, dtype=obs.dtype)
            for ti in range(T - 1, -1, -1):
                eps = policy(obs, x, d.ts[ti])
                x0 = (x * d.sqrt_recip_alphas_cumprod[ti]
                      - eps * d.sqrt_recipm1_alphas_cumprod[ti]).clamp(-1.0, 1.0)
                mean = x0 * d.posterior_mean_coef1[ti] + x * d.posterior_mean_coef2[ti]
                if ti > 0:
                    x = mean + d.posterior_std[ti] * torch.randn_like(x)
                else:
                    x = mean
            return x

        return sample

    def _diffusion_action(self, obs):
        fn = self._sample_grad if torch.is_grad_enabled() else self._sample_nograd
        try:
            return fn(obs)
        except Exception:
            if fn is self._sample_eager:
                raise
            # 컴파일 경로 실패 시 eager 로 폴백(이후 호출도 eager)
            print("[DACERpp] compiled sampler failed; falling back to eager.")
            self._sample_nograd = self._sample_grad = self._sample_eager
            return self._sample_eager(obs)

    def _add_explore_noise(self, action):
        noise = torch.randn_like(action) * torch.exp(self.log_alpha.detach()) * self.cfg.explore_noise_scale
        return torch.clamp(action + noise, -1.0, 1.0)

    def q_beta(self, qvn, obs, act):
        K = self.cfg.risk.num_quantiles
        taus = torch.rand(obs.shape[0], K, device=self.device)
        taus = distort(taus, self.cfg.risk)
        z = qvn(obs, act, taus)
        return z.mean(dim=1)

    @torch.no_grad()
    def select_action(self, obs, with_noise: bool, num_candidates: Optional[int] = None):
        M = num_candidates if num_candidates is not None else self.cfg.num_action_candidates
        B = obs.shape[0]
        if M <= 1:
            act = self._diffusion_action(obs)
            return self._add_explore_noise(act) if with_noise else torch.clamp(act, -1, 1)
        obs_rep = obs.unsqueeze(1).expand(B, M, obs.shape[-1]).reshape(B * M, obs.shape[-1])
        act = self._diffusion_action(obs_rep)
        act = self._add_explore_noise(act) if with_noise else torch.clamp(act, -1, 1)
        q = torch.min(self.q_beta(self._q1, obs_rep, act), self.q_beta(self._q2, obs_rep, act))
        q = q.reshape(B, M)
        act = act.reshape(B, M, self.cfg.act_dim)
        best = q.argmax(dim=1)
        return act[torch.arange(B, device=self.device), best]

    @torch.no_grad()
    def act_batch(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Isaac Lab 용: (N,obs_dim) torch -> (N,act_dim) torch. device 내에서 처리."""
        if obs.device != self.device:
            obs = obs.to(self.device)
        return self.select_action(obs, with_noise=not deterministic)

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """ROS2/단일환경 호환용 (numpy)."""
        single = obs_np.ndim == 1
        obs = torch.as_tensor(np.atleast_2d(obs_np).astype(np.float32), device=self.device)
        action = self.select_action(obs, with_noise=not deterministic).cpu().numpy()
        return action[0] if single else action

    # ----------------------------- 학습 ----------------------------- #
    def _critic_loss(self, batch: Batch):
        cfg = self.cfg
        obs, action = batch.obs, batch.action
        reward = batch.reward * cfg.reward_scale
        next_obs, done = batch.next_obs, batch.done
        B = obs.shape[0]
        with torch.no_grad():
            next_action = self.select_action(next_obs, with_noise=True)
            tau_t = torch.rand(B, cfg.num_target_quantiles, device=self.device)
            z1 = self._tq1(next_obs, next_action, tau_t)
            z2 = self._tq2(next_obs, next_action, tau_t)
            pick1 = (z1.mean(dim=1, keepdim=True) <= z2.mean(dim=1, keepdim=True)).float()
            target_z = pick1 * z1 + (1.0 - pick1) * z2
            q_backup = reward.unsqueeze(1) + (1.0 - done).unsqueeze(1) * cfg.gamma * target_z

        def one_critic(qvn_c, opt):
            taus = torch.rand(B, cfg.num_quantiles, device=self.device)
            cur_z = qvn_c(obs, action, taus)
            loss = quantile_huber_loss(cur_z, q_backup, taus, kappa=cfg.huber_kappa)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                return loss.detach(), cur_z.mean().detach(), cur_z.std(dim=1).mean().detach()

        l1, m1, s1 = one_critic(self._q1, self.critic1_opt)
        l2, m2, s2 = one_critic(self._q2, self.critic2_opt)
        return {"q1_loss": l1, "q2_loss": l2, "q1_mean": m1, "q2_mean": m2, "q1_std": s1, "q2_std": s2}

    def _actor_loss(self, batch: Batch):
        action = self._diffusion_action(batch.obs)
        q = torch.min(self.q_beta(self._q1, batch.obs, action),
                      self.q_beta(self._q2, batch.obs, action))
        loss = -q.mean()
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_opt.step()
        return {"policy_loss": loss.detach()}

    def _maybe_update_alpha(self, batch: Batch):
        cfg = self.cfg
        if self.step % cfg.delay_alpha_update == 0:
            with torch.no_grad():
                S = cfg.entropy_num_samples
                # 배치 자체가 균등 샘플이므로 앞쪽 n개도 균등 샘플이다
                n = min(cfg.entropy_obs_batch, batch.obs.shape[0])
                obs_e = batch.obs[:n]
                obs_rep = obs_e.unsqueeze(1).expand(n, S, cfg.obs_dim).reshape(n * S, cfg.obs_dim)
                # 엔트로피는 '정책 자체'(diffusion 샘플)의 다양성만으로 추정한다.
                # 각 상태에서 S개를 뽑으면 역확산의 posterior_std 노이즈로 정책 고유
                # 확률성이 표본에 담긴다. 여기에 alpha-비례 탐험 노이즈를 더하면(구
                # 코드) 엔트로피가 노이즈로 채워져, 정책이 붕괴(표본 std≈0.02)해도
                # H_hat 이 타깃을 만족해 alpha 가 반응하지 못하는 퇴행 평형에 갇힌다
                # (2026-07-30 실차 분석 원인 B: H_hat 의 91%가 주입 노이즈, 정책 9%).
                # -> alpha 를 정책 붕괴에 반응시키려면 노이즈를 빼야 한다(사용자 지적).
                # 포화 시 GMM 성분 특이화는 estimate_entropy_torch 의 reg_covar +
                # robust cholesky 가, 가짜-저엔트로피로 인한 alpha 폭주는 max_log_alpha
                # 캡이 막는다(둘 다 이미 존재 — 구 코드의 노이즈 트릭은 캡 도입으로 불필요).
                acts = self._diffusion_action(obs_rep).reshape(n, S, cfg.act_dim)
                # GPU 배치 EM (sklearn/CPU 왕복 제거)
                self._entropy = estimate_entropy_torch(acts, num_components=cfg.entropy_num_components)
            alpha_loss = -(self.log_alpha * (self.target_entropy - self._entropy))
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            with torch.no_grad():  # 상한: 퇴행 평형 재진입 차단 / 하한: 수치 폭주 방지
                self.log_alpha.clamp_(cfg.min_log_alpha, cfg.max_log_alpha)
        return {"alpha": torch.exp(self.log_alpha.detach()), "entropy": float(self._entropy)}

    def update(self, batch: Batch) -> Dict[str, float]:
        info = {}
        info.update(self._critic_loss(batch))
        if self.step % self.cfg.delay_update == 0:
            info.update(self._actor_loss(batch))
            soft_update(self.target_qvn1, self.qvn1, self.cfg.tau)
            soft_update(self.target_qvn2, self.qvn2, self.cfg.tau)
        info.update(self._maybe_update_alpha(batch))
        self.step += 1
        return info

    # ----------------------------- 저장/복원 ----------------------------- #
    def state_dict(self):
        return {"qvn1": self.qvn1.state_dict(), "qvn2": self.qvn2.state_dict(),
                "target_qvn1": self.target_qvn1.state_dict(),
                "target_qvn2": self.target_qvn2.state_dict(),
                "policy": self.policy.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(), "step": self.step, "cfg": self.cfg}

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location=None):
        sd = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.qvn1.load_state_dict(sd["qvn1"]); self.qvn2.load_state_dict(sd["qvn2"])
        self.target_qvn1.load_state_dict(sd["target_qvn1"])
        self.target_qvn2.load_state_dict(sd["target_qvn2"])
        self.policy.load_state_dict(sd["policy"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"].to(self.device))
        self.step = sd.get("step", 0)
        return self
