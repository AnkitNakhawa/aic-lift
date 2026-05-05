"""Custom SAC trainer.

Phase 1: image + proprio replay buffer, auxiliary detection loss.
Phase 2: proprio-only replay buffer (no images), no detection loss.
         Much smaller memory footprint; can use a large buffer cheaply.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .networks import SACActor, SACCritic, P1_ACTION_DIM, P2_ACTION_DIM, P2_OBS_DIM

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Replay Buffer ────────────────────────────────────────────────────────────


class ReplayBuffer:
    """Circular replay buffer.

    Phase 1: stores images (uint8) + proprio.
    Phase 2: stores proprio only (no images) — large buffers are cheap.
    """

    def __init__(
        self,
        capacity: int,
        phase: int,
        img_shape=(84, 84, 4),
        proprio_dim: int = 7,
    ):
        self.capacity = capacity
        self.phase = phase
        self.proprio_dim = proprio_dim
        self.action_dim = P1_ACTION_DIM if phase == 1 else P2_ACTION_DIM

        # Phase 1: 1 camera ("image"); Phase 2: no cameras
        self._cam_keys = ["image"] if phase == 1 else []

        if self._cam_keys:
            self._images = {
                k: np.zeros((capacity, *img_shape), dtype=np.uint8)
                for k in self._cam_keys
            }
            self._images_next = {
                k: np.zeros((capacity, *img_shape), dtype=np.uint8)
                for k in self._cam_keys
            }
        else:
            self._images = {}
            self._images_next = {}

        self._proprios = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self._proprios_next = np.zeros((capacity, proprio_dim), dtype=np.float32)
        self._actions = np.zeros((capacity, self.action_dim), dtype=np.float32)
        self._rewards = np.zeros((capacity, 1), dtype=np.float32)
        self._dones = np.zeros((capacity, 1), dtype=np.float32)
        # GT detection labels only used in phase 1
        self._gt_uvs = np.full((capacity, 2), np.nan, dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def add(
        self,
        obs: dict | np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: dict | np.ndarray,
        done: bool,
        gt_uv: np.ndarray | None = None,
    ) -> None:
        i = self._ptr
        # Phase 2 obs is a flat np.ndarray; Phase 1 is a dict with images
        if self.phase == 1:
            for k in self._cam_keys:
                self._images[k][i] = obs[k]
                self._images_next[k][i] = next_obs[k]
            self._proprios[i] = obs["proprio"]
            self._proprios_next[i] = next_obs["proprio"]
        else:
            self._proprios[i] = obs
            self._proprios_next[i] = next_obs

        self._actions[i] = action
        self._rewards[i] = reward
        self._dones[i] = float(done)
        self._gt_uvs[i] = gt_uv if gt_uv is not None else np.array([np.nan, np.nan])
        self._ptr = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple:
        idx = np.random.randint(0, self._size, size=batch_size)

        if self.phase == 1:
            obs = {
                k: _to_tensor(self._images[k][idx], device)
                for k in self._cam_keys
            }
            obs["proprio"] = _f(self._proprios[idx], device)
            nxt = {
                k: _to_tensor(self._images_next[k][idx], device)
                for k in self._cam_keys
            }
            nxt["proprio"] = _f(self._proprios_next[idx], device)
        else:
            proprio_t = _f(self._proprios[idx], device)
            proprio_next_t = _f(self._proprios_next[idx], device)
            obs = {"proprio": proprio_t}
            nxt = {"proprio": proprio_next_t}

        actions = _f(self._actions[idx], device)
        rewards = _f(self._rewards[idx], device)
        dones = _f(self._dones[idx], device)
        gt_uvs = _f(self._gt_uvs[idx], device)
        return obs, actions, rewards, nxt, dones, gt_uvs

    def __len__(self) -> int:
        return self._size


def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(arr).permute(0, 3, 1, 2).float().to(device) / 255.0


def _f(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(arr).to(device)


# ── SAC Trainer ─────────────────────────────────────────────────────────────


class SACTrainer:
    """Soft Actor-Critic.

    Phase 1: includes auxiliary port detection loss (CNN-based).
    Phase 2: standard SAC only, no detection loss (no cameras).
    """

    def __init__(
        self,
        phase: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_init: float = 0.1,
        det_loss_weight: float = 0.1,
        device: torch.device = DEVICE,
    ):
        self.phase = phase
        self.gamma = gamma
        self.tau = tau
        # det_loss only meaningful for phase 1
        self.det_loss_weight = det_loss_weight if phase == 1 else 0.0
        self.device = device

        self.actor = SACActor(phase).to(device)
        self.critic = SACCritic(phase).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        action_dim = P1_ACTION_DIM if phase == 1 else P2_ACTION_DIM
        self.log_alpha = torch.tensor(
            np.log(alpha_init), requires_grad=True, device=device
        )
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = -float(action_dim)

        self._train_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update(self, buffer: ReplayBuffer, batch_size: int = 256) -> dict:
        obs, actions, rewards, next_obs, dones, gt_uvs_batch = buffer.sample(
            batch_size, self.device
        )

        # ── Critic update ──────────────────────────────────────────────────
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor(next_obs)
            q1_next, q2_next = self.critic_target(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ── Actor update ───────────────────────────────────────────────────
        new_actions, log_pi, det_uv = self.actor(obs)
        q1_pi, q2_pi = self.critic(obs, new_actions)
        actor_loss = (self.alpha.detach() * log_pi - torch.min(q1_pi, q2_pi)).mean()

        det_loss = torch.tensor(0.0, device=self.device)
        if self.phase == 1 and self.det_loss_weight > 0:
            valid_mask = ~torch.isnan(gt_uvs_batch).any(dim=1)
            if valid_mask.any():
                det_loss = F.mse_loss(det_uv[valid_mask], gt_uvs_batch[valid_mask])
            actor_loss = actor_loss + self.det_loss_weight * det_loss

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ── Temperature update ─────────────────────────────────────────────
        alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ── Soft target update ─────────────────────────────────────────────
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.copy_(self.tau * p.data + (1.0 - self.tau) * pt.data)

        self._train_steps += 1
        return {
            "critic_loss": float(critic_loss),
            "actor_loss": float(actor_loss),
            "det_loss": float(det_loss),
            "alpha": float(self.alpha),
            "alpha_loss": float(alpha_loss),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "log_alpha": self.log_alpha,
                "train_steps": self._train_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.log_alpha = ckpt["log_alpha"].to(self.device).requires_grad_(True)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=1e-4)
        self._train_steps = ckpt.get("train_steps", 0)
