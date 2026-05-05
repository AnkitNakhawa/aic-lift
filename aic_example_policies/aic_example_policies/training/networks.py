"""Neural network architectures for the 2-phase SAC insertion policy."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ── Observation dimensions ─────────────────────────────────────────────────
# Phase 1 (centering): center camera CNN (128) + xyz_rel (3) + ft_z (1)
#                      + port_type_onehot (2) + step_norm (1) = 135
P1_OBS_DIM = 135
P1_ACTION_DIM = 2  # (dx, dy)

# Phase 2 (insertion): no cameras; flat proprioception only.
# ft_history (12×6=72) + xyz_rel (3) + xyz_vel (3) + port_type (2)
# + step_norm (1) + depth (1) + prev_action (3) + f_mag (1) = 86
FT_HISTORY_LEN = 12
P2_OBS_DIM = FT_HISTORY_LEN * 6 + 3 + 3 + 2 + 1 + 1 + 3 + 1  # = 86
assert P2_OBS_DIM == 86
P2_ACTION_DIM = 3  # (dx_res, dy_res, dz_res) — residuals on top of base descent

# ── Velocity/scale constants (must match training) ─────────────────────────
XY_SCALE = 0.005        # Phase 1 lateral
XY_INS_SCALE = 0.001    # Phase 2 lateral residual per normalized unit
Z_INS_SCALE = 0.002     # Phase 2 Z residual per normalized unit
BASE_VZ = 0.001         # m/step constant downward bias added before SAC residual

LOG_STD_MIN = -5
LOG_STD_MAX = 2


# ── Building blocks (Phase 1 only) ─────────────────────────────────────────


class SpatialSoftmax(nn.Module):
    """Learnable expected position pooling: (B, C, H, W) → (B, 2*C).

    For each channel, computes the softmax over spatial locations and returns
    the expected (x, y) coordinate — a soft argmax that is differentiable and
    forces the CNN to represent spatial positions explicitly.
    """

    def __init__(
        self, channels: int, height: int, width: int, temperature: float = 1.0
    ):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.temperature = nn.Parameter(torch.ones(1) * temperature, requires_grad=True)
        xs = (
            torch.linspace(-1, 1, width)
            .view(1, 1, 1, width)
            .expand(1, channels, height, width)
            .clone()
        )
        ys = (
            torch.linspace(-1, 1, height)
            .view(1, 1, height, 1)
            .expand(1, channels, height, width)
            .clone()
        )
        self.register_buffer("xs", xs)
        self.register_buffer("ys", ys)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        flat = x.view(B, C, -1) / self.temperature
        weights = F.softmax(flat, dim=2).view(B, C, H, W)
        ex = (weights * self.xs).sum(dim=(2, 3))
        ey = (weights * self.ys).sum(dim=(2, 3))
        return torch.cat([ex, ey], dim=1)  # (B, 2*C)


class CNNBackbone(nn.Module):
    """Small CNN: (B, 4, 84, 84) → (B, 128).

    Conv(4→32, 8×8, s4) → ReLU   # 84 → 20
    Conv(32→64, 4×4, s2) → ReLU  # 20 → 9
    Conv(64→64, 3×3, s1) → ReLU  # 9 → 7
    SpatialSoftmax(64, 7, 7)      # → 128
    """

    FEATURE_DIM = 128  # 64 channels × 2 (x + y expected positions)

    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )
        self.spatial_softmax = SpatialSoftmax(64, 7, 7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_softmax(self.conv(x))  # (B, 128)


class DetectionHead(nn.Module):
    """Auxiliary head: 128-d CNN features → (u, v) normalized in [-1, 1]."""

    def __init__(self, in_dim: int = CNNBackbone.FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)  # (B, 2)


# ── SAC Actor ───────────────────────────────────────────────────────────────


class SACActor(nn.Module):
    """Phase 1: one center camera + proprio → (dx, dy).
    Phase 2: flat 86D proprio only → (dx_res, dy_res, dz_res).
    """

    def __init__(self, phase: int, hidden_dim: int = 256):
        super().__init__()
        assert phase in (1, 2)
        self.phase = phase

        if phase == 1:
            self.n_cameras = 1
            action_dim = P1_ACTION_DIM
            self.encoders = nn.ModuleList([CNNBackbone()])
            self.detection_head = DetectionHead()
            mlp_in = P1_OBS_DIM  # 128 CNN features + 7 proprio = 135
        else:
            self.n_cameras = 0
            action_dim = P2_ACTION_DIM
            self.encoders = nn.ModuleList()
            self.detection_head = None
            mlp_in = P2_OBS_DIM  # 86 flat proprio, no cameras

        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def _encode_p1(
        self, obs: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.encoders[0](obs["image"])  # (B, 128)
        det = self.detection_head(feat)
        return feat, det

    def forward(self, obs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob, det_uv). det_uv is zeros for phase 2."""
        if self.phase == 1:
            feat, det = self._encode_p1(obs)
            x = torch.cat([feat, obs["proprio"]], dim=1)
        else:
            x = obs["proprio"]
            det = torch.zeros(x.shape[0], 2, device=x.device)

        h = self.mlp(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True), det

    @torch.no_grad()
    def get_action(self, obs: dict) -> torch.Tensor:
        if self.phase == 1:
            feat, _ = self._encode_p1(obs)
            x = torch.cat([feat, obs["proprio"]], dim=1)
        else:
            x = obs["proprio"]
        h = self.mlp(x)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return torch.tanh(Normal(mean, log_std.exp()).rsample())


# ── SAC Critic ──────────────────────────────────────────────────────────────


class _QNet(nn.Module):
    """Single Q-network. Phase 1 uses CNNs; Phase 2 is a flat MLP."""

    def __init__(self, phase: int, hidden_dim: int = 256):
        super().__init__()
        self.phase = phase
        action_dim = P1_ACTION_DIM if phase == 1 else P2_ACTION_DIM

        if phase == 1:
            self.n_cameras = 1
            self.encoders = nn.ModuleList([CNNBackbone()])
            state_dim = P1_OBS_DIM
        else:
            self.n_cameras = 0
            self.encoders = nn.ModuleList()
            state_dim = P2_OBS_DIM

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: dict, action: torch.Tensor) -> torch.Tensor:
        if self.phase == 1:
            feat = self.encoders[0](obs["image"])
            state = torch.cat([feat, obs["proprio"]], dim=1)
        else:
            state = obs["proprio"]
        return self.net(torch.cat([state, action], dim=1))


class SACCritic(nn.Module):
    """Twin Q-networks."""

    def __init__(self, phase: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = _QNet(phase, hidden_dim)
        self.q2 = _QNet(phase, hidden_dim)

    def forward(
        self, obs: dict, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)
