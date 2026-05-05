"""MuJoCo Gymnasium environment for Phase 2: F/T-guided insertion.

No cameras. Observation is a flat 86-D proprioception vector:
  ft_history (12×6=72) + xyz_rel (3) + xyz_vel (3) + port_type (2)
  + step_norm (1) + depth (1) + prev_action (3) + f_mag (1)

Policy outputs (dx_res, dy_res, dz_res) residuals on top of BASE_VZ constant
descent. The env prepends a scripted spiral search on each reset to find the
socket opening before the SAC policy takes over.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import mujoco
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError(f"Training deps missing: {e}") from e

from .networks import (
    FT_HISTORY_LEN,
    P2_OBS_DIM,
    XY_INS_SCALE,
    Z_INS_SCALE,
    BASE_VZ,
)

_ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
_HOME_QPOS = np.array([-0.1597, -1.3542, -1.6648, -1.6933, 1.5710, 1.4110])
_SFP_TCP_TO_TIP = np.array([0.0, 0.015385, -0.04245])
_SC_TCP_TO_TIP = np.array([0.0, 0.015385, -0.04045])

_INSERTION_DEPTH = 0.015    # m — depth at which success is declared

# F/T limits and contact thresholds
_FORCE_LIMIT = 30.0          # N — hard termination
_JAM_F_THRESHOLD = 8.0       # N — lateral force that indicates jamming
_JAM_WINDOW = 5              # steps of sustained jam before retreat triggers
_CONTACT_F_MIN = 0.5         # N — Fz below this = no contact
_CONTACT_F_MAX = 10.0        # N — Fz above this = too much contact
_JAM_THRESHOLD = _JAM_F_THRESHOLD  # alias used in step()
_RETREAT_DIST = 0.003        # m — how far to pull back on jam
_MAX_RETREATS = 3            # abort episode after this many retreats


@dataclass
class InsertionConfig:
    scene_path: str
    port_type: str = "sfp"
    sfp_port_body: str = "sfp_port_0_link"
    sc_port_body: str = "sc_port_0_link"
    tcp_site: str = "gripper_tcp"
    ft_force_sensor: str = "AtiForceTorqueSensor_force"
    ft_torque_sensor: str = "AtiForceTorqueSensor_torque"
    max_steps: int = 300
    substeps: int = 5
    arm_joint_names: list = field(default_factory=lambda: list(_ARM_JOINT_NAMES))
    # Curriculum: start offset at 0, grows to xy_offset_max during training
    xy_offset_max: float = 0.005   # m — max XY misalignment added at reset
    xy_offset_curriculum: float = 0.0  # current curriculum fraction [0, 1]
    z_start_offset: float = 0.005  # m — start above port surface
    # Spiral search
    spiral_steps: int = 40         # scripted spiral steps before SAC takes over
    spiral_max_r: float = 0.003    # m — max spiral radius
    spiral_angle_step: float = math.radians(30)  # rad per spiral step
    spiral_dz: float = 0.0002      # m — slow descent per spiral step


class InsertionEnv(gym.Env):
    """Phase 2 insertion environment (no cameras, flat proprio observation)."""

    metadata = {"render_modes": []}

    def __init__(self, config: InsertionConfig):
        super().__init__()
        self.cfg = config

        self.model = mujoco.MjModel.from_xml_path(config.scene_path)
        self.data = mujoco.MjData(self.model)

        self._tcp_site_id = self.model.site(config.tcp_site).id
        self._port_body_name = (
            config.sfp_port_body if config.port_type == "sfp" else config.sc_port_body
        )
        try:
            self._port_body_id = self.model.body(self._port_body_name).id
        except Exception:
            raise ValueError(
                f"Port body '{self._port_body_name}' not found in {config.scene_path}."
            )

        self._ft_force_adr = self._find_sensor_adr(config.ft_force_sensor)
        self._ft_torque_adr = self._find_sensor_adr(config.ft_torque_sensor)
        self._arm_joint_ids = self._resolve_joint_ids(config.arm_joint_names)

        self._port_type_enc = np.array(
            [1.0, 0.0] if config.port_type == "sfp" else [0.0, 1.0], dtype=np.float32
        )
        self._tcp_to_tip = (
            _SFP_TCP_TO_TIP if config.port_type == "sfp" else _SC_TCP_TO_TIP
        )

        self.observation_space = spaces.Box(
            -np.inf, np.inf, (P2_OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)

        self._step = 0
        self._start_tcp_pos: Optional[np.ndarray] = None
        self._tare_ft: Optional[np.ndarray] = None
        self._ft_history: deque = deque(
            [np.zeros(6, dtype=np.float32)] * FT_HISTORY_LEN, maxlen=FT_HISTORY_LEN
        )
        self._prev_tcp_pos: Optional[np.ndarray] = None
        self._prev_action: np.ndarray = np.zeros(3, dtype=np.float32)
        self._prev_depth: float = 0.0
        self._jam_history: deque = deque(maxlen=_JAM_WINDOW)
        self._retreat_count: int = 0

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _find_sensor_adr(self, name: str) -> int:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise ValueError(f"Sensor '{name}' not found.")
        return self.model.sensor_adr[sid]

    def _resolve_joint_ids(self, names: list[str]) -> np.ndarray:
        ids = []
        for name in names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found.")
            ids.append(self.model.jnt_dofadr[jid])
        return np.array(ids, dtype=int)

    def _get_tcp_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._tcp_site_id].copy()

    def _get_tcp_mat(self) -> np.ndarray:
        return self.data.site_xmat[self._tcp_site_id].reshape(3, 3).copy()

    def _get_port_pos(self) -> np.ndarray:
        return self.data.xpos[self._port_body_id].copy()

    def _get_plug_tip_pos(self) -> np.ndarray:
        return self._get_tcp_pos() + self._get_tcp_mat() @ self._tcp_to_tip

    def _get_ft_raw(self) -> np.ndarray:
        force = self.data.sensordata[self._ft_force_adr : self._ft_force_adr + 3]
        torque = self.data.sensordata[self._ft_torque_adr : self._ft_torque_adr + 3]
        return np.concatenate([force, torque]).astype(np.float32)

    def _get_ft(self) -> np.ndarray:
        raw = self._get_ft_raw()
        if self._tare_ft is not None:
            raw = raw - self._tare_ft
        return raw

    def _apply_delta_ik(self, delta: np.ndarray) -> None:
        jacp = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, None, self._tcp_site_id)
        J = jacp[:, self._arm_joint_ids]
        lam = 0.05
        dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(3), delta)
        self.data.qpos[self._arm_joint_ids] += dq

    def _physics_step(self) -> None:
        for _ in range(self.cfg.substeps):
            mujoco.mj_step(self.model, self.data)

    def _get_obs(self) -> np.ndarray:
        tcp_pos = self._get_tcp_pos()
        depth = float(self._start_tcp_pos[2] - tcp_pos[2]) if self._start_tcp_pos is not None else 0.0
        xyz_rel = (tcp_pos - self._start_tcp_pos).astype(np.float32) if self._start_tcp_pos is not None else np.zeros(3, dtype=np.float32)
        xyz_vel = (tcp_pos - self._prev_tcp_pos).astype(np.float32) if self._prev_tcp_pos is not None else np.zeros(3, dtype=np.float32)

        ft = self._get_ft()
        self._ft_history.append(ft.copy())
        ft_flat = np.concatenate(list(self._ft_history)).astype(np.float32)
        f_mag = float(np.linalg.norm(ft[:3]))

        step_norm = np.float32(self._step / self.cfg.max_steps)
        proprio = np.concatenate([
            ft_flat,                  # 72
            xyz_rel,                  # 3
            xyz_vel,                  # 3
            self._port_type_enc,      # 2
            [step_norm],              # 1
            [depth],                  # 1
            self._prev_action,        # 3
            [f_mag],                  # 1
        ]).astype(np.float32)
        assert proprio.shape[0] == P2_OBS_DIM, f"proprio dim mismatch: {proprio.shape[0]} != {P2_OBS_DIM}"
        return proprio

    # ── Spiral search (scripted, before SAC takes over) ──────────────────────

    def _run_spiral_search(self) -> None:
        """Expanding spiral to locate socket opening. Stops early on contact."""
        cfg = self.cfg
        prev_xy = np.zeros(2)
        for i in range(cfg.spiral_steps):
            t = (i + 1) / cfg.spiral_steps
            r = cfg.spiral_max_r * t
            angle = i * cfg.spiral_angle_step
            cur_xy = np.array([r * math.cos(angle), r * math.sin(angle)])
            d_xy = cur_xy - prev_xy
            prev_xy = cur_xy
            self._apply_delta_ik(np.array([d_xy[0], d_xy[1], -cfg.spiral_dz]))
            self._physics_step()
            ft = self._get_ft()
            if float(ft[2]) > _CONTACT_F_MIN:
                break

        # Re-tare after spiral so SAC sees forces relative to spiral endpoint
        self._tare_ft = self._get_ft_raw().copy()
        self._start_tcp_pos = self._get_tcp_pos().copy()
        self._prev_tcp_pos = self._start_tcp_pos.copy()
        self._ft_history = deque(
            [np.zeros(6, dtype=np.float32)] * FT_HISTORY_LEN, maxlen=FT_HISTORY_LEN
        )

    # ── Jam detection + retreat ──────────────────────────────────────────────

    def _check_and_retreat(self) -> bool:
        """Returns True if a retreat was executed."""
        ft = self._get_ft()
        f_xy = float(np.linalg.norm(ft[:2]))
        self._jam_history.append(f_xy)
        if len(self._jam_history) < _JAM_WINDOW:
            return False
        if np.mean(self._jam_history) > _JAM_THRESHOLD and self._retreat_count < _MAX_RETREATS:
            self._retreat_count += 1
            # Pull back
            self._apply_delta_ik(np.array([0.0, 0.0, _RETREAT_DIST]))
            self._physics_step()
            self._jam_history.clear()
            return True
        return False

    # ── Gymnasium API ────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        for i, jid in enumerate(self._arm_joint_ids):
            self.data.qpos[jid] = _HOME_QPOS[i]
        mujoco.mj_forward(self.model, self.data)

        # Curriculum: scale XY offset by curriculum fraction
        port_pos = self._get_port_pos()
        rng = self.np_random
        max_off = self.cfg.xy_offset_max * self.cfg.xy_offset_curriculum
        dx = rng.uniform(-max_off, max_off) if max_off > 0 else 0.0
        dy = rng.uniform(-max_off, max_off) if max_off > 0 else 0.0
        target = np.array([port_pos[0] + dx, port_pos[1] + dy, port_pos[2] + self.cfg.z_start_offset])
        self._move_tcp_to(target)
        mujoco.mj_forward(self.model, self.data)

        self._start_tcp_pos = self._get_tcp_pos().copy()
        self._prev_tcp_pos = self._start_tcp_pos.copy()
        self._tare_ft = self._get_ft_raw().copy()
        self._ft_history = deque(
            [np.zeros(6, dtype=np.float32)] * FT_HISTORY_LEN, maxlen=FT_HISTORY_LEN
        )
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._prev_depth = 0.0
        self._jam_history = deque(maxlen=_JAM_WINDOW)
        self._retreat_count = 0
        self._step = 0

        self._run_spiral_search()

        return self._get_obs(), {}

    def _move_tcp_to(self, target: np.ndarray, max_iter: int = 80) -> None:
        for _ in range(max_iter):
            err = target - self._get_tcp_pos()
            if np.linalg.norm(err) < 1e-4:
                break
            self._apply_delta_ik(err * 0.8)
            mujoco.mj_forward(self.model, self.data)

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)

        # Residual on top of constant BASE_VZ descent
        delta = np.array([
            float(action[0]) * XY_INS_SCALE,
            float(action[1]) * XY_INS_SCALE,
            -(BASE_VZ + float(action[2]) * Z_INS_SCALE),  # always descend
        ])
        self._prev_tcp_pos = self._get_tcp_pos().copy()
        self._apply_delta_ik(delta)
        self._physics_step()

        self._check_and_retreat()
        self._prev_action = action.copy()

        plug_tip = self._get_plug_tip_pos()
        port_pos = self._get_port_pos()
        ft = self._get_ft()
        fz = float(ft[2])
        f_mag = float(np.linalg.norm(ft[:3]))
        f_xy = float(np.linalg.norm(ft[:2]))

        depth = float(port_pos[2] - plug_tip[2])
        xy_err = float(np.linalg.norm(plug_tip[:2] - port_pos[:2]))

        # ── Reward ──────────────────────────────────────────────────────────
        reward = 0.0

        # Progress: reward depth gain per step (not cumulative depth)
        depth_gain = depth - self._prev_depth
        reward += 10.0 * max(0.0, depth_gain)
        self._prev_depth = depth

        # Stay centered
        reward -= 1.0 * xy_err

        # Contact maintenance: light reward for maintaining Fz in sweet spot
        if _CONTACT_F_MIN < fz < _CONTACT_F_MAX:
            reward += 0.05

        # Jamming penalty
        if len(self._jam_history) == _JAM_WINDOW and np.mean(self._jam_history) > _JAM_THRESHOLD:
            reward -= 0.5

        # Success
        terminated = False
        if depth > _INSERTION_DEPTH and xy_err < 0.005:
            reward += 100.0
            terminated = True

        # Force limit exceeded
        if f_mag > _FORCE_LIMIT:
            reward -= 20.0
            terminated = True

        # Too many retreats → give up
        if self._retreat_count >= _MAX_RETREATS:
            terminated = True

        self._step += 1
        truncated = self._step >= self.cfg.max_steps

        obs = self._get_obs()
        info = {"xy_error": xy_err, "depth": depth, "ft_mag": f_mag, "fz": fz}
        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        pass

