"""Two-phase SAC insertion policy for the AIC qualification challenge.

Phase 0 — Orientation pre-step:
    SLERP the gripper to a nominal downward orientation (no GT needed).

Phase 1 — XY centering (CNN + SAC):
    Outputs (dx, dy) using the center wrist camera.
    Terminates when estimated XY error falls below threshold or time limit.

Phase 1.5 — Scripted spiral search:
    Expanding spiral descends slowly to locate the socket opening.
    Exits early when Fz contact force exceeds threshold.

Phase 2 — F/T-guided insertion (SAC, no cameras):
    Flat 86-D proprio: F/T history (12×6) + xyz_rel + xyz_vel + port_type
    + step_norm + depth + prev_action + f_mag.
    Residual policy: outputs (dx_res, dy_res, dz_res) on top of BASE_VZ
    constant descent.  Automatic retreat on jam detection.

Checkpoints: set AIC_PHASE1_CKPT / AIC_PHASE2_CKPT env vars.
"""

import math
import os
from collections import deque

import numpy as np
import torch
from geometry_msgs.msg import Point, Pose, Quaternion

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task
from transforms3d._gohlketransforms import quaternion_slerp

from ..training.image_utils import (
    WIDE_CROP_PX,
    add_canny_channel,
    center_crop_and_resize,
    ros_image_to_rgb,
    to_tensor,
)
from ..training.networks import (
    FT_HISTORY_LEN,
    P2_OBS_DIM,
    BASE_VZ,
    XY_INS_SCALE,
    Z_INS_SCALE,
    SACActor,
)

# ── Checkpoint paths ──────────────────────────────────────────────────────────
_DEFAULT_P1_CKPT = os.environ.get(
    "AIC_PHASE1_CKPT",
    os.path.join(os.path.dirname(__file__), "../../checkpoints/phase1/final.pt"),
)
_DEFAULT_P2_CKPT = os.environ.get(
    "AIC_PHASE2_CKPT",
    os.path.join(os.path.dirname(__file__), "../../checkpoints/phase2/final.pt"),
)

# ── Nominal TCP orientations (w, x, y, z) per port type ──────────────────────
_NOMINAL_QUAT = {
    "sfp": np.array([0.6533, 0.2706, 0.6533, 0.2706], dtype=np.float64),
    "sc":  np.array([0.6533, 0.2706, 0.6533, 0.2706], dtype=np.float64),
}

# ── Phase 1 thresholds ────────────────────────────────────────────────────────
_P1_XY_THRESHOLD = 0.004    # m — convergence criterion
_P1_MAX_STEPS = 200
_ORIENT_STEPS = 60

# ── Spiral search parameters ──────────────────────────────────────────────────
_SPIRAL_MAX_STEPS = 40
_SPIRAL_MAX_R = 0.003        # m — maximum spiral radius
_SPIRAL_ANGLE_STEP = math.radians(30)
_SPIRAL_DZ = 0.0002          # m — slow descent per spiral step
_SPIRAL_CONTACT_FZ = 1.0     # N — Fz threshold to stop spiral and start SAC

# ── Phase 2 thresholds ────────────────────────────────────────────────────────
_P2_FORCE_LIMIT = 25.0       # N — abort insertion
_P2_MAX_STEPS = 300
_JAM_F_THRESHOLD = 8.0       # N — sustained lateral force → jam
_JAM_WINDOW = 5              # steps
_RETREAT_DIST = 0.003        # m — pull back on jam
_MAX_RETREATS = 3

_STEP_DT = 0.05              # seconds (20 Hz)
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LocalPrecisionPolicy(Policy):
    """Two-phase SAC policy with scripted spiral search and jam-retreat logic."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._actor1: SACActor | None = None
        self._actor2: SACActor | None = None
        self._load_checkpoints()

    # ── Checkpoint loading ────────────────────────────────────────────────────

    def _load_checkpoints(self) -> None:
        def _load(phase: int, path: str) -> SACActor | None:
            path = os.path.abspath(path)
            if not os.path.exists(path):
                self.get_logger().warn(f"Checkpoint not found: {path}")
                return None
            actor = SACActor(phase).to(_DEVICE)
            ckpt = torch.load(path, map_location=_DEVICE)
            actor.load_state_dict(ckpt["actor"])
            actor.eval()
            self.get_logger().info(f"Loaded Phase {phase} actor from {path}")
            return actor

        self._actor1 = _load(1, _DEFAULT_P1_CKPT)
        self._actor2 = _load(2, _DEFAULT_P2_CKPT)

    # ── Image helpers (Phase 1 only) ──────────────────────────────────────────

    @staticmethod
    def _prep_image(img_msg, half_crop: int = WIDE_CROP_PX) -> torch.Tensor:
        rgb = ros_image_to_rgb(img_msg)
        rgbc = add_canny_channel(rgb)
        cropped = center_crop_and_resize(rgbc, half_crop)
        return to_tensor(cropped).unsqueeze(0).to(_DEVICE)

    # ── F/T helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _get_ft6(obs) -> np.ndarray:
        w = obs.wrist_wrench.wrench
        return np.array(
            [w.force.x, w.force.y, w.force.z,
             w.torque.x, w.torque.y, w.torque.z],
            dtype=np.float32,
        )

    # ── Proprioception builders ───────────────────────────────────────────────

    @staticmethod
    def _proprio_p1(
        tcp_rel: np.ndarray,
        ft_z: float,
        port_type_enc: np.ndarray,
        step_norm: float,
    ) -> torch.Tensor:
        arr = np.concatenate([tcp_rel, [ft_z], port_type_enc, [step_norm]]).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0).to(_DEVICE)

    @staticmethod
    def _proprio_p2(
        ft_history: deque,
        xyz_rel: np.ndarray,
        xyz_vel: np.ndarray,
        port_type_enc: np.ndarray,
        step_norm: float,
        depth: float,
        prev_action: np.ndarray,
        f_mag: float,
    ) -> torch.Tensor:
        ft_flat = np.concatenate(list(ft_history)).astype(np.float32)
        arr = np.concatenate([
            ft_flat,          # 72
            xyz_rel,          # 3
            xyz_vel,          # 3
            port_type_enc,    # 2
            [step_norm],      # 1
            [depth],          # 1
            prev_action,      # 3
            [f_mag],          # 1
        ]).astype(np.float32)
        assert arr.shape[0] == P2_OBS_DIM
        return torch.from_numpy(arr).unsqueeze(0).to(_DEVICE)

    # ── Pose helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _pose_from_tcp(tcp_pose, dx=0.0, dy=0.0, dz=0.0) -> Pose:
        return Pose(
            position=Point(
                x=tcp_pose.position.x + dx,
                y=tcp_pose.position.y + dy,
                z=tcp_pose.position.z + dz,
            ),
            orientation=tcp_pose.orientation,
        )

    @staticmethod
    def _get_tcp_pose(obs) -> Pose:
        return obs.controller_state.tcp_pose

    @staticmethod
    def _tcp_xyz(tcp_pose) -> np.ndarray:
        p = tcp_pose.position
        return np.array([p.x, p.y, p.z], dtype=np.float32)

    # ── Phase 0: orientation alignment ───────────────────────────────────────

    def _orient_step(self, get_observation, move_robot, port_type: str) -> Pose:
        target_q = _NOMINAL_QUAT.get(port_type, _NOMINAL_QUAT["sfp"])
        obs = get_observation()
        tcp = self._get_tcp_pose(obs)
        q_now = np.array([tcp.orientation.w, tcp.orientation.x,
                          tcp.orientation.y, tcp.orientation.z], dtype=np.float64)

        self.get_logger().info(f"Phase 0: orienting for port_type={port_type}")
        for i in range(1, _ORIENT_STEPS + 1):
            q = quaternion_slerp(q_now, target_q, i / _ORIENT_STEPS)
            pose = Pose(
                position=tcp.position,
                orientation=Quaternion(w=float(q[0]), x=float(q[1]),
                                       y=float(q[2]), z=float(q[3])),
            )
            self.set_pose_target(move_robot, pose)
            self.sleep_for(_STEP_DT)

        return self._get_tcp_pose(get_observation())

    # ── Phase 1: XY centering ─────────────────────────────────────────────────

    def _centering_phase(self, get_observation, move_robot, send_feedback,
                         port_type_enc: np.ndarray) -> bool:
        if self._actor1 is None:
            self.get_logger().error("Phase 1 actor not loaded.")
            return False

        self.get_logger().info("Phase 1: XY centering")
        obs = get_observation()
        tcp0_xyz = self._tcp_xyz(self._get_tcp_pose(obs))
        ft_tare = self._get_ft6(obs).copy()

        for step in range(_P1_MAX_STEPS):
            obs = get_observation()
            tcp = self._get_tcp_pose(obs)
            xyz_rel = (self._tcp_xyz(tcp) - tcp0_xyz).astype(np.float32)
            ft = (self._get_ft6(obs) - ft_tare).astype(np.float32)
            step_norm = step / _P1_MAX_STEPS

            img_t = self._prep_image(obs.center_image)
            prop_t = self._proprio_p1(xyz_rel, float(ft[2]), port_type_enc, step_norm)
            with torch.no_grad():
                action = self._actor1.get_action({"image": img_t, "proprio": prop_t})
                action = action.squeeze(0).cpu().numpy()

            dx = float(action[0]) * 0.005
            dy = float(action[1]) * 0.005
            self.set_pose_target(move_robot, self._pose_from_tcp(tcp, dx=dx, dy=dy))

            tcp_err = obs.controller_state.tcp_error
            xy_err = float(np.hypot(tcp_err[0], tcp_err[1]))
            send_feedback(f"P1 step={step} xy_err={xy_err:.4f}m")
            if xy_err < _P1_XY_THRESHOLD:
                self.get_logger().info(f"Phase 1 converged at step {step}")
                return True
            self.sleep_for(_STEP_DT)

        self.get_logger().warn("Phase 1 timed out")
        return True

    # ── Phase 1.5: Scripted spiral search ────────────────────────────────────

    def _spiral_search(self, get_observation, move_robot, send_feedback,
                       ft_tare: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Expands a spiral until contact or max steps. Returns (tcp0_xyz, ft_tare)."""
        self.get_logger().info("Phase 1.5: spiral search")
        obs = get_observation()
        tcp0_xyz = self._tcp_xyz(self._get_tcp_pose(obs))
        prev_xy = np.zeros(2)

        for i in range(_SPIRAL_MAX_STEPS):
            t = (i + 1) / _SPIRAL_MAX_STEPS
            r = _SPIRAL_MAX_R * t
            angle = i * _SPIRAL_ANGLE_STEP
            cur_xy = np.array([r * math.cos(angle), r * math.sin(angle)])
            dxy = cur_xy - prev_xy
            prev_xy = cur_xy

            obs = get_observation()
            tcp = self._get_tcp_pose(obs)
            target = self._pose_from_tcp(tcp, dx=float(dxy[0]),
                                         dy=float(dxy[1]), dz=-_SPIRAL_DZ)
            self.set_pose_target(move_robot, target)
            self.sleep_for(_STEP_DT)

            ft = self._get_ft6(obs) - ft_tare
            fz = float(ft[2])
            send_feedback(f"spiral i={i} fz={fz:.2f}N")
            if fz > _SPIRAL_CONTACT_FZ:
                self.get_logger().info(f"Spiral: contact at step {i}, fz={fz:.2f}N")
                break

        # Re-tare at spiral endpoint so Phase 2 sees forces from this baseline
        obs = get_observation()
        new_ft_tare = self._get_ft6(obs).copy()
        new_tcp0_xyz = self._tcp_xyz(self._get_tcp_pose(obs))
        return new_tcp0_xyz, new_ft_tare

    # ── Phase 2: F/T-guided insertion ────────────────────────────────────────

    def _insertion_phase(self, get_observation, move_robot, send_feedback,
                         port_type_enc: np.ndarray) -> bool:
        if self._actor2 is None:
            self.get_logger().error("Phase 2 actor not loaded.")
            return False

        self.get_logger().info("Phase 2: F/T-guided insertion (no cameras)")

        obs = get_observation()
        ft_tare = self._get_ft6(obs).copy()

        # Spiral search to find socket opening
        tcp0_xyz, ft_tare = self._spiral_search(
            get_observation, move_robot, send_feedback, ft_tare
        )

        # Phase 2 SAC loop
        ft_history: deque = deque(
            [np.zeros(6, dtype=np.float32)] * FT_HISTORY_LEN, maxlen=FT_HISTORY_LEN
        )
        prev_action = np.zeros(3, dtype=np.float32)
        prev_tcp_xyz = tcp0_xyz.copy()
        jam_history: deque = deque(maxlen=_JAM_WINDOW)
        retreat_count = 0

        low_z_stiffness = [90.0, 90.0, 20.0, 30.0, 30.0, 30.0]
        low_z_damping = [50.0, 50.0, 15.0, 15.0, 15.0, 15.0]

        for step in range(_P2_MAX_STEPS):
            obs = get_observation()
            tcp = self._get_tcp_pose(obs)
            tcp_xyz = self._tcp_xyz(tcp)
            xyz_rel = (tcp_xyz - tcp0_xyz).astype(np.float32)
            xyz_vel = (tcp_xyz - prev_tcp_xyz).astype(np.float32)
            depth = float(tcp0_xyz[2] - tcp_xyz[2])

            ft = (self._get_ft6(obs) - ft_tare).astype(np.float32)
            ft_history.append(ft.copy())
            f_mag = float(np.linalg.norm(ft[:3]))
            f_xy = float(np.linalg.norm(ft[:2]))

            if f_mag > _P2_FORCE_LIMIT:
                self.get_logger().warn(f"Force limit {f_mag:.1f}N — aborting")
                return False

            # Jam detection + retreat
            jam_history.append(f_xy)
            if len(jam_history) == _JAM_WINDOW and np.mean(jam_history) > _JAM_F_THRESHOLD:
                if retreat_count >= _MAX_RETREATS:
                    self.get_logger().warn("Max retreats reached — aborting")
                    return False
                self.get_logger().warn(f"Jam detected (f_xy_mean={np.mean(jam_history):.1f}N) — retreating")
                retreat_target = self._pose_from_tcp(tcp, dz=_RETREAT_DIST)
                self.set_pose_target(move_robot, retreat_target,
                                     stiffness=low_z_stiffness, damping=low_z_damping)
                self.sleep_for(_STEP_DT * 3)
                retreat_count += 1
                jam_history.clear()
                obs = get_observation()
                ft_tare = self._get_ft6(obs).copy()
                ft_history = deque(
                    [np.zeros(6, dtype=np.float32)] * FT_HISTORY_LEN, maxlen=FT_HISTORY_LEN
                )
                prev_action = np.zeros(3, dtype=np.float32)
                continue

            step_norm = step / _P2_MAX_STEPS
            prop_t = self._proprio_p2(
                ft_history, xyz_rel, xyz_vel, port_type_enc,
                step_norm, depth, prev_action, f_mag,
            )
            with torch.no_grad():
                action = self._actor2.get_action({"proprio": prop_t})
                action = action.squeeze(0).cpu().numpy()

            dx = float(action[0]) * XY_INS_SCALE
            dy = float(action[1]) * XY_INS_SCALE
            dz = -(BASE_VZ + float(action[2]) * Z_INS_SCALE)

            target_pose = self._pose_from_tcp(tcp, dx=dx, dy=dy, dz=dz)
            self.set_pose_target(move_robot, target_pose,
                                 stiffness=low_z_stiffness, damping=low_z_damping)

            prev_tcp_xyz = tcp_xyz.copy()
            prev_action = action.copy()
            send_feedback(f"P2 step={step} depth={depth:.4f}m fmag={f_mag:.1f}N retreats={retreat_count}")
            self.sleep_for(_STEP_DT)

        self.get_logger().warn("Phase 2 timed out")
        return True

    # ── Main entry point ──────────────────────────────────────────────────────

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(
            f"LocalPrecisionPolicy.insert_cable(): "
            f"plug={task.plug_type} port={task.port_type} target={task.target_module_name}"
        )

        port_type = "sc" if task.port_type.lower().startswith("sc") else "sfp"
        port_type_enc = np.array(
            [1.0, 0.0] if port_type == "sfp" else [0.0, 1.0], dtype=np.float32
        )

        # Wait for first observation
        obs = get_observation()
        if obs is None:
            self.get_logger().error("No observation available.")
            return False

        self._orient_step(get_observation, move_robot, port_type)
        self._centering_phase(get_observation, move_robot, send_feedback, port_type_enc)
        return self._insertion_phase(get_observation, move_robot, send_feedback, port_type_enc)
