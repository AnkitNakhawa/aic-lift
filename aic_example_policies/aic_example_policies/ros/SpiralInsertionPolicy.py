"""Two-phase policy with spiral-search Phase 2.

Phases 0 and 1 are identical to LocalPrecisionPolicy (orientation SLERP +
SAC-based XY centering). Phase 2 replaces the SAC insertion actor with a
deterministic compliant spiral search — no training required.

Phase 2 — Compliant Spiral Search:
  The robot descends slowly while executing an outward Archimedean spiral in
  XY. Low Z stiffness makes it compliant along the insertion axis. When the
  plug tip drifts over the port opening, the Z spring force drives it in.
  F/T lateral feedback corrects for residual misalignment throughout.

  Termination:
    - Success : depth > _INSERTION_DEPTH and f_mag < _SUCCESS_FORCE_LIMIT
    - Abort   : f_mag > _FORCE_LIMIT (jamming / wrong surface)
    - Timeout : _P2_MAX_STEPS steps exceeded
"""

import math
import os
import time

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
    NARROW_CROP_PX,
    WIDE_CROP_PX,
    add_canny_channel,
    center_crop_and_resize,
    ros_image_to_rgb,
    to_tensor,
)
from ..training.networks import SACActor

# ── Phase 1 checkpoint (reused from LocalPrecisionPolicy) ────────────────────
_DEFAULT_P1_CKPT = os.environ.get(
    "AIC_PHASE1_CKPT",
    os.path.join(os.path.dirname(__file__), "../../checkpoints/phase1/final.pt"),
)

# ── Nominal TCP orientations (w, x, y, z) ────────────────────────────────────
_NOMINAL_QUAT = {
    "sfp": np.array([0.6533, 0.2706, 0.6533, 0.2706], dtype=np.float64),
    "sc":  np.array([0.6533, 0.2706, 0.6533, 0.2706], dtype=np.float64),
}

# ── Phase 0 / Phase 1 tuning (unchanged from LocalPrecisionPolicy) ────────────
_P1_XY_THRESHOLD = 0.004   # m
_P1_MAX_STEPS    = 200
_ORIENT_STEPS    = 60
_STEP_DT         = 0.05    # s (20 Hz)
_XY_SCALE        = 0.005   # m per normalized action unit

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Phase 2 — spiral search parameters ───────────────────────────────────────
# Descent: robot moves down _DESCENT_DZ metres every step.
_DESCENT_DZ          = 0.0003   # m/step  (0.3 mm — slow enough to feel the hole)

# Spiral: Archimedean spiral, radius grows by _SPIRAL_DR every full revolution.
_SPIRAL_OMEGA        = 0.15     # rad/step
_SPIRAL_DR_PER_REV   = 0.0006   # m/rev   (0.6 mm outward per loop)
_SPIRAL_MAX_RADIUS   = 0.004    # m        (4 mm covers Phase 1 residual error)

# F/T reactive lateral correction: move away from measured lateral force.
_FT_LATERAL_GAIN     = 0.00015  # m/N

# Compliance: very low Z stiffness lets the arm drop into the port.
_LOW_Z_STIFFNESS  = [90.0, 90.0, 10.0, 30.0, 30.0, 30.0]
_LOW_Z_DAMPING    = [50.0, 50.0,  8.0, 15.0, 15.0, 15.0]

_INSERTION_DEPTH     = 0.015    # m  (15 mm — plug seated)
_SUCCESS_FORCE_LIMIT = 8.0      # N  (low force at depth → clean insertion)
_FORCE_LIMIT         = 25.0     # N  (abort threshold)
_P2_MAX_STEPS        = 400


class SpiralInsertionPolicy(Policy):
    """Spiral-search Phase 2 insertion; Phase 0 + 1 identical to LocalPrecisionPolicy."""

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._actor1: SACActor | None = None
        self._load_phase1_checkpoint()

    # ── Checkpoint loading ────────────────────────────────────────────────────

    def _load_phase1_checkpoint(self) -> None:
        path = os.path.abspath(_DEFAULT_P1_CKPT)
        if not os.path.exists(path):
            self.get_logger().warn(f"Phase 1 checkpoint not found: {path}")
            return
        actor = SACActor(1).to(_DEVICE)
        ckpt = torch.load(path, map_location=_DEVICE)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
        self._actor1 = actor
        self.get_logger().info(f"Loaded Phase 1 actor from {path}")

    # ── Shared observation helpers ────────────────────────────────────────────

    @staticmethod
    def _prep_image(img_msg, half_crop: int = WIDE_CROP_PX) -> torch.Tensor:
        rgb = ros_image_to_rgb(img_msg)
        rgbc = add_canny_channel(rgb)
        return to_tensor(center_crop_and_resize(rgbc, half_crop)).unsqueeze(0).to(_DEVICE)

    @staticmethod
    def _proprio_p1(tcp_rel, ft_z, port_type_enc, step_norm) -> torch.Tensor:
        arr = np.concatenate([tcp_rel, [ft_z], port_type_enc, [step_norm]]).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0).to(_DEVICE)

    @staticmethod
    def _get_tcp_pose(obs) -> Pose:
        return obs.controller_state.tcp_pose

    @staticmethod
    def _get_ft6(obs) -> np.ndarray:
        w = obs.wrist_wrench.wrench
        return np.array(
            [w.force.x, w.force.y, w.force.z,
             w.torque.x, w.torque.y, w.torque.z],
            dtype=np.float32,
        )

    @staticmethod
    def _pose_from_xyz_quat(x, y, z, quat: Pose.orientation) -> Pose:
        return Pose(
            position=Point(x=x, y=y, z=z),
            orientation=quat,
        )

    # ── Phase 0: orientation alignment ───────────────────────────────────────

    def _orient_step(self, get_observation, move_robot, port_type: str) -> Pose:
        target_q = _NOMINAL_QUAT.get(port_type, _NOMINAL_QUAT["sfp"])
        obs = get_observation()
        tcp = self._get_tcp_pose(obs)
        q_now = np.array(
            [tcp.orientation.w, tcp.orientation.x,
             tcp.orientation.y, tcp.orientation.z],
            dtype=np.float64,
        )
        self.get_logger().info(f"Phase 0: orienting for port_type={port_type}")
        for i in range(1, _ORIENT_STEPS + 1):
            q = quaternion_slerp(q_now, target_q, i / _ORIENT_STEPS)
            self.set_pose_target(
                move_robot,
                Pose(
                    position=tcp.position,
                    orientation=Quaternion(w=float(q[0]), x=float(q[1]),
                                          y=float(q[2]), z=float(q[3])),
                ),
            )
            self.sleep_for(_STEP_DT)
        return self._get_tcp_pose(get_observation())

    # ── Phase 1: XY centering (SAC) ───────────────────────────────────────────

    def _centering_phase(
        self, get_observation, move_robot, send_feedback, port_type_enc
    ) -> bool:
        if self._actor1 is None:
            self.get_logger().error("Phase 1 actor not loaded — cannot center.")
            return False

        self.get_logger().info("Phase 1: XY centering")
        obs  = get_observation()
        tcp0 = self._get_tcp_pose(obs)
        tcp0_xyz = np.array([tcp0.position.x, tcp0.position.y, tcp0.position.z])
        ft_tare  = self._get_ft6(obs).copy()

        for step in range(_P1_MAX_STEPS):
            obs     = get_observation()
            tcp     = self._get_tcp_pose(obs)
            tcp_xyz = np.array([tcp.position.x, tcp.position.y, tcp.position.z])
            xyz_rel = (tcp_xyz - tcp0_xyz).astype(np.float32)

            ft      = (self._get_ft6(obs) - ft_tare).astype(np.float32)
            prop_t  = self._proprio_p1(xyz_rel, float(ft[2]), port_type_enc,
                                       float(step / _P1_MAX_STEPS))
            obs_t   = {"image": self._prep_image(obs.center_image), "proprio": prop_t}

            with torch.no_grad():
                action = self._actor1.get_action(obs_t).squeeze(0).cpu().numpy()

            target = Pose(
                position=Point(
                    x=tcp.position.x + float(action[0]) * _XY_SCALE,
                    y=tcp.position.y + float(action[1]) * _XY_SCALE,
                    z=tcp.position.z,
                ),
                orientation=tcp.orientation,
            )
            self.set_pose_target(move_robot, target)

            tcp_err = obs.controller_state.tcp_error
            xy_err  = float(np.hypot(tcp_err[0], tcp_err[1]))
            send_feedback(f"P1 step={step} xy_err={xy_err:.4f}m")

            if xy_err < _P1_XY_THRESHOLD:
                self.get_logger().info(f"Phase 1 converged at step {step}")
                return True
            self.sleep_for(_STEP_DT)

        self.get_logger().warn("Phase 1 timed out — proceeding to insertion anyway")
        return True

    # ── Phase 2: compliant spiral search ─────────────────────────────────────

    def _spiral_insertion_phase(
        self, get_observation, move_robot, send_feedback
    ) -> bool:
        self.get_logger().info("Phase 2: compliant spiral search")

        obs   = get_observation()
        tcp0  = self._get_tcp_pose(obs)
        # Anchor: XY center from which the spiral radiates.
        anchor_x = tcp0.position.x
        anchor_y = tcp0.position.y
        anchor_z = tcp0.position.z
        quat     = tcp0.orientation
        ft_tare  = self._get_ft6(obs).copy()

        angle    = 0.0   # current spiral angle (rad)
        depth    = 0.0   # accumulated descent (m, positive = down)
        # Accumulated F/T lateral correction (world XY).
        ft_corr_x = 0.0
        ft_corr_y = 0.0

        for step in range(_P2_MAX_STEPS):
            obs  = get_observation()
            tcp  = self._get_tcp_pose(obs)
            ft6  = (self._get_ft6(obs) - ft_tare).astype(np.float32)
            f_mag = float(np.linalg.norm(ft6[:3]))

            # ── Safety abort ───────────────────────────────────────────────
            if f_mag > _FORCE_LIMIT:
                self.get_logger().warn(
                    f"Force limit exceeded ({f_mag:.1f} N) — aborting"
                )
                return False

            # ── Depth measurement ──────────────────────────────────────────
            depth = float(anchor_z - tcp.position.z)

            # ── Success check ──────────────────────────────────────────────
            if depth > _INSERTION_DEPTH and f_mag < _SUCCESS_FORCE_LIMIT:
                self.get_logger().info(
                    f"Insertion complete at step {step}: "
                    f"depth={depth:.4f}m f_mag={f_mag:.1f}N"
                )
                return True

            # ── Spiral XY offset ───────────────────────────────────────────
            # Archimedean spiral: r grows linearly with angle.
            radius = min(
                (_SPIRAL_DR_PER_REV / (2 * math.pi)) * angle,
                _SPIRAL_MAX_RADIUS,
            )
            spiral_x = radius * math.cos(angle)
            spiral_y = radius * math.sin(angle)
            angle += _SPIRAL_OMEGA

            # ── F/T lateral correction ─────────────────────────────────────
            # Accumulate a small correction opposing lateral force.
            # This moves the arm away from whatever surface it's pressing on.
            ft_corr_x -= _FT_LATERAL_GAIN * float(ft6[0])
            ft_corr_y -= _FT_LATERAL_GAIN * float(ft6[1])
            # Clamp correction to stay within Phase 1 residual band.
            ft_corr_x = float(np.clip(ft_corr_x, -_SPIRAL_MAX_RADIUS, _SPIRAL_MAX_RADIUS))
            ft_corr_y = float(np.clip(ft_corr_y, -_SPIRAL_MAX_RADIUS, _SPIRAL_MAX_RADIUS))

            # ── Commanded pose ─────────────────────────────────────────────
            cmd_x = anchor_x + spiral_x + ft_corr_x
            cmd_y = anchor_y + spiral_y + ft_corr_y
            cmd_z = anchor_z - depth - _DESCENT_DZ  # keep descending

            self.set_pose_target(
                move_robot,
                self._pose_from_xyz_quat(cmd_x, cmd_y, cmd_z, quat),
                stiffness=_LOW_Z_STIFFNESS,
                damping=_LOW_Z_DAMPING,
            )

            send_feedback(
                f"P2 step={step} depth={depth:.4f}m "
                f"r={radius*1000:.1f}mm f_mag={f_mag:.1f}N"
            )
            self.sleep_for(_STEP_DT)

        self.get_logger().warn("Phase 2 timed out")
        return False

    # ── Main entry point ──────────────────────────────────────────────────────

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(
            f"SpiralInsertionPolicy.insert_cable(): "
            f"plug={task.plug_type} port={task.port_type}"
        )

        port_type = "sc" if task.port_type.lower().startswith("sc") else "sfp"
        port_type_enc = np.array(
            [1.0, 0.0] if port_type == "sfp" else [0.0, 1.0], dtype=np.float32
        )

        self.get_logger().info("Waiting for first observation...")
        for _ in range(100):
            if get_observation() is not None:
                break
            time.sleep(0.1)
        else:
            self.get_logger().error("Timed out waiting for observations — aborting")
            return False

        # Phase 0: orient gripper
        send_feedback("Phase 0: orienting")
        self._orient_step(get_observation, move_robot, port_type)

        # Phase 1: SAC XY centering
        send_feedback("Phase 1: centering")
        self._centering_phase(get_observation, move_robot, send_feedback, port_type_enc)

        # Phase 2: spiral search insertion (no checkpoint needed)
        send_feedback("Phase 2: spiral insertion")
        success = self._spiral_insertion_phase(get_observation, move_robot, send_feedback)

        if success:
            self.get_logger().info("Insertion complete. Waiting for stabilization.")
            self.sleep_for(3.0)
        else:
            self.get_logger().error("Insertion failed.")

        return success
