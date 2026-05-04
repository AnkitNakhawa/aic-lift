"""Gymnasium environment for Phase 2 F/T-guided insertion backed by a running Gazebo/ROS 2 simulation.

Unlike insertion_env.py (MuJoCo-based), this environment communicates with the eval
container via ROS 2 topics and services.  Training runs at real-time speed (1×) but
uses the same sensors and controller as official evaluation time.

Prerequisites — start the eval container *with ground truth* in a separate terminal:

    ./scripts/start_eval.sh ground_truth:=true start_aic_engine:=false gazebo_gui:=false

Then, from the repo root with the pixi environment active:

    pixi run python -m aic_example_policies.training.train_insertion_gz \\
        --port_type sfp \\
        --port_frame task_board/nic_card_mount_0/sfp_port_0_link \\
        --save_dir checkpoints/gazebo/phase2
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError("pip install gymnasium") from e

try:
    import rclpy
    from aic_control_interfaces.msg import (
        MotionUpdate,
        TargetMode,
        TrajectoryGenerationMode,
    )
    from aic_control_interfaces.srv import ChangeTargetMode
    from aic_model_interfaces.msg import Observation as RosObs
    from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
    from std_msgs.msg import Header
    from tf2_ros import Buffer, TransformListener
    from rclpy.time import Time as RosTime
except ImportError as e:
    raise ImportError(
        f"ROS 2 deps missing ({e}). "
        "Source the ROS 2 workspace or run via 'pixi run python'."
    ) from e

from .image_utils import (
    IMG_SIZE,
    NARROW_CROP_PX,
    WIDE_CROP_PX,
    add_canny_channel,
    center_crop_and_resize,
    ros_image_to_rgb,
)

_SFP_TCP_TO_TIP = np.array([0.0, 0.015385, -0.04245])
_SC_TCP_TO_TIP = np.array([0.0, 0.015385, -0.04045])
_NOMINAL_QUAT_WXYZ = np.array([0.6533, 0.2706, 0.6533, 0.2706])

_INSERTION_DEPTH = 0.015      # 15 mm — success threshold
_FORCE_PENALTY_THRESHOLD = 10.0
_FORCE_LIMIT = 30.0

_DEFAULT_STIFFNESS = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
_DEFAULT_DAMPING = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]
# Reduced Z stiffness for compliant insertion
_INSERTION_STIFFNESS = [90.0, 90.0, 20.0, 30.0, 30.0, 30.0]
_INSERTION_DAMPING = [50.0, 50.0, 15.0, 15.0, 15.0, 15.0]


@dataclass
class GazeboInsertionConfig:
    port_type: str = "sfp"
    port_frame: str = "task_board/nic_card_mount_0/sfp_port_0_link"
    base_frame: str = "base_link"
    center_cam_frame: str = ""
    max_steps: int = 300
    # Small residual XY offset at episode start (mimics end of Phase 1)
    xy_offset_range: tuple = (-0.003, 0.003)
    # Start just above port surface
    z_start_offset: float = 0.005
    obs_timeout_sec: float = 3.0
    reset_move_time_sec: float = 4.0
    step_dt: float = 0.05


class GazeboInsertionEnv(gym.Env):
    """Phase 2 insertion environment driven by the live Gazebo simulation via ROS 2."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, config: GazeboInsertionConfig):
        super().__init__()
        self.cfg = config

        self._tcp_to_tip = (
            _SFP_TCP_TO_TIP if config.port_type == "sfp" else _SC_TCP_TO_TIP
        )
        self._port_type_enc = np.array(
            [1.0, 0.0] if config.port_type == "sfp" else [0.0, 1.0], dtype=np.float32
        )

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node("gz_insertion_env")
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(
            self._tf_buffer, self._node, spin_thread=False
        )
        self._motion_pub = self._node.create_publisher(
            MotionUpdate, "/aic_controller/pose_commands", 2
        )
        self._target_mode_client = self._node.create_client(
            ChangeTargetMode, "/aic_controller/change_target_mode"
        )

        self._latest_obs: RosObs | None = None
        self._obs_lock = threading.Lock()
        self._obs_event = threading.Event()
        self._node.create_subscription(
            RosObs, "observations", self._obs_callback, 5
        )

        self._spin_thread = threading.Thread(
            target=rclpy.spin, args=(self._node,), daemon=True
        )
        self._spin_thread.start()

        if not self._target_mode_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError(
                "change_target_mode service not available. "
                "Is the eval container running? Run: ./scripts/start_eval.sh"
            )
        self._set_cartesian_mode()

        self._step_count: int = 0
        self._start_tcp_xyz: np.ndarray | None = None
        self._tare_ft: np.ndarray | None = None
        self._cached_port_pos: np.ndarray | None = None

        self.observation_space = spaces.Dict(
            {
                "center": spaces.Box(0, 255, (IMG_SIZE, IMG_SIZE, 4), dtype=np.uint8),
                "left": spaces.Box(0, 255, (IMG_SIZE, IMG_SIZE, 4), dtype=np.uint8),
                "right": spaces.Box(0, 255, (IMG_SIZE, IMG_SIZE, 4), dtype=np.uint8),
                "proprio": spaces.Box(-np.inf, np.inf, (13,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)

    # ── ROS helpers ────────────────────────────────────────────────────────────

    def _obs_callback(self, msg: RosObs) -> None:
        with self._obs_lock:
            self._latest_obs = msg
        self._obs_event.set()

    def _get_latest_obs(self) -> RosObs:
        self._obs_event.clear()
        if not self._obs_event.wait(timeout=self.cfg.obs_timeout_sec):
            raise TimeoutError(
                f"No observation in {self.cfg.obs_timeout_sec}s. "
                "Is aic_adapter running?"
            )
        with self._obs_lock:
            return self._latest_obs

    def _set_cartesian_mode(self) -> None:
        req = ChangeTargetMode.Request()
        req.target_mode.mode = TargetMode.MODE_CARTESIAN
        future = self._target_mode_client.call_async(req)
        deadline = time.monotonic() + 5.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.05)

    def _publish_pose(
        self,
        pose: Pose,
        stiffness: list = _DEFAULT_STIFFNESS,
        damping: list = _DEFAULT_DAMPING,
    ) -> None:
        msg = MotionUpdate(
            header=Header(
                frame_id="base_link",
                stamp=self._node.get_clock().now().to_msg(),
            ),
            pose=pose,
            target_stiffness=np.diag(stiffness).flatten(),
            target_damping=np.diag(damping).flatten(),
            feedforward_wrench_at_tip=Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION
            ),
        )
        self._motion_pub.publish(msg)

    # ── State helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _quat_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    def _tcp_xyz(self, obs: RosObs) -> np.ndarray:
        p = obs.controller_state.tcp_pose.position
        return np.array([p.x, p.y, p.z])

    def _plug_tip_xyz(self, obs: RosObs) -> np.ndarray:
        p = obs.controller_state.tcp_pose.position
        q = obs.controller_state.tcp_pose.orientation
        rot = self._quat_to_rot(q.w, q.x, q.y, q.z)
        return np.array([p.x, p.y, p.z]) + rot @ self._tcp_to_tip

    def _ft6(self, obs: RosObs) -> np.ndarray:
        w = obs.wrist_wrench.wrench
        return np.array(
            [w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z],
            dtype=np.float32,
        )

    def _port_pos(self) -> np.ndarray | None:
        try:
            tf = self._tf_buffer.lookup_transform(
                self.cfg.base_frame, self.cfg.port_frame, RosTime()
            )
            t = tf.transform.translation
            return np.array([t.x, t.y, t.z])
        except Exception:
            return None

    def _compute_gt_uv(self, obs: RosObs) -> np.ndarray | None:
        port_base = self._port_pos()
        if port_base is None:
            return None
        cam_frame = (
            self.cfg.center_cam_frame or obs.center_camera_info.header.frame_id
        )
        if not cam_frame:
            return None
        try:
            tf = self._tf_buffer.lookup_transform(
                cam_frame, self.cfg.base_frame, RosTime()
            )
        except Exception:
            return None

        ct = tf.transform.translation
        cr = tf.transform.rotation
        rot = self._quat_to_rot(cr.w, cr.x, cr.y, cr.z)
        p_cam = rot @ (port_base - np.array([ct.x, ct.y, ct.z]))
        if p_cam[2] <= 0:
            return None

        K = np.array(obs.center_camera_info.k).reshape(3, 3)
        u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
        v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
        h = obs.center_image.height or 240
        w_img = obs.center_image.width or 320
        return np.array(
            [2.0 * u / w_img - 1.0, 2.0 * v / h - 1.0], dtype=np.float32
        )

    def _render_camera(self, img_msg, half_crop: int = WIDE_CROP_PX) -> np.ndarray:
        rgb = ros_image_to_rgb(img_msg)
        return center_crop_and_resize(add_canny_channel(rgb), half_crop)

    def _build_gym_obs(self, ros_obs: RosObs, depth: float = 0.0) -> dict:
        half_crop = NARROW_CROP_PX if depth > 0.005 else WIDE_CROP_PX
        images = {
            "center": self._render_camera(ros_obs.center_image, half_crop),
            "left": self._render_camera(ros_obs.left_image, half_crop),
            "right": self._render_camera(ros_obs.right_image, half_crop),
        }
        tcp_xyz = self._tcp_xyz(ros_obs)
        xyz_rel = (tcp_xyz - self._start_tcp_xyz).astype(np.float32)
        ft6 = self._ft6(ros_obs)
        if self._tare_ft is not None:
            ft6 = ft6 - self._tare_ft
        step_norm = np.float32(self._step_count / self.cfg.max_steps)
        proprio = np.concatenate(
            [xyz_rel, ft6, self._port_type_enc, [step_norm], [depth]]
        ).astype(np.float32)
        return {**images, "proprio": proprio}

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random

        port_pos = self._port_pos()
        if port_pos is None:
            raise RuntimeError(
                f"TF frame '{self.cfg.port_frame}' not available. "
                "Start the eval container with ground_truth:=true and verify port_frame."
            )
        self._cached_port_pos = port_pos.copy()

        dx = rng.uniform(*self.cfg.xy_offset_range)
        dy = rng.uniform(*self.cfg.xy_offset_range)
        start_xyz = port_pos + np.array([dx, dy, self.cfg.z_start_offset])

        w, x, y, z = _NOMINAL_QUAT_WXYZ
        self._publish_pose(
            Pose(
                position=Point(
                    x=float(start_xyz[0]),
                    y=float(start_xyz[1]),
                    z=float(start_xyz[2]),
                ),
                orientation=Quaternion(w=float(w), x=float(x), y=float(y), z=float(z)),
            )
        )
        time.sleep(self.cfg.reset_move_time_sec)

        ros_obs = self._get_latest_obs()
        self._start_tcp_xyz = self._tcp_xyz(ros_obs).copy()
        self._tare_ft = self._ft6(ros_obs).copy()
        self._step_count = 0

        return self._build_gym_obs(ros_obs, depth=0.0), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        from .networks import XY_INS_SCALE, Z_INS_SCALE

        ros_obs = self._get_latest_obs()
        tcp_pose = ros_obs.controller_state.tcp_pose
        tcp_xyz = self._tcp_xyz(ros_obs)
        depth = float(self._start_tcp_xyz[2] - tcp_xyz[2])

        dx = float(action[0]) * XY_INS_SCALE
        dy = float(action[1]) * XY_INS_SCALE
        dz = -float(action[2]) * Z_INS_SCALE  # positive action → descend
        self._publish_pose(
            Pose(
                position=Point(
                    x=tcp_pose.position.x + dx,
                    y=tcp_pose.position.y + dy,
                    z=tcp_pose.position.z + dz,
                ),
                orientation=tcp_pose.orientation,
            ),
            stiffness=_INSERTION_STIFFNESS,
            damping=_INSERTION_DAMPING,
        )
        time.sleep(self.cfg.step_dt)

        ros_obs_next = self._get_latest_obs()
        port_pos = self._port_pos() or self._cached_port_pos
        plug_tip = self._plug_tip_xyz(ros_obs_next)
        ft6 = self._ft6(ros_obs_next)
        if self._tare_ft is not None:
            ft6 = ft6 - self._tare_ft
        f_mag = float(np.linalg.norm(ft6[:3]))
        depth_next = float(self._start_tcp_xyz[2] - self._tcp_xyz(ros_obs_next)[2])
        xy_err = float(np.linalg.norm(plug_tip[:2] - port_pos[:2]))

        reward = 5.0 * max(0.0, depth_next) - 2.0 * xy_err
        if f_mag > _FORCE_PENALTY_THRESHOLD:
            reward -= 0.5 * (f_mag - _FORCE_PENALTY_THRESHOLD)

        terminated = False
        if depth_next > _INSERTION_DEPTH and xy_err < 0.005:
            reward += 50.0
            terminated = True
        if f_mag > _FORCE_LIMIT:
            reward -= 20.0
            terminated = True

        self._step_count += 1
        truncated = self._step_count >= self.cfg.max_steps

        obs = self._build_gym_obs(ros_obs_next, depth=depth_next)
        info = {
            "xy_error": xy_err,
            "depth": depth_next,
            "ft_mag": f_mag,
            "fz": float(ft6[2]),
            "gt_uv": self._compute_gt_uv(ros_obs_next),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        with self._obs_lock:
            if self._latest_obs is None:
                return np.zeros((240, 320, 3), dtype=np.uint8)
            return ros_image_to_rgb(self._latest_obs.center_image)

    def close(self):
        self._node.destroy_node()
