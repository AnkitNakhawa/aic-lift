#!/usr/bin/env python3
"""Stream distance-to-goal from /aic_controller/controller_state."""
import math
import rclpy
from rclpy.node import Node
from aic_control_interfaces.msg import ControllerState


class Watcher(Node):
    def __init__(self):
        super().__init__("goal_distance_watcher")
        self.create_subscription(
            ControllerState,
            "/aic_controller/controller_state",
            self.cb,
            10,
        )
        self.get_logger().info("Watching /aic_controller/controller_state ...")

    def cb(self, msg: ControllerState):
        e = msg.tcp_error
        pos_dist = math.sqrt(e[0] ** 2 + e[1] ** 2 + e[2] ** 2) * 1000  # mm
        rot_dist = math.sqrt(e[3] ** 2 + e[4] ** 2 + e[5] ** 2)          # rad
        t = msg.header.stamp
        print(
            f"[{t.sec}.{t.nanosec // 1_000_000:03d}] "
            f"dist={pos_dist:7.2f}mm  rot={rot_dist:.4f}rad  "
            f"xyz=({e[0]*1000:+.1f}, {e[1]*1000:+.1f}, {e[2]*1000:+.1f})mm",
            flush=True,
        )


def main():
    rclpy.init()
    node = Watcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
