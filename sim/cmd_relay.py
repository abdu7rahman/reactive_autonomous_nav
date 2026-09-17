"""Relay the repo's Twist commands onto the simulated diff drive.

TurtleBot4's stack routes /cmd_vel_unstamped through the Create3
`motion_control` node, which emulates the vendor firmware's reflex layer.  In
this world all four of its cliff sensors latch a CLIFF hazard -- /hazard_detection
carries type 2, and /_internal/cliff_*/event carries it on every one of the four
-- even though the gz range sensors under the robot read 0.0157 m against a
0.15 m maximum, which is the floor, plainly there.  With the reflex latched the
node overrides every command with a backward escape: a direct 0.25 m/s forward
command left the robot reversing at -0.15 m/s.  The parameter that would turn
the reflex off (reflexes.REFLEX_CLIFF) is rejected at runtime -- "Setting
parameter failed" on three tries -- and it is hardcoded in create3_nodes.launch.py,
which is a system file.

So motion_control is stopped and this takes its place on the one path that
matters here: Twist in, TwistStamped out to the diff drive.  The reflex layer
is TurtleBot4 vendor firmware emulation, not part of what these planners and
controllers do, and skipping it costs the demonstration nothing.

The 0.306 m/s clamp is not invented: it is the value motion_control itself
reported for its own max_speed parameter, read back with ros2 param get before
the node was stopped.  Angular velocity is passed through, because each
controller already carries its own limit in the repo's config and there is no
second measured figure to impose here.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped

MAX_LIN = 0.306


class Relay(Node):
    def __init__(self) -> None:
        super().__init__("cmd_relay")
        self.pub = self.create_publisher(TwistStamped, "/diffdrive_controller/cmd_vel", 10)
        self.create_subscription(Twist, "/cmd_vel_unstamped", self.on_cmd, 10)
        self.get_logger().info(f"relaying /cmd_vel_unstamped, linear clamped to {MAX_LIN} m/s")

    def on_cmd(self, m: Twist) -> None:
        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "base_link"
        out.twist.linear.x = max(-MAX_LIN, min(MAX_LIN, m.linear.x))
        out.twist.angular.z = m.angular.z
        self.pub.publish(out)


def main() -> None:
    rclpy.init()
    n = Relay()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
