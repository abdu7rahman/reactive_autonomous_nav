"""Draw the line every robot actually drove, from one node, for all five.

The controllers in this package publish their own /driven_path and the nav2
controller plugins publish nothing of the kind, so a race between the two
would have had trails for some lanes and not others -- and even within one
field, five controllers each drawing their own idea of where they had been is
five different measurements on one picture.  This is one: each robot's own
odometry, transformed into `map` by the pin its odom frame was created at, at
one rate, with one threshold.

No transform lookup.  A robot's odom origin is the pose it was spawned at --
(lane_x, START_Y) facing YAW -- so odom (x, y) is map
(lane_x - y, START_Y + x) for the quarter turn this course uses, which is the
same arithmetic sim/grid_tf.py publishes on /tf and needs no listener.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path

from reactive_autonomous_nav.race_course import LANES, START_Y, YAW

RATE = 5.0
# Metres between kept points.  At 0.03 a six-metre run is about 200 poses,
# which is a Path message RViz redraws without noticing; at the raw odom rate
# it would be 600 a lane and growing for as long as the race lasts.
STEP = 0.03


class Trails(Node):

    def __init__(self) -> None:
        super().__init__('trails')
        self.cos, self.sin = math.cos(YAW), math.sin(YAW)
        self.paths: dict[str, Path] = {}
        self.pubs = {}
        for ns in LANES:
            p = Path()
            p.header.frame_id = 'map'
            self.paths[ns] = p
            self.pubs[ns] = self.create_publisher(Path, f'/{ns}/trail', 10)
            self.create_subscription(
                Odometry, f'/{ns}/odom',
                lambda m, ns=ns: self._odom(ns, m), qos_profile_sensor_data)
        self.create_timer(1.0 / RATE, self._tick)
        self.get_logger().info(f'trails for {len(LANES)} robots at {RATE:.0f} Hz')

    def _odom(self, ns: str, msg: Odometry) -> None:
        ox, oy = msg.pose.pose.position.x, msg.pose.pose.position.y
        x = LANES[ns] + ox * self.cos - oy * self.sin
        y = START_Y + ox * self.sin + oy * self.cos
        poses = self.paths[ns].poses
        if poses:
            last = poses[-1].pose.position
            if math.hypot(x - last.x, y - last.y) < STEP:
                return
        p = PoseStamped()
        p.header.frame_id = 'map'
        p.pose.position.x, p.pose.position.y = x, y
        p.pose.orientation.w = 1.0
        poses.append(p)

    def _tick(self) -> None:
        now = self.get_clock().now().to_msg()
        for ns, path in self.paths.items():
            if not path.poses:
                continue
            path.header.stamp = now
            self.pubs[ns].publish(path)


def main() -> None:
    rclpy.init()
    n = Trails()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
