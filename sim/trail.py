"""Who is racing, and the line each of them actually drove.

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

The name labels are here rather than in grid_tf.py for the same reason the
trails are: they say who is racing, which is a property of the field and not of
the course.  grid_tf.py is started by race_up.sh, which brings a grid up
without being told what will race on it, and the first clip of the versus field
came out labelled with the five names of the wrong one.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray

from reactive_autonomous_nav.race_course import (
    COLOUR, CONTROLLER, FIELD, LANES, START_Y, YAW)

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
        # Five coloured trails with no legend is five coloured trails: the
        # clip is 560 px wide and carries no caption, so the name has to be in
        # the scene.  Each label lives in its own robot's base frame, so it
        # follows the robot without anyone publishing its pose, and it is
        # frame-locked -- a marker that is not is transformed once, when it
        # arrives, and either freezes where the robot was or is dropped.
        latched = QoSProfile(depth=1)
        latched.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.name_pub = self.create_publisher(MarkerArray, '/race_labels', latched)

        self.create_timer(1.0 / RATE, self._tick)
        self.create_timer(1.0, self._labels)
        self._labels()
        self.get_logger().info(
            f'field {FIELD}: trails and labels for '
            f'{", ".join(CONTROLLER[ns] for ns in LANES)}')

    def _labels(self) -> None:
        ma = MarkerArray()
        for i, ns in enumerate(LANES):
            m = Marker()
            m.header.frame_id = f'{ns}/base_link'
            m.ns, m.id = 'labels', i
            m.type, m.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            m.frame_locked = True
            m.pose.position.z = 0.62      # clear of the robot's own tower
            m.pose.orientation.w = 1.0
            m.scale.z = 0.30              # cap height, legible at 560 px
            r, g, b = COLOUR[ns]
            m.color.r, m.color.g, m.color.b = r / 255.0, g / 255.0, b / 255.0
            m.color.a = 1.0
            m.text = CONTROLLER[ns]
            ma.markers.append(m)
        self.name_pub.publish(ma)

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
