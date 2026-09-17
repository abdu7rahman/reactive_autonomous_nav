"""Publish map->odom from the simulator's ground truth.

The static transform this replaces was measured once, at a reset, and was
correct only while wheel odometry stayed honest.  It did not: with the robot
wedged against its charging dock the wheels turned freely, odometry integrated
better than ten metres of travel that never happened, and the robot's computed
map pose walked off into a part of the warehouse it had never visited.  Laser
agreement against the map fell from 95% to 2% while the robot sat still.

Ground truth closes that loop.  T_map_odom = T_map_base . T_odom_base^-1 is
what a localiser publishes; taking T_map_base from /sim_ground_truth_pose
rather than from a particle filter keeps localisation error out of a
measurement that is about planners and controllers.  The map frame and the
simulator's world frame are the same frame -- checked by standing the robot on
the world origin with an identity transform and finding 95% of laser returns on
mapped wall.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster


def yaw_of(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class Localize(Node):
    def __init__(self) -> None:
        super().__init__("localize")
        self.truth: tuple[float, float, float] | None = None
        self.odom: tuple[float, float, float] | None = None
        self.br = TransformBroadcaster(self)
        # ground truth is published best-effort; a reliable subscription
        # silently receives nothing from it
        self.create_subscription(Odometry, "/sim_ground_truth_pose", self.on_truth,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        # No timer.  A 20 Hz timer on sim time is about 2.6 Hz of wall clock at
        # this world's real-time factor, and a Python node starved by RViz
        # loading meshes on the same four cores publishes well below even that.
        # The global costmap's activation gives up on base_link->map after a
        # short window and nav2's lifecycle manager then aborts the whole
        # bringup permanently rather than retrying -- one clip was lost to that
        # twice.  Publishing from the odom callback ties the correction to the
        # diff drive's own 62 Hz and keeps the stamps aligned with it.

    def on_truth(self, m: Odometry) -> None:
        p = m.pose.pose.position
        self.truth = (p.x, p.y, yaw_of(m.pose.pose.orientation))

    def on_odom(self, m: Odometry) -> None:
        p = m.pose.pose.position
        self.odom = (p.x, p.y, yaw_of(m.pose.pose.orientation))
        self.tick()

    def tick(self) -> None:
        if self.truth is None or self.odom is None:
            return
        mx, my, myaw = self.truth
        ox, oy, oyaw = self.odom
        # T_map_odom = T_map_base . T_odom_base^-1
        yaw = myaw - oyaw
        c, s = math.cos(yaw), math.sin(yaw)
        x = mx - (ox * c - oy * s)
        y = my - (ox * s + oy * c)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "odom"
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.rotation.z = math.sin(yaw / 2)
        t.transform.rotation.w = math.cos(yaw / 2)
        self.br.sendTransform(t)


def main() -> None:
    rclpy.init()
    n = Localize()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
