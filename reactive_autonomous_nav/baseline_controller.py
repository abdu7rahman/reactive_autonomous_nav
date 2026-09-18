"""Races somebody else's DWA on a real robot, in the live graph.

bench/dwa_compare.py times these implementations and bench/gif_compare.py
drives them across a drawn field. Neither puts them on a robot. This does:
one node per lane, the reference implementation's own planner functions
choosing the commands, a TurtleBot 4 in Gazebo carrying them out and its own
lidar and costmap telling it what is in the way. What the matplotlib figure
could only assert -- that these are the same seven controllers solving the
same problem -- the race shows, because every lane is the same robot on the
same course at the same instant.

Two things had to be decided to put a goal seeker on a tracking course, and
both are choices rather than facts, so they are here rather than buried:

Obstacles. The references keep an explicit obstacle list and measure every
rollout point against it; this package's DWA looks a costmap cell up and
calls it lethal at 253, which nav2 marks wherever the robot's centre would
be inside its 0.22 m inscribed radius of something. The same test, written
their way, is the occupied cells themselves at 0.22 m of radius -- so that
is what they are handed, the 254 cells of the lane's own 6 x 6 m local
costmap. Not the scan, which is not where the robot may not be.

Handing them the inscribed ring as well -- every cell at 253 or above, at
half a cell of radius -- is the same collision test and was tried first. It
is not the same clearance: two of the three score clearance as 1 / (distance
to the nearest obstacle point), unnormalised, and dilating every wall by
0.22 m inflates that term by the same amount everywhere. Measured, on the
chicane: PythonRobotics and kmilo7204 reached 0.23 m and 0.22 m of 5.80,
spun in place and stayed there, with 1,135 inscribed cells against 88
occupied ones in the costmap they were given. It is also 14 times the
obstacle list, and their cost is linear in it.

The carrot. The references drive at a goal; the chicane hands every lane one
reference path and asks how well it is tracked. A goal seeker given the far
end of a slalom drives into the first wall, which measures the course and not
the controller, so each is given the same lookahead point this package's own
DWA steers at -- eight waypoints ahead, walked back to the furthest one the
costmap says is visible. That is the fair reading of "same problem" on a
course whose whole purpose is tracking, and it is the one difference from
bench/gif_compare.py, where the baselines get the goal because there the
route is this repo's advantage and hiding it would flatter nobody.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import math
import os
import statistics
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from reactive_autonomous_nav.dwa_controller import (
    LETHAL_COST, _costs_from_grid)

# nav2's LETHAL_OBSTACLE. LETHAL_COST above is its INSCRIBED_INFLATED_OBSTACLE
# at 253, which is the cost this package's controllers refuse a rollout point
# on; 254 is the cell that is actually occupied.
OCCUPIED_COST = 254

# The plant's own limits and this package's own sampling, so the lane differs
# by its scoring function and nothing else. Same source as dwa_controller.py:
# irobot_create_control/config/control.yaml by way of sim/race_robot.py.
MAX_V, MIN_V, MAX_W = 0.50, 0.0, 2.0
ACC_V, ACC_W = 0.9, 7.725
VEL_RES, YAW_RES = 0.02, 0.04
DT, HORIZON = 0.1, 25
LOOKAHEAD_WPS, WP_TOL, GOAL_TOL = 8, 0.25, 0.15

# The costmap's own inscribed radius, out of config/race_costmap_params.yaml:
# a cell is 253 when the robot's centre there would be within 0.22 m of
# something occupied, so a reference that refuses a rollout point within
# 0.22 m of an occupied cell is applying this package's lethal test.
INSCRIBED_R = 0.22

# These implementations are O(rollouts x obstacles) in Python -- PythonRobotics
# vectorises the inner comparison, kmilo7204 does not -- and only the cells a
# rollout can reach can change a decision: 0.5 m/s for 2.5 s is 1.25 m, and
# turning makes that a disc rather than a line.
NEAR_R = 1.6


def _yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y ** 2 + q.z ** 2))


class BaselineController(Node):
    """One lane, one reference implementation, chosen by the `impl` parameter."""

    def __init__(self):
        super().__init__('baseline_controller_node')
        self.impl = self.declare_parameter('impl', 'pythonrobotics').value
        self.map_frame = self.declare_parameter('map_frame', 'map').value
        self.odom_frame = self.declare_parameter('odom_frame', 'odom').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value

        # bench/ is not installed with the package -- it is the harness, not
        # the robot -- so it is found from the source tree this file lives in.
        # With colcon --symlink-install that is the repository, which is where
        # the fetched baselines are already cached.
        root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        from bench.baselines_py import load_kmilo, load_reference

        if self.impl == 'pythonrobotics':
            self.mod = load_reference()
            self.cfg = self.mod.Config()
            c = self.cfg
            c.dt, c.predict_time = DT, HORIZON * DT
            c.max_speed, c.min_speed = MAX_V, MIN_V
            c.max_yaw_rate = MAX_W
            c.max_accel, c.max_delta_yaw_rate = ACC_V, ACC_W
            c.v_resolution, c.yaw_rate_resolution = VEL_RES, YAW_RES
            c.robot_radius = INSCRIBED_R
        elif self.impl == 'kmilo7204':
            self.mod = load_kmilo()
            self.km = self.mod.DWA()
            c = self.km.config_params
            c.dt, c.dw_time = DT, HORIZON * DT
            c.max_v, c.min_v = MAX_V, MIN_V
            c.max_w, c.min_w = MAX_W, -MAX_W
            c.max_a, c.max_d_w = ACC_V, ACC_W
            c.v_res, c.w_res = VEL_RES, YAW_RES
            c.chassis_radius = INSCRIBED_R
        else:
            raise SystemExit(
                f'impl={self.impl!r} is not one of pythonrobotics, kmilo7204')

        self.pose = None
        self.vel = (0.0, 0.0)
        self.costs = None
        self.origin = (0.0, 0.0)
        self.res = 0.05
        self.path = None
        self.wp_idx = 0
        self.reached = False
        self.ticks = []

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        cmap_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_unstamped', 10)
        self.status_pub = self.create_publisher(String, '/controller_status', 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(Path, '/plan', self._path_cb, 10)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap',
                                 self._costmap_cb, cmap_qos)
        self.create_timer(DT, self._loop)
        self.get_logger().info(f'baseline controller: {self.impl} — ready')

    # -- callbacks ---------------------------------------------------
    def _odom_cb(self, msg):
        self.vel = (msg.twist.twist.linear.x, msg.twist.twist.angular.z)

    def _path_cb(self, msg):
        if not msg.poses:
            return
        self.path = msg.poses
        self.wp_idx = 0
        self.reached = False

    def _costmap_cb(self, msg):
        self.costs = _costs_from_grid(msg)
        self.origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.res = msg.info.resolution

    # -- helpers -----------------------------------------------------
    def _tf(self, target, source):
        try:
            t = self.tf_buffer.lookup_transform(target, source,
                                                rclpy.time.Time(),
                                                Duration(seconds=0.2))
            return (t.transform.translation.x, t.transform.translation.y,
                    _yaw(t.transform.rotation))
        except Exception:
            return None

    def _cost_at(self, x, y):
        gx = int((x - self.origin[0]) / self.res)
        gy = int((y - self.origin[1]) / self.res)
        h, w = self.costs.shape
        if 0 <= gx < w and 0 <= gy < h:
            return self.costs[gy, gx]
        return -1

    def _visible(self, x0, y0, x1, y1):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / self.res))
        for i in range(n + 1):
            t = i / n
            if self._cost_at(x0 + (x1 - x0) * t,
                             y0 + (y1 - y0) * t) >= LETHAL_COST:
                return False
        return True

    def _obstacles(self, rx, ry):
        """The occupied cells within reach of one rollout, as their centres."""
        ys, xs = np.nonzero(self.costs >= OCCUPIED_COST)
        if xs.size == 0:
            # Their scoring divides by the distance to the nearest obstacle, so
            # an empty list is a division by zero in two of the three and an
            # empty min() in the third. One cell far enough away to change no
            # decision costs a branch and removes that whole class of crash.
            return np.array([[rx + 1e3, ry + 1e3]])
        px = self.origin[0] + (xs + 0.5) * self.res
        py = self.origin[1] + (ys + 0.5) * self.res
        near = (px - rx) ** 2 + (py - ry) ** 2 <= NEAR_R ** 2
        if not near.any():
            return np.array([[rx + 1e3, ry + 1e3]])
        return np.column_stack((px[near], py[near]))

    def _carrot(self, rx, ry):
        """The same lookahead point this package's own DWA steers at."""
        while self.wp_idx < len(self.path) - 1:
            cur = self.path[self.wp_idx].pose.position
            nxt = self.path[self.wp_idx + 1].pose.position
            d = math.hypot(cur.x - rx, cur.y - ry)
            if d < WP_TOL or math.hypot(nxt.x - rx, nxt.y - ry) < d:
                self.wp_idx += 1
            else:
                break
        tidx = min(self.wp_idx + LOOKAHEAD_WPS, len(self.path) - 1)
        while tidx > self.wp_idx:
            p = self.path[tidx].pose.position
            if self._visible(rx, ry, p.x, p.y):
                break
            tidx -= 1
        p = self.path[tidx].pose.position
        return p.x, p.y

    # -- the loop ----------------------------------------------------
    def _loop(self):
        if self.path is None or self.costs is None:
            return
        pose = self._tf(self.map_frame, self.base_frame)
        if pose is None:
            return
        rx, ry, ryaw = pose

        final = self.path[-1].pose.position
        if math.hypot(final.x - rx, final.y - ry) < GOAL_TOL:
            if not self.reached:
                self.get_logger().info('Goal REACHED')
                self.status_pub.publish(String(data='REACHED'))
                self.reached = True
            self.cmd_pub.publish(Twist())
            return

        gx, gy = self._carrot(rx, ry)
        ob = self._obstacles(rx, ry)
        x = np.array([rx, ry, ryaw, self.vel[0], self.vel[1]])
        goal = np.array([gx, gy])

        t0 = time.perf_counter()
        if self.impl == 'pythonrobotics':
            dw = self.mod.calc_dynamic_window(x, self.cfg)
            u, _traj = self.mod.calc_control_and_trajectory(
                x, dw, self.cfg, goal, ob)
        else:
            self.km.config_params.obstacles = ob
            out = self.km.calculate_ctrl_traj(x, goal)
            u = out[0] if isinstance(out, (tuple, list)) else out
        self.ticks.append((time.perf_counter() - t0) * 1000.0)

        # The same clamp bench/trace.h applies, and for the same measured
        # reason: PythonRobotics reuses max_delta_yaw_rate as its stuck
        # recovery command, so a stall asks the base for the whole 7.725 rad/s
        # at once. The plant would refuse it anyway; refusing it here keeps the
        # commanded and the executed velocity the same number.
        cmd = Twist()
        cmd.linear.x = float(min(max(u[0], -MAX_V), MAX_V))
        cmd.angular.z = float(min(max(u[1], -MAX_W), MAX_W))
        self.cmd_pub.publish(cmd)

        if len(self.ticks) % 50 == 0:
            self.status_pub.publish(String(data=(
                f'{self.impl} wp={self.wp_idx}/{len(self.path)} '
                f'obs={len(ob)} {statistics.median(self.ticks[-50:]):.1f} ms')))


def main(args=None):
    rclpy.init(args=args)
    node = BaselineController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
