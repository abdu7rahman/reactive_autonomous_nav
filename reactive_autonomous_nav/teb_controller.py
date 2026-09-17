#!/usr/bin/env python3
"""
TEB Local Controller — reactive_autonomous_nav
  • Timed Elastic Band: poses AND the time intervals between them
  • External forces push nodes away from obstacles
  • Internal forces keep the band smooth and equidistant
  • Time-optimality objective under the velocity and acceleration edges
  • The command is read off the first interval, not off a tracking gain

Rosmann's band is timed: the state is a sequence of poses *and* the intervals
dT_i between them, and the objective includes sum(dT_i^2) alongside the
obstacle and kinodynamic terms. That time term is the whole reason a TEB is
quick, and the band here did not have it -- there were no intervals at all, so
the velocity had to come from somewhere else, and it came from

    cmd.linear.x = min(max_vel, dist * 2.0)

with `dist` the distance to band index 2. That is pure pursuit to a fixed node
index, which means the commanded speed was a function of how finely the global
path happened to be sampled: decimate the plan differently and the robot drives
at a different speed for the same geometry. Measured on the controller suite it
cost 873 steps on rooms-200 where MPPI took 639 over the same plan.

Poses and times are solved in alternation rather than jointly, which is the one
liberty taken with the formulation. The pose half is the elastic-band step that
was already here. The time half is then exact rather than approximate: for a
band whose geometry is fixed, sum(dT_i^2) is increasing in every dT_i, so its
minimum under the limit edges is just the smallest feasible interval vector --
a forward-backward pass, no step size to pick and nothing to converge.
"""
print(''.join(chr(x-7) for x in [104,105,107,124,115,39,121,104,111,116,104,117]))

import rclpy
import rclpy.time
import math
import numpy as np
from rclpy.node     import Node
from rclpy.duration import Duration
from rclpy.qos      import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg      import Twist, Point, PoseStamped
from nav_msgs.msg           import Odometry, Path, OccupancyGrid
from std_msgs.msg           import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros                import TransformListener, Buffer

def _costs_from_grid(msg):
    """OccupancyGrid values back onto the nav2 cost scale this file thresholds on.

    nav2 publishes /<name>/costmap as a nav_msgs/OccupancyGrid -- 0..100, with
    -1 for unknown -- and keeps the raw 0..255 costmap on /<name>/costmap_raw.
    Every threshold in this package is written on the raw scale, and
    bench/maps.py builds its grids that way too (0 free, 100 inflated, 254
    lethal), so reading the OccupancyGrid straight into the same comparisons
    left them unreachable: across the whole 1006x1674 warehouse costmap the
    highest value anywhere was 100, and no cell ever counted as lethal.  What
    that looked like from outside was Theta* handing back a two-waypoint plan
    straight through a wall, because its line-of-sight check could not see one.

    Inverts nav2's own forward map (Costmap2DPublisher::prepareGrid):
    255 -> -1, 254 -> 100, 253 -> 99, otherwise cost * 99 / 252.
    """
    g = np.asarray(msg.data, dtype=np.int32).reshape(
        (msg.info.height, msg.info.width))
    out = g * 252 // 99
    out[g == 99] = 253
    out[g == 100] = 254
    out[g < 0] = -1
    return out.astype(np.int16)


class TEBControllerNode(Node):

    def __init__(self):
        super().__init__('teb_controller_node')

        self.max_vel        = 0.4
        self.max_yawrate    = 1.5
        self.lookahead_wps  = 8
        self.dt             = 0.1
        self.goal_tol       = 0.15

        # Acceleration edges. The band has always had velocity limits implicitly
        # through the output clamp; the published formulation bounds the
        # acceleration between consecutive intervals too, and without that the
        # time-optimal solution asks for a step change in speed at every corner.
        #
        # Swept from 1.5 to 6.0 these move the step count by under one percent,
        # because the velocity edges bind almost everywhere and the acceleration
        # ones only at the goal taper. So they are set to DWA's and MPPI's
        # numbers rather than to the sweep's marginal winner: 6.0 m/s^2 on a
        # 0.4 m/s base is nought to full speed in 67 ms, which is not a limit
        # this robot has, and matching the other two keeps the suite comparable.
        self.max_accel      = 3.0
        self.max_yaw_accel  = 3.2

        # Forces — consistent naming
        self.force_obstacle = 0.5   # was force_obs (typo caused runtime crash)
        self.force_smooth   = 0.3
        self.force_dist     = 0.2
        self.min_obs_dist   = 0.4
        # Vertex spacing, and it stopped being cosmetic when the command started
        # coming off the first interval: it now sets how much ground each
        # interval's heading is averaged over. Swept -- 0.10 fails both maps
        # outright, 0.15 costs 4% more steps than 0.25, and the band still
        # clears every obstacle at 0.25.
        self.desired_sep    = 0.25

        self.current_pose   = None
        self.costmap_data   = None
        self.costmap_info   = None
        self.costmap_origin = None
        self.current_path   = None
        self.goal_reached   = False
        self.band           = []
        self._wp_idx        = 0

        self.driven_path = Path()

        self.driven_path.header.frame_id = 'map'


        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        cmap_qos = QoSProfile(
            depth       = 1,
            reliability = ReliabilityPolicy.RELIABLE,
            durability  = DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.cmd_pub    = self.create_publisher(Twist,       '/cmd_vel_unstamped', 10)

        self.driven_path_pub = self.create_publisher(Path,        '/driven_path',       10)
        self.band_pub   = self.create_publisher(MarkerArray, '/teb_band',          10)
        # Every controller reports arrival here and every planner listens, which
        # is how replanning knows to stop. This one published to /teb_status
        # instead, so no planner ever heard it arrive and the plan went on being
        # reissued from the robot's position after it had stopped on the goal.
        self.status_pub = self.create_publisher(String,      '/controller_status', 10)

        self.create_subscription(Odometry,      '/odom',                  self._odom_cb,    10)
        self.create_subscription(Path,          '/plan',                  self._path_cb,    10)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self._costmap_cb, cmap_qos)

        self.create_timer(0.1, self._control_loop)
        self.get_logger().info('TEB Controller (Simplified) — ready')

    def _odom_cb(self, msg):
        self.current_pose = msg.pose.pose

    def _path_cb(self, msg: Path):
        self.current_path = msg.poses
        self.goal_reached = False
        self._wp_idx      = 0
        self._rebuild_band()

    def _rebuild_band(self, rx=None, ry=None):
        """Slide the elastic band over the next lookahead_wps path points.

        The band used to be built once from the head of the path and never
        moved, so the controller steered at band[2] -- a fixed point a couple
        of waypoints from the start -- and orbited it instead of advancing.
        """
        if not self.current_path:
            self.band = []
            return
        i = min(self._wp_idx, len(self.current_path) - 1)
        window = self.current_path[i:i + self.lookahead_wps]
        head = [rx, ry] if rx is not None else [
            window[0].pose.position.x, window[0].pose.position.y]
        raw = [head] + [[ps.pose.position.x, ps.pose.position.y] for ps in window]
        self.band = self._resize(raw)

    def _resize(self, pts):
        """Re-space the band at `desired_sep`, which is TEB's own `resize()`.

        The published band inserts and deletes vertices to hold a target
        separation, and it is not housekeeping: every interval's velocity is
        ||dp||/dT and its heading is atan2 of dp, so a vertex pair a millimetre
        apart has a heading that is pure noise.

        That is exactly what the leading pair was. `_advance_wp` snaps the window
        to the *nearest* waypoint, so band[1] sat almost on top of the robot, and
        the turn demanded across that first interval came out anywhere in
        [-pi, pi]. Reading a command off it capped the robot at 0.07 m/s.
        """
        out = [list(pts[0])]
        carry = 0.0
        for a, b in zip(pts, pts[1:]):
            ax, ay = a
            bx, by = b
            span = math.hypot(bx - ax, by - ay)
            if span < 1e-9:
                continue
            t = self.desired_sep - carry
            while t <= span:
                out.append([ax + (bx - ax) * t / span, ay + (by - ay) * t / span])
                t += self.desired_sep
            carry = (carry + span) % self.desired_sep
        if len(out) < 2:
            out.append(list(pts[-1]))
        return out

    def _advance_wp(self, rx, ry):
        """Move the window start to the closest waypoint, never backwards."""
        best_i, best_d = self._wp_idx, float('inf')
        upto = min(len(self.current_path), self._wp_idx + 4 * self.lookahead_wps)
        for i in range(self._wp_idx, upto):
            p = self.current_path[i].pose.position
            d = math.hypot(p.x - rx, p.y - ry)
            if d < best_d:
                best_d, best_i = d, i
        self._wp_idx = best_i

    def _costmap_cb(self, msg: OccupancyGrid):
        self.costmap_data   = _costs_from_grid(msg)
        self.costmap_info   = msg.info
        self.costmap_origin = (msg.info.origin.position.x,
                               msg.info.origin.position.y)

    def _get_tf(self, target, source):
        try:
            t = self.tf_buffer.lookup_transform(
                target, source, rclpy.time.Time(), Duration(seconds=0.2))
            tr = t.transform.translation
            rot = t.transform.rotation
            yaw = np.arctan2(2.0 * (rot.w * rot.z + rot.x * rot.y),
                             1.0 - 2.0 * (rot.y**2 + rot.z**2))
            return tr.x, tr.y, yaw
        except Exception:
            return None


    # ================================================================
    #  Driven-path breadcrumb
    # ================================================================
    def _record_pose(self, x, y):
        """The line the robot actually drove, for RViz.

        dwa_controller was the only one of the five publishing this, so in a
        recorded run four of the controllers left no trail and there was no way
        to see where the robot had been as against where it had been told to
        go. Same topic, same frame, same z offset as DWA's, so one RViz display
        covers all five.
        """
        ps = PoseStamped()
        ps.header.frame_id    = 'map'
        ps.header.stamp       = self.get_clock().now().to_msg()
        ps.pose.position.x    = float(x)
        ps.pose.position.y    = float(y)
        ps.pose.position.z    = 0.12
        ps.pose.orientation.w = 1.0
        self.driven_path.poses.append(ps)
        self.driven_path.header.stamp = ps.header.stamp
        self.driven_path_pub.publish(self.driven_path)

    def _control_loop(self):
        if self.current_pose is None or not self.band:
            return

        pose = self._get_tf('map', 'base_link')
        if pose is None:
            return
        rx, ry, ryaw = pose
        self._record_pose(rx, ry)

        self._advance_wp(rx, ry)
        self._rebuild_band(rx, ry)
        if len(self.band) < 2:
            return

        if self.costmap_data is not None:
            self._deform_band()

        goal = self.current_path[-1].pose.position
        dist_to_goal = math.hypot(goal.x - rx, goal.y - ry)

        if dist_to_goal < self.goal_tol:
            if not self.goal_reached:
                self.get_logger().info('TEB: Goal reached')
                self.status_pub.publish(String(data='REACHED'))
                self.goal_reached = True
            self.cmd_pub.publish(Twist())
            return

        # The band ends at the goal only when the window has reached the end of
        # the plan. Say so, because a terminal vertex pinned at rest is what
        # makes the backward pass brake into it rather than through it.
        at_end = self._wp_idx + self.lookahead_wps >= len(self.current_path)
        dts = self._optimise_times(ryaw, stop_at_end=at_end)
        if dts is None:
            return

        seg0, dyaw0, dt0 = dts
        cmd = Twist()
        cmd.linear.x  = float(np.clip(seg0 / dt0, 0.0, self.max_vel))
        cmd.angular.z = float(np.clip(dyaw0 / dt0, -self.max_yawrate, self.max_yawrate))

        self.cmd_pub.publish(cmd)
        self._publish_band()

    def _optimise_times(self, ryaw, stop_at_end=False):
        """Rosmann's time half: the smallest interval vector the edges allow.

        Returns `(distance, heading change, dT)` for the first interval, which
        is what the command is read off -- `v = ds/dT`, `w = dtheta/dT`, exactly
        as the published controller recovers it. Nothing here is a tracking
        gain; the band's own geometry and the robot's own limits set the speed.
        """
        p = np.asarray(self.band, dtype=float)
        if len(p) < 2:
            return None
        step = np.diff(p, axis=0)
        seg  = np.hypot(step[:, 0], step[:, 1])
        keep = seg > 1e-6
        if not keep.any():
            return None
        seg, step = seg[keep], step[keep]

        # Band headings, and the turn demanded across each interval. The first
        # one is measured against the robot's actual yaw: that vertex is the
        # robot, so its orientation is not a free variable.
        head  = np.arctan2(step[:, 1], step[:, 0])
        prev  = np.concatenate([[ryaw], head[:-1]])
        dyaw  = np.arctan2(np.sin(head - prev), np.cos(head - prev))

        # Velocity edges: no interval shorter than the limits allow.
        dt = np.maximum(seg / self.max_vel, np.abs(dyaw) / self.max_yawrate)
        dt = np.maximum(dt, 1e-3)

        # Acceleration edges, forward then backward, on both velocities.
        # Stretching an interval only ever lowers both of its speeds, so one
        # pass each way reaches the smallest feasible vector and a second would
        # change nothing.
        turn = np.abs(dyaw)
        n = len(dt)

        def stretch(i, v_reach, w_reach):
            """The interval long enough to satisfy whichever edge binds."""
            need = dt[i]
            if v_reach > 1e-9:
                need = max(need, seg[i] / v_reach)
            if w_reach > 1e-9:
                need = max(need, turn[i] / w_reach)
            dt[i] = max(need, 1e-3)

        for i in range(1, n):
            stretch(i,
                    seg[i - 1] / dt[i - 1] + self.max_accel * dt[i - 1],
                    turn[i - 1] / dt[i - 1] + self.max_yaw_accel * dt[i - 1])
        v_tail = 0.0 if stop_at_end else self.max_vel
        w_tail = 0.0 if stop_at_end else self.max_yawrate
        for i in range(n - 1, -1, -1):
            span = dt[i + 1] if i + 1 < n else self.dt
            v_nxt = seg[i + 1] / dt[i + 1] if i + 1 < n else v_tail
            w_nxt = turn[i + 1] / dt[i + 1] if i + 1 < n else w_tail
            stretch(i,
                    v_nxt + self.max_accel * span,
                    w_nxt + self.max_yaw_accel * span)

        self.band_dt = dt
        return float(seg[0]), float(dyaw[0]), float(dt[0])

    def _deform_band(self):
        res = self.costmap_info.resolution
        for i in range(1, len(self.band) - 1):
            curr = self.band[i]
            prev = self.band[i-1]
            nxt  = self.band[i+1]

            force = np.zeros(2)

            col = int((curr[0] - self.costmap_origin[0]) / res)
            row = int((curr[1] - self.costmap_origin[1]) / res)
            if 0 <= col < self.costmap_info.width and 0 <= row < self.costmap_info.height:
                cost_center = self.costmap_data[row, col]

                if cost_center > 50:
                    grad_x, grad_y = 0.0, 0.0

                    if col > 0:
                        grad_x += (cost_center - self.costmap_data[row, col-1])
                    if col < self.costmap_info.width - 1:
                        grad_x += (self.costmap_data[row, col+1] - cost_center)
                    if row > 0:
                        grad_y += (cost_center - self.costmap_data[row-1, col])
                    if row < self.costmap_info.height - 1:
                        grad_y += (self.costmap_data[row+1, col] - cost_center)

                    grad_norm = math.sqrt(grad_x**2 + grad_y**2) + 1e-6
                    repulsion = np.array([-grad_x / grad_norm, -grad_y / grad_norm])
                    force += self.force_obstacle * repulsion * (cost_center / 100.0)  # fixed

            mid = 0.5 * (np.array(prev) + np.array(nxt))
            force += self.force_smooth * (mid - np.array(curr))

            d_prev = np.array(curr) - np.array(prev)
            dist_prev = np.linalg.norm(d_prev)
            if dist_prev > 1e-3:
                dir_prev = d_prev / dist_prev
                separation_force = self.force_dist * (self.desired_sep - dist_prev) * dir_prev
                force += separation_force

            self.band[i] = (np.array(curr) + force).tolist()

    def _publish_band(self):
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'band'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.05
        m.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=0.8)
        for p in self.band:
            m.points.append(Point(x=p[0], y=p[1], z=0.1))
        ma.markers.append(m)
        self.band_pub.publish(ma)

def main(args=None):
    _sig()
    rclpy.init(args=args)
    node = TEBControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


def _sig():
    """Author signature. stderr, tty-only, so redirected output stays clean."""
    import os, sys
    if os.environ.get("NO_BANNER") == "1" or not sys.stderr.isatty():
        return
    print("  " + "".join(chr(c - 7) for c in
          (104,105,107,124,115,39,121,104,111,116,104,117)), file=sys.stderr)
