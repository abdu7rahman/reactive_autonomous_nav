#!/usr/bin/env python3
"""
DWA Local Controller — reactive_autonomous_nav
  • Fully vectorised trajectory rollout (meshgrid + matrix ops)
  • Batch costmap lookups via numpy fancy-indexing
  • HSV-scored trajectory colouring in RViz (cold→hot)
  • Thick glowing best-trajectory line
"""
print(''.join(chr(x-7) for x in [104,105,107,124,115,39,121,104,111,116,104,117]))

import rclpy
import rclpy.time
import math
import time
import numpy as np
from rclpy.node     import Node
from rclpy.duration import Duration
from rclpy.qos      import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg      import Twist, Point, PoseStamped
from nav_msgs.msg           import Odometry, Path, OccupancyGrid
from std_msgs.msg           import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros                import TransformListener, Buffer
from collections            import deque

LETHAL_COST = 253
WARN_COST   = 80


# ── helper: HSV → RGB (vectorised, S=1 V=1) ────────────────────────
def _hsv_to_rgb(h):
    """h in [0,1] → (r,g,b) with full saturation & value."""
    h6 = h * 6.0
    c  = 1.0
    x  = 1.0 - abs(h6 % 2.0 - 1.0)
    sector = int(h6) % 6
    if   sector == 0: return (c, x, 0.0)
    elif sector == 1: return (x, c, 0.0)
    elif sector == 2: return (0.0, c, x)
    elif sector == 3: return (0.0, x, c)
    elif sector == 4: return (x, 0.0, c)
    else:             return (c, 0.0, x)


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


class DWAControllerNode(Node):

    # Frame names, so more than one of these can run in one graph.
    #
    # They were literals, which is fine for one robot and impossible for five:
    # every node looked up 'base_link' and every path went out stamped 'map',
    # so five of them in one ROS graph would all have been talking about the
    # same robot. They are parameters now, and these class-level defaults keep
    # the single-robot case exactly as it was -- and keep bench/rig.py working,
    # since it builds nodes with object.__new__ and never runs __init__.
    map_frame  = 'map'
    odom_frame = 'odom'
    base_frame = 'base_link'

    # Loop-rate bookkeeping, so the log can say how many ticks a simulated
    # second actually got.  The dynamic window is the measured yaw rate plus or
    # minus max_dyawrate * dt = 0.1 rad/s per tick, and dt is 0.1 s because the
    # timer is 10 Hz: at ten ticks a simulated second the yaw rate can swing
    # 1 rad/s per second, and at two it can swing 0.2.  A controller that can
    # only unwind a 1.4 rad/s turn over seven simulated seconds will have
    # driven a circle first, which is what a run that never reached its goal
    # looked like. Class-level, because bench/rig.py builds these nodes with
    # object.__new__ and never runs __init__.
    _tick_n = 0
    _tick_sim = 0.0
    _tick_hz = 0.0
    _tick_ms = 0.0

    def __init__(self):
        super().__init__('dwa_controller_node')
        self.map_frame  = self.declare_parameter('map_frame',  'map').value
        self.odom_frame = self.declare_parameter('odom_frame', 'odom').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value

        # ── DWA params ───────────────────────────────────────────────
        self.max_vel            = 0.50
        self.min_vel            = 0.0
        self.max_yawrate        = 2.0
        # The accelerations are the robot's own, out of
        # irobot_create_control/config/control.yaml -- the same numbers
        # sim/race_robot.py puts into Gazebo's DiffDrive. They were 0.4 and
        # 1.0, which is 2.2 and 7.7 times more conservative than the plant,
        # and the dynamic window is defined as the velocities reachable in the
        # next interval given the robot's accelerations (Fox, Burgard and
        # Thrun): at 1.0 rad/s^2 it was not that, it was a rate limiter. The
        # consequence is asymmetric and it is the expensive half -- a
        # controller that can only add 0.1 rad/s of turn per tick can also only
        # take 0.1 rad/s away, so once it is turning at 1.4 rad/s it needs
        # fourteen ticks to stop, and by then it has driven a circle. Runs that
        # never reached their goal looked exactly like that.
        #
        # Measured on bench/test_planners.py, steps and metres, with the
        # sampler in its final form (see _samples):
        #
        #                   maze-wide-153        rooms-200
        #   0.4, 1.0        325 / 12.82 m      469 / 21.13 m
        #   0.9, 7.725      307 / 12.18 m      461 / 20.90 m
        #
        # Fewer steps and a shorter path on both maps. Keeping the sampling
        # resolution: coarsening it to 0.03 and 0.15 to hold the rollout count
        # down measured worse than either setting, 396 and 586 steps.
        self.max_accel          = 0.9
        self.max_dyawrate       = 7.725
        self.vel_res            = 0.02
        self.yawrate_res        = 0.04
        # How many of the fan to draw.  The window is 13 times wider than it
        # was, so the fan is about 390 rollouts rather than 30, and every one
        # of them was a MarkerArray entry built from 25 Points in a Python
        # loop -- 9,750 Point constructions a tick, on a 100 ms budget, for a
        # picture that is a grey smear at 390 lines anyway.
        self.traj_draw          = 48
        self.predict_time       = 2.5
        self.dt                 = 0.1
        self.heading_cost_gain  = 5.0
        self.speed_cost_gain    = 0.5
        self.obstacle_cost_gain = 5.0
        self.lookahead_wps      = 8      # ~0.4 m on a plan at costmap resolution
        self.goal_tol           = 0.15
        self.wp_tol             = 0.25

        # ── state ────────────────────────────────────────────────────
        self.current_pose   = None
        self.current_vel    = {'v': 0.0, 'omega': 0.0}
        self.costmap_data   = None
        self.costmap_info   = None
        self.costmap_origin = None
        self.current_path   = None
        self.wp_idx         = 0
        self.goal_reached   = False

        self.position_history = deque(maxlen=50)
        self.stuck_threshold  = 0.03
        self.recovery_mode    = False
        self.recovery_timer   = 0
        self.recovery_dir     = 1.0
        self.best_mid         = 0

        self.driven_path = Path()
        self.driven_path.header.frame_id = self.map_frame

        # ── TF ───────────────────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        cmap_qos = QoSProfile(
            depth       = 1,
            reliability = ReliabilityPolicy.RELIABLE,
            durability  = DurabilityPolicy.TRANSIENT_LOCAL,
        )

        # ── pubs ─────────────────────────────────────────────────────
        self.cmd_pub         = self.create_publisher(Twist,       '/cmd_vel_unstamped', 10)
        self.traj_pub        = self.create_publisher(MarkerArray, '/dwa_trajectories',  10)
        self.best_traj_pub   = self.create_publisher(MarkerArray, '/dwa_best_traj',     10)
        self.goal_mrk        = self.create_publisher(Marker,      '/dwa_goal',          10)
        self.status_pub      = self.create_publisher(String,      '/controller_status',        10)
        self.driven_path_pub = self.create_publisher(Path,        '/driven_path',       10)
        self.replan_pub      = self.create_publisher(String,      '/replan_request',    10)

        # ── subs ─────────────────────────────────────────────────────
        self.create_subscription(Odometry,      '/odom',                  self._odom_cb,    10)
        self.create_subscription(Path,          '/plan',                  self._path_cb,    10)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self._costmap_cb, cmap_qos)

        # ── control loop ─────────────────────────────────────────────
        self.create_timer(0.1, self._control_loop)
        self.get_logger().info('DWA Controller (vectorised) — ready')

    # ================================================================
    #  Callbacks
    # ================================================================
    def _odom_cb(self, msg):
        self.current_pose         = msg.pose.pose
        self.current_vel['v']     = msg.twist.twist.linear.x
        self.current_vel['omega'] = msg.twist.twist.angular.z

    def _path_cb(self, msg: Path):
        if not msg.poses:
            return
        self.current_path  = msg.poses
        self.goal_reached  = False
        self.recovery_mode = False
        self.best_mid      = 0
        # The history is about progress on the current path.  Left in place
        # across a replan it carries the old path's waypoint indices, and the
        # new path's index starts at zero, so _is_stuck would read the reset as
        # a lack of progress and go into recovery on a plan one tick old.
        self.position_history.clear()

        map_pose = self._get_tf(self.map_frame, self.base_frame)
        if map_pose is not None:
            rx, ry, rtheta = map_pose
            best_idx  = 0
            min_dist  = float('inf')
            for i, ps in enumerate(self.current_path):
                wp = ps.pose.position
                d  = np.hypot(wp.x - rx, wp.y - ry)
                heading_diff = abs(np.arctan2(
                    np.sin(np.arctan2(wp.y - ry, wp.x - rx) - rtheta),
                    np.cos(np.arctan2(wp.y - ry, wp.x - rx) - rtheta)
                ))
                if d < min_dist and heading_diff < np.pi * 0.75:
                    min_dist = d
                    best_idx = i
            self.wp_idx = best_idx
        else:
            self.wp_idx = 0

        self.get_logger().info(
            f'New path: {len(self.current_path)} wps, start wp={self.wp_idx}')

    def _costmap_cb(self, msg: OccupancyGrid):
        self.costmap_data   = _costs_from_grid(msg)
        self.costmap_info   = msg.info
        self.costmap_origin = (msg.info.origin.position.x,
                               msg.info.origin.position.y)

    # ================================================================
    #  TF
    # ================================================================
    def _get_tf(self, target, source):
        try:
            t = self.tf_buffer.lookup_transform(
                target, source, rclpy.time.Time(), Duration(seconds=0.2))
            tr  = t.transform.translation
            rot = t.transform.rotation
            yaw = np.arctan2(2.0 * (rot.w * rot.z + rot.x * rot.y),
                             1.0 - 2.0 * (rot.y**2 + rot.z**2))
            return tr.x, tr.y, yaw
        except Exception:
            return None

    # ================================================================
    #  Dynamic window
    # ================================================================
    def _dynamic_window(self):
        v = self.current_vel['v']
        w = self.current_vel['omega']
        return [
            max(self.min_vel,       v - self.max_accel    * self.dt),
            min(self.max_vel,       v + self.max_accel    * self.dt),
            max(-self.max_yawrate,  w - self.max_dyawrate * self.dt),
            min( self.max_yawrate,  w + self.max_dyawrate * self.dt),
        ]

    @staticmethod
    def _samples(lo, hi, res):
        """Inclusive samples across [lo, hi], never empty, never past hi.

        np.arange(lo, hi + res, res) gets both ends wrong: it admits one
        sample beyond hi (so the command can exceed max_vel or the clearance
        cap), and it returns nothing at all when hi < lo, which happens every
        time an external cap is applied below the window's own floor.

        A lattice of multiples of res, plus the two bounds, rather than
        linspace across the span.  linspace with a rounded count gives a gap
        wider than res whenever the span is not a whole number of res -- a
        0.09 m span at 0.02 m rounds to four intervals and samples every
        0.0225 m -- which went unnoticed while the window was 0.08 m wide and
        the arithmetic came out even, and which four of the twenty-seven cases
        in bench/test_dwa_window.py reported once the window was widened to
        the robot's own acceleration. Using ceil for the count fixes the gap
        but re-spaces every sample whenever a clamp moves either bound, so the
        clearance cap re-quantises the whole window on each tick and the chosen
        speed jitters with it. Measured on rooms-200: the lattice is 461 steps
        and 20.90 m, ceil-linspace 491 and 22.23, and the round-linspace this
        replaces 456 and 20.90 -- so the lattice buys the twenty-seven sound
        windows for five steps, and ceil-linspace would have cost thirty-five
        and 1.33 m.

        On the lattice, interior gaps are exactly res and the two end gaps are
        smaller, so the declared resolution holds; and a clamp drops candidates
        instead of shifting them.
        """
        if hi <= lo:
            return np.array([lo])
        k0 = math.ceil(lo / res - 1e-9)
        k1 = math.floor(hi / res + 1e-9)
        s = np.arange(k0, k1 + 1) * res if k1 >= k0 else np.empty(0)
        if s.size == 0 or s[0] > lo + 1e-9:
            s = np.r_[lo, s]
        if s[-1] < hi - 1e-9:
            s = np.r_[s, hi]
        return s

    # ================================================================
    #  Vectorised costmap helpers
    # ================================================================
    def _costmap_value(self, wx, wy) -> int:
        """Single-point lookup (used only by escape & clearance)."""
        if self.costmap_info is None:
            return -1
        res = self.costmap_info.resolution
        col = int((wx - self.costmap_origin[0]) / res)
        row = int((wy - self.costmap_origin[1]) / res)
        if not (0 <= col < self.costmap_info.width and
                0 <= row < self.costmap_info.height):
            return -1
        return int(self.costmap_data[row, col])

    def _segment_clear(self, x0, y0, x1, y1) -> bool:
        """Is the straight line between two world points free of lethal cells?

        Sampled at half a cell. Bresenham would walk a thin line and skip cells
        the segment really passes through, which is the bug the global planners
        in this package were fixed for; at half the cell pitch no cell the
        segment crosses can be stepped over.
        """
        if self.costmap_info is None:
            return True
        step = self.costmap_info.resolution * 0.5
        dist = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(dist / step) + 1)
        for i in range(n + 1):
            t = i / n
            if self._costmap_value(x0 + (x1 - x0) * t,
                                   y0 + (y1 - y0) * t) >= LETHAL_COST:
                return False
        return True

    def _batch_costmap(self, x_all, y_all):
        """
        Vectorised costmap lookup.
        x_all, y_all : (N, T) arrays of world coords
        Returns       : (N, T) int16 array of costs, -1 for out-of-bounds
        """
        res = self.costmap_info.resolution
        cols = ((x_all - self.costmap_origin[0]) / res).astype(np.intp)
        rows = ((y_all - self.costmap_origin[1]) / res).astype(np.intp)
        valid = ((cols >= 0) & (cols < self.costmap_info.width) &
                 (rows >= 0) & (rows < self.costmap_info.height))
        # clamp for safe indexing, then mask invalid later
        cols_safe = np.clip(cols, 0, self.costmap_info.width  - 1)
        rows_safe = np.clip(rows, 0, self.costmap_info.height - 1)
        costs = self.costmap_data[rows_safe, cols_safe].astype(np.int16)
        costs[~valid] = -1
        return costs

    # ================================================================
    #  Vectorised trajectory scoring
    # ================================================================
    def _score_trajectories(self, ox, oy, otheta, vs, ws, gx_odom, gy_odom):
        """
        Rollout + score every (v,w) pair simultaneously.
        Returns (best_v, best_w, best_score, all_trajs, all_scores, lethal_mask)
        """
        V, W = np.meshgrid(vs, ws)
        v_flat = V.ravel()
        w_flat = W.ravel()
        N = len(v_flat)
        T = int(self.predict_time / self.dt)

        # ── rollout all trajectories in parallel ─────────────────────
        x     = np.empty((N, T))
        y     = np.empty((N, T))
        theta = np.empty((N, T))
        x[:, 0]     = ox
        y[:, 0]     = oy
        theta[:, 0] = otheta

        for t in range(1, T):
            theta[:, t] = theta[:, t-1] + w_flat * self.dt
            x[:, t]     = x[:, t-1] + v_flat * np.cos(theta[:, t-1]) * self.dt
            y[:, t]     = y[:, t-1] + v_flat * np.sin(theta[:, t-1]) * self.dt

        # ── batch costmap evaluation ─────────────────────────────────
        costs = self._batch_costmap(x, y)               # (N, T)
        lethal_hit = np.any(costs >= LETHAL_COST, axis=1)  # (N,)

        # penalty per trajectory
        warn_mask = (costs > WARN_COST) & (costs < LETHAL_COST) & (costs >= 0)
        pen = np.where(warn_mask,
                       (costs - WARN_COST).astype(np.float64) / (LETHAL_COST - WARN_COST),
                       0.0)
        obs_cost = np.minimum(10.0, pen.sum(axis=1) / T * 10.0)

        # ── heading cost ─────────────────────────────────────────────
        # Scored where the trajectory *arrives*, not where it ends up.
        #
        # Reading the bearing at the last rollout sample is the textbook
        # formulation and it has a pathology whenever the goal is nearer than
        # the rollout reaches. The horizon spans predict_time * max_vel =
        # 1.25 m; the lookahead waypoint sits about 0.4 m out. Any trajectory
        # fast enough to get there flies past it, the bearing from its endpoint
        # back to the goal inverts, and heading -> 0. The controller settles
        # where v * predict_time ~ the lookahead distance, so it holds
        # 0.10-0.14 m/s down a clear straight line with a 0.50 m/s limit and
        # nothing in front of it -- measured on an empty 4.25 x 1.50 m field.
        #
        # Truncating at the first sample inside wp_tol removes the penalty
        # without adding a gain to trade off: every trajectory that reaches the
        # waypoint now scores near 1.0, and the speed term breaks the tie in
        # favour of the one that gets there soonest.
        reach = np.hypot(x - gx_odom, y - gy_odom) <= self.wp_tol
        idx   = np.where(reach.any(axis=1), reach.argmax(axis=1), T - 1)
        rows  = np.arange(N)
        diff = np.arctan2(gy_odom - y[rows, idx],
                          gx_odom - x[rows, idx]) - theta[rows, idx]
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        heading = 1.0 - np.abs(diff) / np.pi

        # ── composite score ──────────────────────────────────────────
        scores = (self.heading_cost_gain  * heading +
                  self.speed_cost_gain    * v_flat / self.max_vel -
                  self.obstacle_cost_gain * obs_cost)
        scores[lethal_hit] = -np.inf

        best_idx = int(np.argmax(scores))
        return (v_flat[best_idx], w_flat[best_idx], scores[best_idx],
                x, y, scores, lethal_hit, N, T)

    # ================================================================
    #  Forward clearance
    # ================================================================
    def _forward_clearance(self, ox, oy, otheta) -> float:
        """Free distance straight ahead of the robot.

        Three rays fanned out at ±0.3 rad measured the range to whatever those
        rays happened to hit rather than the room in front of the robot: in a
        straight corridor, being 0.1 m off-centre put a side wall 0.6 m down
        the angled ray and throttled the robot to 0.18 m/s with metres of clear
        floor ahead of it.

        The centre line is the right query and the consistent one. The costmap
        carries the inscribed band, which is what makes a centre-point lethal
        check a footprint check -- it is how _score_trajectories reads the map
        two calls later, and widening the scan by the robot radius on top of it
        asks for twice the robot's width of clearance.
        """
        ca, sa = math.cos(otheta), math.sin(otheta)
        for d in np.arange(0.1, 2.0, 0.05):
            if self._costmap_value(ox + d * ca, oy + d * sa) >= LETHAL_COST:
                return float(d)
        return float('inf')

    # ================================================================
    #  Stuck detection / recovery
    # ================================================================
    def _is_stuck(self):
        """No displacement, or no progress along the path.

        The displacement half was the whole test, and it catches a robot
        pressed against a wall while missing the one that matters more: a
        robot driving a circle. Orbiting its own lookahead point at 0.18 m/s
        and 0.9 rad/s -- which is what two recorded runs did, for a hundred
        seconds each, with the planner replanning around them -- the robot
        sweeps 4.5 rad over this fifty-tick window and covers 0.31 m of chord,
        ten times the 3 cm threshold. The detector saw a robot travelling
        perfectly well. It was travelling in a circle.

        Waypoint progress is what reaching the goal actually depends on, so
        that is the second half: fifty ticks, five simulated seconds, without
        advancing a single 0.05 m waypoint while moving is stuck. A robot
        genuinely crawling past an obstacle at 0.05 m/s still covers five
        waypoints in that window.
        """
        if len(self.position_history) < 50:
            return False
        p0, p1 = self.position_history[0], self.position_history[-1]
        if np.hypot(p1[0] - p0[0], p1[1] - p0[1]) < self.stuck_threshold:
            return True
        return p1[2] <= p0[2]

    def _recovery(self, ox, oy, otheta):
        """Spin toward open space, then nudge forward only if there is any.

        This used to run entirely open loop: alternate the spin direction, then
        command +0.1 m/s for ten ticks with no costmap check. The one situation
        it exists to handle is an obstacle in front of the robot, so the
        unconditional push drove into the thing that caused the stall.
        """
        self.recovery_timer += 1
        t = Twist()

        if self.recovery_timer == 1:
            left  = self._forward_clearance(ox, oy, otheta + math.pi / 2.0)
            right = self._forward_clearance(ox, oy, otheta - math.pi / 2.0)
            self.recovery_dir = 1.0 if left >= right else -1.0

        if self.recovery_timer <= 20 or self._forward_clearance(ox, oy, otheta) < 0.35:
            t.angular.z = 0.8 * self.recovery_dir
        else:
            t.linear.x = 0.1

        if self.recovery_timer >= 30:
            self.recovery_mode  = False
            self.recovery_timer = 0
            self.position_history.clear()
            self.replan_pub.publish(String(data='replan'))
            self.get_logger().info('Recovery done — requesting replan')
        self.cmd_pub.publish(t)

    # ================================================================
    #  Driven-path breadcrumb
    # ================================================================
    def _record_pose(self, x, y):
        ps = PoseStamped()
        ps.header.frame_id    = self.map_frame
        ps.header.stamp       = self.get_clock().now().to_msg()
        ps.pose.position.x    = x
        ps.pose.position.y    = y
        ps.pose.position.z    = 0.12
        ps.pose.orientation.w = 1.0
        self.driven_path.poses.append(ps)
        self.driven_path.header.stamp = ps.header.stamp
        self.driven_path_pub.publish(self.driven_path)

    # ================================================================
    #  RViz marker helpers
    # ================================================================
    def _make_traj_marker(self, x_row, y_row, mid, r, g, b, a, w):
        m = Marker()
        m.header.frame_id = self.odom_frame
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns     = 'candidates'
        m.id     = mid
        m.type   = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = w
        m.color   = ColorRGBA(r=r, g=g, b=b, a=a)
        m.points  = [Point(x=float(x_row[t]), y=float(y_row[t]), z=0.15)
                     for t in range(len(x_row))]
        return m

    def _publish_scored_markers(self, x, y, scores, lethal_mask, N, T):
        """
        HSV-coloured trajectory fan:
          lethal  → thin red (0.35 alpha)
          scored  → hue mapped cold (blue) → hot (red) by normalised score
        """
        ma  = MarkerArray()
        cid = 0

        valid_mask = ~lethal_mask
        if valid_mask.any():
            s_min = scores[valid_mask].min()
            s_max = scores[valid_mask].max()
            s_range = max(s_max - s_min, 1e-6)

        # An evenly spaced sample of the fan, which is ordered by (v, omega),
        # so the sample spans it rather than clustering in one corner. DELETEALL
        # first because the fan's size follows the window and RViz keeps a
        # marker until something replaces or deletes it.
        clear = Marker()
        clear.action = Marker.DELETEALL
        ma.markers.append(clear)
        stride = max(1, N // self.traj_draw)

        for i in range(0, N, stride):
            if lethal_mask[i]:
                # thin red — lethal
                ma.markers.append(self._make_traj_marker(
                    x[i], y[i], cid, 1.0, 0.0, 0.0, 0.20, 0.008))
            else:
                # hue 0.65 (blue) → 0.0 (red) scaled by score
                t = (scores[i] - s_min) / s_range
                hue = 0.65 * (1.0 - t)          # blue→red
                r, g, b = _hsv_to_rgb(hue)
                alpha = 0.25 + 0.55 * t          # low-score = faint, high-score = bright
                width = 0.012 + 0.03 * t          # low-score = thin, high-score = thick
                ma.markers.append(self._make_traj_marker(
                    x[i], y[i], cid, r, g, b, alpha, width))
            cid += 1

        self.traj_pub.publish(ma)

    def _publish_best_traj(self, x_row, y_row):
        """Thick bright-green glow for the chosen trajectory."""
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = self.odom_frame
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns     = 'best'
        m.id     = self.best_mid
        m.type   = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.07
        m.color   = ColorRGBA(r=0.1, g=1.0, b=0.3, a=0.95)
        m.points  = [Point(x=float(x_row[t]), y=float(y_row[t]), z=0.16)
                     for t in range(len(x_row))]
        ma.markers.append(m)

        # outer glow
        m2 = Marker()
        m2.header = m.header
        m2.ns     = 'best_glow'
        m2.id     = self.best_mid
        m2.type   = Marker.LINE_STRIP
        m2.action = Marker.ADD
        m2.scale.x = 0.14
        m2.color   = ColorRGBA(r=0.1, g=1.0, b=0.3, a=0.25)
        m2.points  = list(m.points)
        ma.markers.append(m2)

        self.best_mid += 1
        self.best_traj_pub.publish(ma)

    def _publish_goal_marker(self, gx, gy):
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp    = self.get_clock().now().to_msg()
        m.ns     = 'dwa_goal'
        m.id     = 0
        m.type   = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x    = gx
        m.pose.position.y    = gy
        m.pose.position.z    = 0.15
        m.pose.orientation.w = 1.0
        m.scale.x = 0.35
        m.scale.y = 0.35
        m.scale.z = 0.35
        m.color = ColorRGBA(r=1.0, g=0.5, b=0.0, a=1.0)

        # outer glow ring
        self.goal_mrk.publish(m)

    # ================================================================
    #  Main control loop
    # ================================================================
    def _control_loop(self):
        t_tick = time.perf_counter()
        now = self.get_clock().now().nanoseconds * 1e-9
        self._tick_n += 1
        if self._tick_sim <= 0.0:
            self._tick_sim = now
        elif now - self._tick_sim >= 1.0:
            self._tick_hz = self._tick_n / (now - self._tick_sim)
            self._tick_n, self._tick_sim = 0, now

        if self.current_pose is None or self.current_path is None:
            return
        if self.costmap_data is None:
            self.get_logger().warn('Waiting for costmap…',
                                   throttle_duration_sec=3.0)
            return

        map_pose = self._get_tf(self.map_frame, self.base_frame)
        if map_pose is None:
            # A silent return leaves the robot with no new command and the
            # log with no reason: mppi_controller stopped dead at 3.44 m of a
            # 6 m race this way and its entire log for the run was three
            # lines. Stop and say so, throttled.
            self.cmd_pub.publish(Twist())
            self.get_logger().warn(
                f'No {self.map_frame} -> {self.base_frame} transform; holding',
                throttle_duration_sec=2.0)
            return
        mx, my, mtheta = map_pose

        odom_pose = self._get_tf(self.odom_frame, self.base_frame)
        ox, oy, otheta = odom_pose if odom_pose else (mx, my, mtheta)

        # The waypoint index rides along: _is_stuck needs to know whether the
        # robot has made progress on the path, not just whether it has moved.
        self.position_history.append((mx, my, self.wp_idx))
        self._record_pose(mx, my)

        # ── advance waypoint ─────────────────────────────────────────
        # Proximity alone strands the tracker: a waypoint the robot flew past
        # without ever coming within wp_tol is never retired, so it spends the
        # rest of the run steering at a point behind itself. Retiring one whose
        # successor is already nearer makes the advance monotonic in the robot's
        # progress along the plan rather than in how close it happened to pass.
        while self.wp_idx < len(self.current_path) - 1:
            cur = self.current_path[self.wp_idx].pose.position
            nxt = self.current_path[self.wp_idx + 1].pose.position
            d_cur = np.hypot(cur.x - mx, cur.y - my)
            if d_cur < self.wp_tol or np.hypot(nxt.x - mx, nxt.y - my) < d_cur:
                self.wp_idx += 1
            else:
                break

        # ── goal check ───────────────────────────────────────────────
        final = self.current_path[-1].pose.position
        if np.hypot(final.x - mx, final.y - my) < self.goal_tol:
            if not self.goal_reached:
                self.get_logger().info('Goal REACHED')
                self.status_pub.publish(String(data='REACHED'))
                self.goal_reached = True
            self.cmd_pub.publish(Twist())
            return

        # ── lookahead goal (map frame) ───────────────────────────────
        # A waypoint count rather than a distance, so it tracks the plan's
        # resolution -- fine here because every global planner in this package
        # emits at costmap resolution, which puts eight waypoints at ~0.4 m.
        # Swept on bench/test_dwa_window.py's maps: 3 tracks the maze tightly
        # (1.02x plan length) but oscillates across open rooms (1.32x); 12 is
        # the reverse (1.01x rooms, 1.13x maze and 74% slower); 16 never
        # finishes the maze. Eight holds both under 1.09x.
        tidx   = min(self.wp_idx + self.lookahead_wps,
                     len(self.current_path) - 1)
        # ...clamped to a waypoint the robot can actually see.
        #
        # A fixed count assumes the plan runs roughly away from the robot, and
        # where it wraps an obstacle it does not: on maze-wide-153 the route
        # rounds the tip of a pillar at x = 2.45, so from waypoint 73 the eighth
        # waypoint ahead sits on the *far side* of that wall. The controller
        # steered at it, drove into the near face, and stalled there for six
        # hundred steps -- 4.57 m from the goal with a clear route to it.
        #
        # Walking back to the furthest visible waypoint is what pure pursuit
        # does for the same reason. The near ones stay reachable, so a plan that
        # does not double back is unaffected and the swept lookahead of eight
        # still applies wherever it is honest.
        while tidx > self.wp_idx:
            tp = self.current_path[tidx].pose.position
            if self._segment_clear(mx, my, tp.x, tp.y):
                break
            tidx -= 1
        gx_map = self.current_path[tidx].pose.position.x
        gy_map = self.current_path[tidx].pose.position.y
        self._publish_goal_marker(gx_map, gy_map)

        # ── convert goal map→odom for DWA rollout ────────────────────
        odom_from_map = self._get_tf(self.odom_frame, self.map_frame)
        if odom_from_map is not None:
            tmx, tmy, tyaw = odom_from_map
            gx_odom = math.cos(tyaw) * gx_map - math.sin(tyaw) * gy_map + tmx
            gy_odom = math.sin(tyaw) * gx_map + math.cos(tyaw) * gy_map + tmy
        else:
            gx_odom, gy_odom = gx_map, gy_map

        # ── stuck? ───────────────────────────────────────────────────
        if not self.recovery_mode and self._is_stuck():
            self.get_logger().warn('Robot stuck — recovery')
            self.recovery_mode  = True
            self.recovery_timer = 0
        if self.recovery_mode:
            self._recovery(ox, oy, otheta)
            return

        # ── forward clearance cap ────────────────────────────────────
        fwd   = self._forward_clearance(ox, oy, otheta)
        v_cap = 0.08 if fwd < 0.35 else (0.18 if fwd < 0.7 else self.max_vel)

        # ── dynamic window ───────────────────────────────────────────
        # The cap trims the reachable top speed; it cannot push the window
        # below the floor set by max_accel. Closing on an obstacle at 0.30 m/s
        # used to leave dw = [0.26, 0.08] and command a dead stop -- the one
        # velocity the base cannot actually produce.
        dw = self._dynamic_window()
        dw[1] = max(dw[0], min(dw[1], v_cap))

        vs = self._samples(dw[0], dw[1], self.vel_res)
        ws = self._samples(dw[2], dw[3], self.yawrate_res)

        # ── vectorised scoring ───────────────────────────────────────
        (best_v, best_w, best_score,
         x_all, y_all, scores, lethal_mask,
         N, T) = self._score_trajectories(
             ox, oy, otheta, vs, ws, gx_odom, gy_odom)

        # ── publish scored trajectory markers ────────────────────────
        # The fan is capped in one place, inside the publisher: it strides the
        # whole fan to self.traj_draw markers and colours them against the
        # whole fan's score range. There used to be a second cap here as well,
        # subsampling to 200 first, which normalised the colours against the
        # subsample rather than the fan.
        self._publish_scored_markers(x_all, y_all, scores, lethal_mask, N, T)

        # ── no feasible trajectory → escape ──────────────────────────
        if best_score == -np.inf:
            self.get_logger().warn('No valid trajectory — escape')
            best_escape_cost = float('inf')
            best_escape_yaw  = otheta + math.pi
            for a in np.arange(0, 2 * math.pi, 0.2):
                c = self._costmap_value(
                    ox + 0.3 * math.cos(otheta + a),
                    oy + 0.3 * math.sin(otheta + a))
                if 0 <= c < best_escape_cost:
                    best_escape_cost = c
                    best_escape_yaw  = otheta + a
            diff = np.arctan2(
                math.sin(best_escape_yaw - otheta),
                math.cos(best_escape_yaw - otheta))
            t = Twist()
            t.angular.z = float(np.clip(diff * 2.0,
                                        -self.max_yawrate, self.max_yawrate))
            t.linear.x  = -0.05
            self.cmd_pub.publish(t)
            return

        # ── publish best trajectory ──────────────────────────────────
        best_idx = int(np.argmax(scores))
        self._publish_best_traj(x_all[best_idx], y_all[best_idx])

        # ── command ──────────────────────────────────────────────────
        t = Twist()
        t.linear.x  = float(best_v)
        t.angular.z = float(np.clip(best_w,
                                     -self.max_yawrate, self.max_yawrate))
        self.cmd_pub.publish(t)

        self._tick_ms = 1e3 * (time.perf_counter() - t_tick)
        self.get_logger().info(
            f'v={best_v:.2f} ω={best_w:.2f} fwd={fwd:.2f}m '
            f'wp={self.wp_idx}/{len(self.current_path)} '
            f'dist={np.hypot(gx_map - mx, gy_map - my):.2f}m '
            f'tick={self._tick_ms:.0f}ms hz={self._tick_hz:.1f}',
            throttle_duration_sec=1.0)


def main(args=None):
    _sig()
    rclpy.init(args=args)
    node = DWAControllerNode()
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


def _sig():
    """Author signature. stderr, tty-only, so redirected output stays clean."""
    import os, sys
    if os.environ.get("NO_BANNER") == "1" or not sys.stderr.isatty():
        return
    print("  " + "".join(chr(c - 7) for c in
          (104,105,107,124,115,39,121,104,111,116,104,117)), file=sys.stderr)
