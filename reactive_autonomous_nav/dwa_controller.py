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


class DWAControllerNode(Node):

    def __init__(self):
        super().__init__('dwa_controller_node')

        # ── DWA params ───────────────────────────────────────────────
        self.max_vel            = 0.50
        self.min_vel            = 0.0
        self.max_yawrate        = 2.0
        self.max_accel          = 0.4
        self.max_dyawrate       = 1.0
        self.vel_res            = 0.02
        self.yawrate_res        = 0.04
        self.predict_time       = 2.5
        self.dt                 = 0.1
        self.heading_cost_gain  = 5.0
        self.speed_cost_gain    = 0.5
        self.obstacle_cost_gain = 5.0
        self.lookahead_wps      = 3
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
        self.driven_path.header.frame_id = 'map'

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
        self.status_pub      = self.create_publisher(String,      '/dwa_status',        10)
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

        map_pose = self._get_tf('map', 'base_link')
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
        self.costmap_data   = np.array(msg.data, dtype=np.int16).reshape(
            (msg.info.height, msg.info.width))
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
        end_x     = x[:, -1]
        end_y     = y[:, -1]
        end_theta = theta[:, -1]
        diff = np.arctan2(gy_odom - end_y, gx_odom - end_x) - end_theta
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
    #  Forward clearance (3-ray cone)
    # ================================================================
    def _forward_clearance(self, ox, oy, otheta) -> float:
        min_clear = float('inf')
        for angle_off in [-0.3, 0.0, 0.3]:
            a = otheta + angle_off
            for d in np.arange(0.1, 2.0, 0.05):
                c = self._costmap_value(ox + d * np.cos(a),
                                        oy + d * np.sin(a))
                if c >= LETHAL_COST:
                    min_clear = min(min_clear, d)
                    break
        return min_clear

    # ================================================================
    #  Stuck detection / recovery
    # ================================================================
    def _is_stuck(self):
        if len(self.position_history) < 50:
            return False
        p0, p1 = self.position_history[0], self.position_history[-1]
        return np.hypot(p1[0] - p0[0], p1[1] - p0[1]) < self.stuck_threshold

    def _recovery(self):
        self.recovery_timer += 1
        t = Twist()
        if self.recovery_timer <= 20:
            t.angular.z = 0.8 * self.recovery_dir
        else:
            t.linear.x = 0.1
            if self.recovery_timer >= 30:
                self.recovery_mode  = False
                self.recovery_timer = 0
                self.recovery_dir   = -self.recovery_dir
                self.position_history.clear()
                self.replan_pub.publish(String(data='replan'))
                self.get_logger().info('Recovery done — requesting replan')
        self.cmd_pub.publish(t)

    # ================================================================
    #  Driven-path breadcrumb
    # ================================================================
    def _record_pose(self, x, y):
        ps = PoseStamped()
        ps.header.frame_id    = 'map'
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
        m.header.frame_id = 'odom'
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

        for i in range(N):
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
        m.header.frame_id = 'odom'
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
        m.header.frame_id = 'map'
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
        if self.current_pose is None or self.current_path is None:
            return
        if self.costmap_data is None:
            self.get_logger().warn('Waiting for costmap…',
                                   throttle_duration_sec=3.0)
            return

        map_pose = self._get_tf('map', 'base_link')
        if map_pose is None:
            return
        mx, my, mtheta = map_pose

        odom_pose = self._get_tf('odom', 'base_link')
        ox, oy, otheta = odom_pose if odom_pose else (mx, my, mtheta)

        self.position_history.append((mx, my))
        self._record_pose(mx, my)

        # ── advance waypoint ─────────────────────────────────────────
        while self.wp_idx < len(self.current_path) - 1:
            wp = self.current_path[self.wp_idx].pose.position
            if np.hypot(wp.x - mx, wp.y - my) < self.wp_tol:
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
        tidx   = min(self.wp_idx + self.lookahead_wps,
                     len(self.current_path) - 1)
        gx_map = self.current_path[tidx].pose.position.x
        gy_map = self.current_path[tidx].pose.position.y
        self._publish_goal_marker(gx_map, gy_map)

        # ── convert goal map→odom for DWA rollout ────────────────────
        odom_from_map = self._get_tf('odom', 'map')
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
            self._recovery()
            return

        # ── forward clearance cap ────────────────────────────────────
        fwd   = self._forward_clearance(ox, oy, otheta)
        v_cap = 0.08 if fwd < 0.35 else (0.18 if fwd < 0.7 else self.max_vel)

        # ── dynamic window ───────────────────────────────────────────
        dw = self._dynamic_window()
        dw[1] = min(dw[1], v_cap)

        vs = np.arange(dw[0], dw[1] + self.vel_res,     self.vel_res)
        ws = np.arange(dw[2], dw[3] + self.yawrate_res,  self.yawrate_res)
        if vs.size == 0:
            vs = np.array([0.0])

        # ── vectorised scoring ───────────────────────────────────────
        (best_v, best_w, best_score,
         x_all, y_all, scores, lethal_mask,
         N, T) = self._score_trajectories(
             ox, oy, otheta, vs, ws, gx_odom, gy_odom)

        # ── publish scored trajectory markers ────────────────────────
        # subsample trajectories for RViz if there are many
        max_viz = 200
        if N > max_viz:
            viz_idx = np.linspace(0, N - 1, max_viz, dtype=int)
            self._publish_scored_markers(
                x_all[viz_idx], y_all[viz_idx],
                scores[viz_idx], lethal_mask[viz_idx],
                max_viz, T)
        else:
            self._publish_scored_markers(
                x_all, y_all, scores, lethal_mask, N, T)

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

        self.get_logger().info(
            f'v={best_v:.2f} ω={best_w:.2f} fwd={fwd:.2f}m '
            f'wp={self.wp_idx}/{len(self.current_path)} '
            f'dist={np.hypot(gx_map - mx, gy_map - my):.2f}m',
            throttle_duration_sec=1.0)


def main(args=None):
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
