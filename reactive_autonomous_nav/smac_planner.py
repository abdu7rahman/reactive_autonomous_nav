#!/usr/bin/env python3
"""
SMAC-style Hybrid-A* Global Planner — reactive_autonomous_nav
  • State = (x, y, heading) with 72 heading bins (5° each)
  • Forward-arc motion primitives respecting min turning radius
  • Dubins analytic expansion when close to goal
  • Collision checking along arcs on the global costmap
  • RViz markers: orange→red explored gradient
"""
print(''.join(chr(x-7) for x in [104,105,107,124,115,39,121,104,111,116,104,117]))

import rclpy
import rclpy.time
import heapq
import math
import time
import numpy as np

from rclpy.node             import Node
from rclpy.duration         import Duration
from rclpy.qos              import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from tf2_ros                import TransformListener, Buffer
from nav_msgs.msg           import OccupancyGrid, Path
from geometry_msgs.msg      import PoseStamped, Point
from std_msgs.msg           import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

# ── cost thresholds ──────────────────────────────────────────────────
LETHAL_COST = 253
AVOID_COST  = 200
FREE_COST   = 10

# ── heading discretisation ───────────────────────────────────────────
NUM_HEADINGS  = 72                       # 5° per bin
HEADING_RES   = 2.0 * math.pi / NUM_HEADINGS

# ── robot kinematics (TurtleBot 4) ───────────────────────────────────
MIN_TURN_RADIUS = 0.22                   # m
ARC_LENGTH      = 0.15                   # m per motion primitive step
MAX_STEER_ANGLE = ARC_LENGTH / MIN_TURN_RADIUS  # ~0.68 rad

# ── steering samples ────────────────────────────────────────────────
STEER_ANGLES = [
    -MAX_STEER_ANGLE,
    -MAX_STEER_ANGLE * 0.5,
    0.0,
    MAX_STEER_ANGLE * 0.5,
    MAX_STEER_ANGLE,
]

# ── analytic expansion ──────────────────────────────────────────────
ANALYTIC_EXPANSION_DIST = 2.0   # m — try Dubins when this close

# ── replan gates ─────────────────────────────────────────────────────
PATH_BLOCKED_LOOKAHEAD = 2.0
MAX_PATH_DEVIATION     = 0.8
MIN_REPLAN_DISTANCE    = 0.3


def _angle_wrap(a):
    """Wrap angle to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _heading_to_bin(theta):
    return int(round(_angle_wrap(theta) / HEADING_RES)) % NUM_HEADINGS


def _bin_to_heading(b):
    return b * HEADING_RES - math.pi


class SmacPlannerNode(Node):

    def __init__(self):
        super().__init__('smac_planner_node')

        # ── state ────────────────────────────────────────────────────
        self.global_data   = None
        self.global_info   = None
        self.global_origin = None

        self.local_data    = None
        self.local_info    = None
        self.local_origin  = None

        self.odom_to_map       = None
        self.last_goal         = None
        self.current_path      = None
        self.goal_reached      = False
        self.last_replan_pose  = None

        # ── TF ───────────────────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        cmap_qos = QoSProfile(
            depth       = 1,
            durability  = DurabilityPolicy.TRANSIENT_LOCAL,
            reliability = ReliabilityPolicy.RELIABLE,
        )

        # ── subs ─────────────────────────────────────────────────────
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap',
                                 self._global_cb,  cmap_qos)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap',
                                 self._local_cb,   cmap_qos)
        self.create_subscription(PoseStamped,   '/goal_pose',
                                 self._goal_cb,    10)
        self.create_subscription(String,        '/replan_request',
                                 self._replan_cb,  10)
        self.create_subscription(String,        '/dwa_status',
                                 self._status_cb,  10)

        # ── pubs ─────────────────────────────────────────────────────
        self.path_pub      = self.create_publisher(Path,        '/plan',               10)
        self.status_pub    = self.create_publisher(String,      '/smac_status',        10)
        self.explored_pub  = self.create_publisher(MarkerArray, '/smac_explored',      10)
        self.marker_pub    = self.create_publisher(MarkerArray, '/smac_markers',       10)

        # ── timer ────────────────────────────────────────────────────
        self.create_timer(0.5, self._check_and_replan)

        self.get_logger().info('SMAC Hybrid-A* Global Planner — ready')

    # ================================================================
    #  Callbacks
    # ================================================================
    def _global_cb(self, msg: OccupancyGrid):
        self.global_data   = np.array(msg.data, dtype=np.int16).reshape(
            (msg.info.height, msg.info.width))
        self.global_info   = msg.info
        self.global_origin = (msg.info.origin.position.x,
                              msg.info.origin.position.y)
        self.get_logger().info(
            f'Global costmap: {msg.info.width}×{msg.info.height} '
            f'res={msg.info.resolution:.3f}',
            throttle_duration_sec=10.0)

    def _local_cb(self, msg: OccupancyGrid):
        self.local_data   = np.array(msg.data, dtype=np.int16).reshape(
            (msg.info.height, msg.info.width))
        self.local_info   = msg.info
        self.local_origin = (msg.info.origin.position.x,
                             msg.info.origin.position.y)

    def _status_cb(self, msg: String):
        if msg.data == 'REACHED':
            self.goal_reached = True
            self.last_goal    = None
            self.get_logger().info('Goal reached — stopping replanning')

    def _replan_cb(self, msg: String):
        if self.last_goal is not None:
            self.get_logger().info('Force replan from DWA')
            self._plan(self.last_goal)

    def _goal_cb(self, msg: PoseStamped):
        self.last_goal        = msg
        self.goal_reached     = False
        self.current_path     = None
        self.last_replan_pose = None
        self._plan(msg)

    # ================================================================
    #  TF helpers
    # ================================================================
    def _get_odom_to_map(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'odom', rclpy.time.Time(), Duration(seconds=0.2))
            tr  = t.transform.translation
            rot = t.transform.rotation
            yaw = math.atan2(2.0 * (rot.w * rot.z + rot.x * rot.y),
                             1.0 - 2.0 * (rot.y**2 + rot.z**2))
            self.odom_to_map = (tr.x, tr.y, yaw)
        except Exception:
            pass

    def _get_robot_pose_yaw(self):
        """Return (x, y, yaw) or None."""
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(), Duration(seconds=0.5))
            tr  = t.transform.translation
            rot = t.transform.rotation
            yaw = math.atan2(2.0 * (rot.w * rot.z + rot.x * rot.y),
                             1.0 - 2.0 * (rot.y**2 + rot.z**2))
            return tr.x, tr.y, yaw
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return None

    # ================================================================
    #  Costmap helpers
    # ================================================================
    def _w2g(self, wx, wy):
        res = self.global_info.resolution
        col = int((wx - self.global_origin[0]) / res)
        row = int((wy - self.global_origin[1]) / res)
        return row, col

    def _g2w(self, row, col):
        res = self.global_info.resolution
        return ((col + 0.5) * res + self.global_origin[0],
                (row + 0.5) * res + self.global_origin[1])

    def _is_free(self, wx, wy):
        """Check if world coordinate is free using merged costmaps."""
        row, col = self._w2g(wx, wy)
        return self._is_traversable(row, col)
    
    def _is_traversable(self, row, col) -> bool:
        """Check if a grid cell is traversable (merged global + local)."""
        return self._merged_cell_cost(row, col) < LETHAL_COST

    def _local_cost_at_map(self, wx_map, wy_map) -> int:
        if self.local_data is None or self.odom_to_map is None:
            return -1
        tmx, tmy, tyaw = self.odom_to_map
        dx = wx_map - tmx
        dy = wy_map - tmy
        wx_odom =  math.cos(tyaw) * dx + math.sin(tyaw) * dy
        wy_odom = -math.sin(tyaw) * dx + math.cos(tyaw) * dy
        res = self.local_info.resolution
        col = int((wx_odom - self.local_origin[0]) / res)
        row = int((wy_odom - self.local_origin[1]) / res)
        if not (0 <= col < self.local_info.width and
                0 <= row < self.local_info.height):
            return -1
        return int(self.local_data[row, col])

    def _merged_cell_cost(self, row, col) -> int:
        H, W = self.global_info.height, self.global_info.width
        if not (0 <= row < H and 0 <= col < W):
            return 255
        gc = int(self.global_data[row, col])
        if gc < 0:
            gc = 255
        wx, wy = self._g2w(row, col)
        lc = self._local_cost_at_map(wx, wy)
        return max(gc, lc) if lc >= 0 else gc

    # ================================================================
    #  Replan gating
    # ================================================================
    def _is_path_blocked(self, rx, ry) -> bool:
        if self.current_path is None or self.local_data is None:
            return False
        for ps in self.current_path.poses:
            wp = ps.pose.position
            d  = math.hypot(wp.x - rx, wp.y - ry)
            if d > PATH_BLOCKED_LOOKAHEAD:
                continue
            row, col = self._w2g(wp.x, wp.y)
            c = self._merged_cell_cost(row, col)
            if c >= LETHAL_COST:
                self.get_logger().info(
                    f'Path blocked at ({wp.x:.2f},{wp.y:.2f}) cost={c}')
                return True
        return False

    def _path_deviation(self, rx, ry) -> float:
        if self.current_path is None:
            return 0.0
        min_d = float('inf')
        for ps in self.current_path.poses:
            wp = ps.pose.position
            d  = math.hypot(wp.x - rx, wp.y - ry)
            if d < min_d:
                min_d = d
        return min_d

    def _robot_has_moved(self, rx, ry) -> bool:
        if self.last_replan_pose is None:
            return True
        lx, ly = self.last_replan_pose
        return math.hypot(rx - lx, ry - ly) >= MIN_REPLAN_DISTANCE

    def _check_and_replan(self):
        if self.last_goal is None or self.goal_reached:
            return
        if self.global_data is None or self.current_path is None:
            return
        self._get_odom_to_map()
        p = self._get_robot_pose_yaw()
        if p is None:
            return
        rx, ry, _ = p

        if self._is_path_blocked(rx, ry):
            self.get_logger().info('Replan: path blocked by local obstacle')
            self._plan(self.last_goal)
            return

        dev = self._path_deviation(rx, ry)
        if dev > MAX_PATH_DEVIATION:
            if self._robot_has_moved(rx, ry):
                self.get_logger().info(
                    f'Replan: deviation {dev:.2f}m > {MAX_PATH_DEVIATION}m')
                self._plan(self.last_goal)
            return

    # ================================================================
    #  Motion primitives — forward arcs
    # ================================================================
    @staticmethod
    def _expand_arc(wx, wy, theta, steer, arc_len, n_steps=5):
        """Generate waypoints along a forward arc.
        Returns list of (x, y, theta) and the total arc length."""
        if abs(steer) < 1e-6:
            # Straight line
            dx = arc_len * math.cos(theta)
            dy = arc_len * math.sin(theta)
            pts = [(wx + dx, wy + dy, theta)]
            return pts, arc_len

        radius  = arc_len / steer
        dtheta  = steer
        dt      = 1.0 / n_steps
        pts = []
        cx, cy, ct = wx, wy, theta
        step_len = arc_len / n_steps
        for _ in range(n_steps):
            ct += dtheta * dt
            cx += step_len * math.cos(ct)
            cy += step_len * math.sin(ct)
            pts.append((cx, cy, ct))
        return pts, arc_len

    # ================================================================
    #  Dubins-like analytic expansion (simplified 3-segment)
    # ================================================================
    def _try_dubins(self, wx, wy, wt, gx, gy, gt):
        """Simple analytic expansion: turn-straight-turn.
        Returns list of (x, y, theta) waypoints or None if blocked."""
        dx = gx - wx
        dy = gy - wy
        dist = math.hypot(dx, dy)
        if dist < 0.05:
            return [(gx, gy, gt)]

        # target heading for straight segment
        bearing = math.atan2(dy, dx)
        turn1 = _angle_wrap(bearing - wt)
        turn2 = _angle_wrap(gt - bearing)

        # check turning radius feasibility
        if abs(turn1) > math.pi * 0.75 or abs(turn2) > math.pi * 0.75:
            return None   # too sharp, skip

        pts = []
        # ── segment 1: turn towards goal ─────────────────────────────
        n_turn1 = max(1, int(abs(turn1) / 0.1))
        cx, cy, ct = wx, wy, wt
        step_d = 0.05
        for i in range(n_turn1):
            ct += turn1 / n_turn1
            cx += step_d * math.cos(ct)
            cy += step_d * math.sin(ct)
            if not self._is_free(cx, cy):
                return None
            pts.append((cx, cy, ct))

        # ── segment 2: straight ──────────────────────────────────────
        straight_dist = dist - abs(turn1) * 0.05 - abs(turn2) * 0.05
        if straight_dist > 0:
            n_straight = max(1, int(straight_dist / 0.05))
            sd = straight_dist / n_straight
            for _ in range(n_straight):
                cx += sd * math.cos(bearing)
                cy += sd * math.sin(bearing)
                if not self._is_free(cx, cy):
                    return None
                pts.append((cx, cy, bearing))

        # ── segment 3: turn to goal heading ──────────────────────────
        n_turn2 = max(1, int(abs(turn2) / 0.1))
        for i in range(n_turn2):
            ct = bearing + turn2 * (i + 1) / n_turn2
            cx += step_d * math.cos(ct)
            cy += step_d * math.sin(ct)
            if not self._is_free(cx, cy):
                return None
            pts.append((cx, cy, ct))

        pts.append((gx, gy, gt))
        return pts

    # ================================================================
    #  Hybrid-A* core
    # ================================================================
    def _hybrid_astar(self, start_w, goal_w):
        """start_w, goal_w = (wx, wy, theta) in world coordinates."""
        sx, sy, st = start_w
        gx, gy, gt = goal_w

        start_bin = (_heading_to_bin(st),)
        sr, sc = self._w2g(sx, sy)
        start_key = (sr, sc, start_bin[0])

        gr, gc = self._w2g(gx, gy)
        goal_key = (gr, gc, _heading_to_bin(gt))

        # f, counter, key, (wx, wy, wt)
        counter = 0
        open_set = [(0.0, counter, start_key, (sx, sy, st))]
        came_from = {}                    # key → (parent_key, list of (wx,wy,wt))
        g_cost = {start_key: 0.0}
        closed = set()
        explored = []

        while open_set:
            f, _, current_key, current_w = heapq.heappop(open_set)

            if current_key in closed:
                continue
            closed.add(current_key)
            explored.append((current_w[0], current_w[1]))

            cwx, cwy, cwt = current_w

            # ── goal check (position tolerance) ──────────────────────
            d2g = math.hypot(cwx - gx, cwy - gy)
            if d2g < 0.15:
                # reconstruct
                return self._reconstruct(came_from, current_key, start_key,
                                         sx, sy, st), explored

            # ── analytic expansion ───────────────────────────────────
            if d2g < ANALYTIC_EXPANSION_DIST:
                dubins = self._try_dubins(cwx, cwy, cwt, gx, gy, gt)
                if dubins is not None:
                    # stitch dubins onto the reconstructed path
                    path = self._reconstruct(came_from, current_key,
                                             start_key, sx, sy, st)
                    path.extend(dubins)
                    return path, explored

            # ── expand motion primitives ─────────────────────────────
            for steer in STEER_ANGLES:
                arc_pts, arc_len = self._expand_arc(
                    cwx, cwy, cwt, steer, ARC_LENGTH)

                # collision check along arc using merged costmap
                blocked = False
                for px, py, _ in arc_pts:
                    if not self._is_free(px, py):
                        blocked = True
                        break
                if blocked:
                    continue

                nx, ny, nt = arc_pts[-1]
                nr, nc = self._w2g(nx, ny)
                nb = _heading_to_bin(nt)
                nb_key = (nr, nc, nb)

                if nb_key in closed:
                    continue

                # cost: arc_len + small penalty for steering
                new_g = g_cost[current_key] + arc_len + abs(steer) * 0.1

                # cell traversal cost using merged costmap
                cell_cost = self._merged_cell_cost(nr, nc)
                if cell_cost > FREE_COST and cell_cost < LETHAL_COST:
                    new_g += (cell_cost - FREE_COST) * 0.02

                if nb_key not in g_cost or new_g < g_cost[nb_key]:
                    g_cost[nb_key] = new_g
                    came_from[nb_key] = (current_key, arc_pts)

                    # heuristic: Euclidean + heading penalty
                    hdist = math.hypot(nx - gx, ny - gy)
                    hhead = abs(_angle_wrap(nt - gt)) * MIN_TURN_RADIUS
                    h = max(hdist, hhead)

                    counter += 1
                    heapq.heappush(open_set, (new_g + h, counter, nb_key,
                                              (nx, ny, nt)))

        return None, explored

    def _reconstruct(self, came_from, current_key, start_key, sx, sy, st):
        """Reconstruct the path as a list of (wx, wy, theta)."""
        segments = []
        k = current_key
        while k in came_from:
            parent_key, arc_pts = came_from[k]
            segments.append(arc_pts)
            k = parent_key
        segments.reverse()
        path = [(sx, sy, st)]
        for seg in segments:
            path.extend(seg)
        return path

    # ================================================================
    #  RViz marker helpers
    # ================================================================
    def _publish_explored(self, explored):
        step     = max(1, len(explored) // 1200)
        sampled  = explored[::step]

        now = self.get_clock().now().to_msg()
        ma  = MarkerArray()

        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp    = now
        m.ns     = 'explored'
        m.id     = 0
        m.type   = Marker.CUBE_LIST
        m.action = Marker.ADD
        res = self.global_info.resolution
        m.scale.x = res
        m.scale.y = res
        m.scale.z = 0.01

        n = max(len(sampled), 1)
        for idx, (wx, wy) in enumerate(sampled):
            m.points.append(Point(x=wx, y=wy, z=0.02))
            # orange → red gradient
            t = idx / n
            m.colors.append(ColorRGBA(
                r=0.9 + 0.1 * t,
                g=0.5 - 0.4 * t,
                b=0.0,
                a=0.25 + 0.15 * t,
            ))

        ma.markers.append(m)
        self.explored_pub.publish(ma)

    def _publish_start_goal_markers(self, rx, ry, gx, gy):
        now = self.get_clock().now().to_msg()
        ma  = MarkerArray()

        for i, (x, y, r, g, b, label) in enumerate([
            (rx, ry, 0.2, 0.8, 1.0, 'start'),
            (gx, gy, 1.0, 0.3, 0.3, 'goal'),
        ]):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = now
            m.ns = label
            m.id = i
            m.type   = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x    = x
            m.pose.position.y    = y
            m.pose.position.z    = 0.25
            m.pose.orientation.w = 1.0
            m.scale.x = 0.3
            m.scale.y = 0.3
            m.scale.z = 0.3
            m.color = ColorRGBA(r=r, g=g, b=b, a=0.9)
            ma.markers.append(m)

        self.marker_pub.publish(ma)

    # ================================================================
    #  Plan
    # ================================================================
    def _plan(self, goal_msg: PoseStamped):
        if self.global_data is None:
            self.get_logger().warn('Cannot plan: global_data is None')
            return
        self._get_odom_to_map()
        if self.odom_to_map is None:
            self.get_logger().warn('Cannot plan: odom_to_map is None')
        p = self._get_robot_pose_yaw()
        if p is None:
            self.get_logger().warn('Cannot plan: robot pose unavailable')
            return
        rx, ry, ryaw = p
        gx = goal_msg.pose.position.x
        gy = goal_msg.pose.position.y
        # extract goal yaw from quaternion
        q = goal_msg.pose.orientation
        gyaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y**2 + q.z**2))

        if math.hypot(gx - rx, gy - ry) < 0.15:
            return

        # check goal is free using merged costmap
        gr, gc = self._w2g(gx, gy)
        if not self._is_traversable(gr, gc):
            s = 'UNREACHABLE: Goal inside obstacle'
            self.get_logger().warn(s)
            self.status_pub.publish(String(data=s))
            return

        t0 = time.perf_counter()
        result, explored = self._hybrid_astar(
            (rx, ry, ryaw), (gx, gy, gyaw))
        dt = time.perf_counter() - t0
        self.get_logger().info(
            f'Hybrid-A* search: {len(explored)} nodes in {dt*1000:.1f}ms')

        # ── RViz ─────────────────────────────────────────────────────
        self._publish_explored(explored)
        self._publish_start_goal_markers(rx, ry, gx, gy)

        if result is None:
            s = 'UNREACHABLE: Hybrid-A* found no path'
            self.get_logger().warn(s)
            self.status_pub.publish(String(data=s))
            return

        # ── publish nav_msgs/Path ────────────────────────────────────
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp    = self.get_clock().now().to_msg()
        for wx, wy, wt in result:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x    = wx
            ps.pose.position.y    = wy
            ps.pose.orientation.z = math.sin(wt / 2.0)
            ps.pose.orientation.w = math.cos(wt / 2.0)
            path_msg.poses.append(ps)
        self.path_pub.publish(path_msg)

        self.current_path     = path_msg
        self.last_replan_pose = (rx, ry)
        self.get_logger().info(
            f'Path: {len(path_msg.poses)} wps '
            f'({rx:.2f},{ry:.2f})→({gx:.2f},{gy:.2f})')


def main(args=None):
    rclpy.init(args=args)
    node = SmacPlannerNode()
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
