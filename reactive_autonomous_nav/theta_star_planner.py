#!/usr/bin/env python3
"""
Theta* Global Planner — reactive_autonomous_nav (optimised)
  • Any-angle path planning via line-of-sight parent rewiring
  • Bresenham-based collision checking (direct array access)
  • Closed set prevents re-expansion
  • Global-only cost in inner loop (DWA handles local obstacles)
  • Rich RViz markers: explored heat-map, start/goal spheres
"""
print(''.join(chr(x-7) for x in [104,105,107,124,115,39,121,104,111,116,104,117]))

import rclpy
import rclpy.time
import heapq
import math
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

# ── replan gates ─────────────────────────────────────────────────────
PATH_BLOCKED_LOOKAHEAD = 2.0
MAX_PATH_DEVIATION     = 0.8
MIN_REPLAN_DISTANCE    = 0.3


class ThetaStarPlannerNode(Node):

    def __init__(self):
        super().__init__('theta_star_planner_node')

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
        self.path_pub      = self.create_publisher(Path,        '/plan',                  10)
        self.status_pub    = self.create_publisher(String,      '/theta_star_status',     10)
        self.explored_pub  = self.create_publisher(MarkerArray, '/theta_star_explored',   10)
        self.marker_pub    = self.create_publisher(MarkerArray, '/theta_star_markers',    10)

        # ── timer ────────────────────────────────────────────────────
        self.create_timer(0.5, self._check_and_replan)

        self.get_logger().info('Theta* Global Planner (optimised) — ready')

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

    def _get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(), Duration(seconds=0.5))
            return t.transform.translation.x, t.transform.translation.y
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed: {e}')
            return None

    # ================================================================
    #  Costmap helpers (used outside inner loop)
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
        """Used only for replan gating, NOT in the inner search loop."""
        h, w = self.global_info.height, self.global_info.width
        if not (0 <= row < h and 0 <= col < w):
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
        pose = self._get_robot_pose()
        if pose is None:
            return
        rx, ry = pose

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
    #  Theta* core — optimised with closed set + inlined lookups
    # ================================================================
    def _theta_star(self, start, goal):
        # ── cache everything as locals for speed ─────────────────────
        grid   = self.global_data
        H, W   = self.global_info.height, self.global_info.width
        LETHAL = LETHAL_COST

        MOVES = (
            (-1,  0, 1.0), ( 1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
        )

        open_set = [(0.0, start)]
        came_from = {start: start}
        g = {start: 0.0}
        closed = set()
        explored = []
        gr, gc = goal

        while open_set:
            _, current = heapq.heappop(open_set)

            if current in closed:
                continue
            closed.add(current)
            explored.append(current)

            if current == goal:
                path = []
                while current != came_from[current]:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return list(reversed(path)), explored

            cr, cc = current
            parent = came_from[current]

            for dr, dc, move_cost in MOVES:
                nr, nc = cr + dr, cc + dc

                # ── inlined bounds + traversability check ────────────
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                cell_v = int(grid[nr, nc])
                if cell_v < 0 or cell_v >= LETHAL:
                    continue
                nb = (nr, nc)
                if nb in closed:
                    continue

                # ── Theta* line-of-sight rewiring ────────────────────
                if self._line_of_sight_fast(parent, nb, grid, H, W, LETHAL):
                    dist = math.hypot(nr - parent[0], nc - parent[1])
                    ng   = g[parent] + dist
                    if nb not in g or ng < g[nb]:
                        came_from[nb] = parent
                        g[nb] = ng
                        dx = abs(nr - gr)
                        dy = abs(nc - gc)
                        h  = max(dx, dy) + 0.4142135 * min(dx, dy)
                        heapq.heappush(open_set, (ng + h, nb))
                else:
                    # ── inlined traversal cost (consistent penalty scaling) ───
                    if cell_v <= FREE_COST:
                        tc = 1.0
                    elif cell_v < AVOID_COST:
                        # Linear interpolation: 1.0 → 50.0
                        tc = 1.0 + (cell_v - FREE_COST) / (AVOID_COST - FREE_COST) * 49.0
                    elif cell_v < LETHAL_COST:
                        # High but not infinite for near-lethal areas
                        tc = 50.0 + (cell_v - AVOID_COST) / (LETHAL_COST - AVOID_COST) * 200.0
                    else:
                        # Lethal obstacles
                        tc = 1000.0

                    ng = g[current] + move_cost * tc
                    if nb not in g or ng < g[nb]:
                        came_from[nb] = current
                        g[nb] = ng
                        dx = abs(nr - gr)
                        dy = abs(nc - gc)
                        h  = max(dx, dy) + 0.4142135 * min(dx, dy)
                        heapq.heappush(open_set, (ng + h, nb))

        return None, explored

    # ================================================================
    #  Line-of-sight — fast Bresenham with direct array access
    # ================================================================
    @staticmethod
    def _line_of_sight_fast(p1, p2, grid, H, W, lethal):
        """Bresenham line check using direct numpy array indexing."""
        r0, c0 = p1
        r1, c1 = p2
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r1 > r0 else -1
        sc = 1 if c1 > c0 else -1
        err = dr - dc
        while True:
            if not (0 <= r0 < H and 0 <= c0 < W):
                return False
            v = grid[r0, c0]
            if v < 0 or v >= lethal:
                return False
            if r0 == r1 and c0 == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r0  += sr
            if e2 < dr:
                err += dr
                c0  += sc
        return True

    # ================================================================
    #  RViz marker helpers
    # ================================================================
    def _publish_explored(self, explored, path_cells):
        path_set = set(map(tuple, path_cells)) if path_cells else set()
        rejected = [c for c in explored if tuple(c) not in path_set]
        step     = max(1, len(rejected) // 1200)
        rejected = rejected[::step]

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

        n = max(len(rejected), 1)
        for idx, (row, col) in enumerate(rejected):
            wx, wy = self._g2w(row, col)
            m.points.append(Point(x=wx, y=wy, z=0.02))
            t = idx / n
            m.colors.append(ColorRGBA(
                r=0.4 + 0.5 * t,
                g=0.0 + 0.2 * t,
                b=0.8 - 0.2 * t,
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
        pose = self._get_robot_pose()
        if pose is None:
            self.get_logger().warn('Cannot plan: _get_robot_pose returned None')
            return
        rx, ry = pose
        gx = goal_msg.pose.position.x
        gy = goal_msg.pose.position.y

        if math.hypot(gx - rx, gy - ry) < 0.15:
            return

        start = self._w2g(rx, ry)
        goal  = self._w2g(gx, gy)

        H, W = self.global_info.height, self.global_info.width
        gr, gc = goal
        if not (0 <= gr < H and 0 <= gc < W):
            self.get_logger().warn('Goal outside costmap bounds')
            return

        gv = int(self.global_data[gr, gc])
        if gv < 0 or gv >= LETHAL_COST:
            s = 'UNREACHABLE: Goal inside obstacle'
            self.get_logger().warn(s)
            self.status_pub.publish(String(data=s))
            return

        sr, sc = start
        if not (0 <= sr < H and 0 <= sc < W) or \
           int(self.global_data[sr, sc]) >= LETHAL_COST:
            found = False
            for dr in range(-5, 6):
                for dc in range(-5, 6):
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < H and 0 <= nc < W:
                        v = int(self.global_data[nr, nc])
                        if 0 <= v < LETHAL_COST:
                            start = (nr, nc)
                            found = True
                            break
                if found:
                    break

        import time
        t0 = time.perf_counter()
        path_cells, explored = self._theta_star(start, goal)
        dt = time.perf_counter() - t0
        self.get_logger().info(
            f'Theta* search: {len(explored)} nodes in {dt*1000:.1f}ms')

        # ── RViz ─────────────────────────────────────────────────────
        self._publish_explored(explored, path_cells)
        self._publish_start_goal_markers(rx, ry, gx, gy)

        if path_cells is None:
            s = 'UNREACHABLE: Theta* found no path'
            self.get_logger().warn(s)
            self.status_pub.publish(String(data=s))
            return

        path_world = [self._g2w(r, c) for r, c in path_cells]

        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp    = self.get_clock().now().to_msg()
        for wx, wy in path_world:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x    = wx
            ps.pose.position.y    = wy
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.path_pub.publish(path_msg)

        self.current_path     = path_msg
        self.last_replan_pose = (rx, ry)
        self.get_logger().info(
            f'Path: {len(path_msg.poses)} wps '
            f'({rx:.2f},{ry:.2f})→({gx:.2f},{gy:.2f})')


def main(args=None):
    rclpy.init(args=args)
    node = ThetaStarPlannerNode()
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
