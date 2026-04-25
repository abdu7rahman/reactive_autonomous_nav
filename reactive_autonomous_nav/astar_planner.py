#!/usr/bin/env python3
"""
A* Global Planner — reactive_autonomous_nav
  • Octile heuristic (tight for 8-connected grids, no sqrt)
  • Collision-checked Laplacian path smoothing
  • Rich RViz markers: explored heat-map, start/goal spheres, glowing path
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
PATH_BLOCKED_LOOKAHEAD = 2.0   # m — check this far ahead for obstacles
MAX_PATH_DEVIATION     = 0.8   # m — replan if robot this far from path
MIN_REPLAN_DISTANCE    = 0.3   # m — don't replan if robot hasn't moved


class AStarPlannerNode(Node):

    def __init__(self):
        super().__init__('astar_planner_node')

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
        self.path_pub      = self.create_publisher(Path,        '/plan',             10)
        self.status_pub    = self.create_publisher(String,      '/astar_status',     10)
        self.explored_pub  = self.create_publisher(MarkerArray, '/astar_explored',   10)
        self.marker_pub    = self.create_publisher(MarkerArray, '/astar_markers',    10)

        # ── timer ────────────────────────────────────────────────────
        self.create_timer(0.5, self._check_and_replan)

        self.get_logger().info('A* Global Planner — ready')

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
        """Force replan from DWA — skip all gates."""
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
    #  Costmap helpers
    # ================================================================
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

    def _w2g(self, wx, wy):
        res = self.global_info.resolution
        col = int((wx - self.global_origin[0]) / res)
        row = int((wy - self.global_origin[1]) / res)
        return row, col

    def _g2w(self, row, col):
        res = self.global_info.resolution
        return ((col + 0.5) * res + self.global_origin[0],
                (row + 0.5) * res + self.global_origin[1])

    def _in_bounds(self, row, col):
        return (0 <= row < self.global_info.height and
                0 <= col < self.global_info.width)

    def _global_cell_cost(self, row, col) -> int:
        if not self._in_bounds(row, col):
            return 255
        v = int(self.global_data[row, col])
        return v if v >= 0 else 255

    def _merged_cell_cost(self, row, col) -> int:
        global_c = self._global_cell_cost(row, col)
        wx, wy   = self._g2w(row, col)
        local_c  = self._local_cost_at_map(wx, wy)
        if local_c < 0:
            return global_c
        return max(global_c, local_c)

    def _is_traversable(self, row, col) -> bool:
        return self._merged_cell_cost(row, col) < LETHAL_COST

    def _traversal_cost(self, cell_c: int) -> float:
        if cell_c <= FREE_COST:
            return 1.0
        if cell_c < AVOID_COST:
            return 1.0 + (cell_c - FREE_COST) / (AVOID_COST - FREE_COST) * 49.0
        return 500.0

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

        self.get_logger().debug(
            'Path valid — no replan needed', throttle_duration_sec=2.0)

    # ================================================================
    #  A* core — octile heuristic
    # ================================================================
    def _astar(self, start, goal):
        MOVES = [
            (-1,  0, 1.0), ( 1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
        ]
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g = {start: 0.0}
        explored = []

        while open_set:
            _, current = heapq.heappop(open_set)
            explored.append(current)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return list(reversed(path)), explored

            for dr, dc, move_cost in MOVES:
                nb = (current[0] + dr, current[1] + dc)
                if not self._is_traversable(*nb):
                    continue
                cell_c = self._merged_cell_cost(*nb)
                ng     = g[current] + move_cost * self._traversal_cost(cell_c)
                if nb not in g or ng < g[nb]:
                    came_from[nb] = current
                    g[nb] = ng
                    # ── octile heuristic (no sqrt) ───────────────────
                    dx = abs(nb[0] - goal[0])
                    dy = abs(nb[1] - goal[1])
                    h  = max(dx, dy) + 0.4142135 * min(dx, dy)
                    heapq.heappush(open_set, (ng + h, nb))

        return None, explored

    # ================================================================
    #  Collision-checked Laplacian smoothing
    # ================================================================
    def _smooth(self, path_world, iterations=50):
        if len(path_world) < 3:
            return path_world
        smooth = [list(p) for p in path_world]
        wd, ws = 0.5, 0.3
        for _ in range(iterations):
            for i in range(1, len(smooth) - 1):
                candidate = [smooth[i][0], smooth[i][1]]
                for j in range(2):
                    candidate[j] += wd * (path_world[i][j] - candidate[j])
                    candidate[j] += ws * (smooth[i-1][j] + smooth[i+1][j]
                                          - 2.0 * candidate[j])
                # ── collision guard: only accept if still traversable ──
                row, col = self._w2g(candidate[0], candidate[1])
                if self._merged_cell_cost(row, col) < LETHAL_COST:
                    smooth[i] = candidate
        return [tuple(p) for p in smooth]

    # ================================================================
    #  RViz marker helpers
    # ================================================================
    def _publish_explored(self, explored, path_cells):
        """Explored-cell cubes with a blue→cyan heat-map gradient."""
        path_set = set(map(tuple, path_cells)) if path_cells else set()
        rejected = [c for c in explored if tuple(c) not in path_set]
        step     = max(1, len(rejected) // 1200)
        rejected = rejected[::step]

        now = self.get_clock().now().to_msg()
        ma  = MarkerArray()

        # single CUBE_LIST marker is far cheaper than N individual cubes
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
        m.scale.z = 0.01  # thin wafer

        n = max(len(rejected), 1)
        for idx, (row, col) in enumerate(rejected):
            wx, wy = self._g2w(row, col)
            m.points.append(Point(x=wx, y=wy, z=0.02))
            # gradient from deep blue (early) → cyan (late)
            t = idx / n
            m.colors.append(ColorRGBA(
                r=0.0,
                g=0.2 + 0.6 * t,
                b=0.9 - 0.3 * t,
                a=0.25 + 0.15 * t,
            ))

        ma.markers.append(m)
        self.explored_pub.publish(ma)

    def _publish_start_goal_markers(self, rx, ry, gx, gy):
        """Bright spheres on start & goal."""
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

        if not self._is_traversable(*goal):
            s = 'UNREACHABLE: Goal inside obstacle'
            self.get_logger().warn(s)
            self.status_pub.publish(String(data=s))
            return

        if not self._is_traversable(*start):
            found = False
            for dr in range(-5, 6):
                for dc in range(-5, 6):
                    ns = (start[0] + dr, start[1] + dc)
                    if self._is_traversable(*ns):
                        start = ns
                        found = True
                        break
                if found:
                    break

        path_cells, explored = self._astar(start, goal)

        # ── RViz: explored heat-map + start/goal spheres ─────────────
        self._publish_explored(explored, path_cells)
        self._publish_start_goal_markers(rx, ry, gx, gy)

        if path_cells is None:
            s = 'UNREACHABLE: A* found no path'
            self.get_logger().warn(s)
            self.status_pub.publish(String(data=s))
            return

        path_world = [self._g2w(r, c) for r, c in path_cells]
        path_world = self._smooth(path_world)

        # ── publish nav_msgs/Path ────────────────────────────────────
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
    node = AStarPlannerNode()
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
