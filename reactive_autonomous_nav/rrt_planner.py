#!/usr/bin/env python3
"""
RRT Global Planner — reactive_autonomous_nav
  • Rapidly-exploring Random Tree for global path planning
  • Sampling-based approach with collision-checked edge expansion
  • Laplacian smoothing for path refinement
  • Tree visualization in RViz (green lines)
"""
print(''.join(chr(x-7) for x in [104,105,107,124,115,39,121,104,111,116,104,117]))

import rclpy
import rclpy.time
import math
import random
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

class RRTNode:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None

class RRTPlannerNode(Node):

    def __init__(self):
        super().__init__('rrt_planner_node')

        # ── RRT params ───────────────────────────────────────────────
        self.max_iter   = 2000
        self.step_size  = 0.3    # m
        self.goal_bias  = 0.1
        self.goal_reach_dist = 0.2

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
        self.status_pub    = self.create_publisher(String,      '/rrt_status',       10)
        self.tree_pub      = self.create_publisher(MarkerArray, '/rrt_tree',         10)
        self.marker_pub    = self.create_publisher(MarkerArray, '/rrt_markers',      10)

        self.get_logger().info('RRT Global Planner — ready')

    # ================================================================
    #  Callbacks
    # ================================================================
    def _global_cb(self, msg: OccupancyGrid):
        self.global_data   = np.array(msg.data, dtype=np.int16).reshape(
            (msg.info.height, msg.info.width))
        self.global_info   = msg.info
        self.global_origin = (msg.info.origin.position.x,
                              msg.info.origin.position.y)

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

    def _replan_cb(self, msg: String):
        if self.last_goal is not None:
            self._plan(self.last_goal)

    def _goal_cb(self, msg: PoseStamped):
        self.last_goal        = msg
        self.goal_reached     = False
        self.current_path     = None
        self._plan(msg)

    # ================================================================
    #  Helpers
    # ================================================================
    def _get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(), Duration(seconds=0.5))
            return t.transform.translation.x, t.transform.translation.y
        except Exception:
            return None

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
        r, c = self._w2g(wx, wy)
        if not (0 <= r < self.global_info.height and 0 <= c < self.global_info.width):
            return False
        return self.global_data[r, c] < LETHAL_COST

    def _line_of_sight(self, x1, y1, x2, y2):
        dist = math.hypot(x2 - x1, y2 - y1)
        steps = int(dist / (self.global_info.resolution * 0.5))
        for i in range(steps + 1):
            t = i / float(steps) if steps > 0 else 0
            px = x1 + t * (x2 - x1)
            py = y1 + t * (y2 - y1)
            if not self._is_free(px, py):
                return False
        return True

    # ================================================================
    #  RRT Algorithm
    # ================================================================
    def _plan(self, goal_msg):
        if self.global_data is None:
            return

        pose = self._get_robot_pose()
        if pose is None:
            return
        sx, sy = pose
        gx = goal_msg.pose.position.x
        gy = goal_msg.pose.position.y

        nodes = [RRTNode(sx, sy)]
        found = False
        t0 = time.perf_counter()

        W = self.global_info.width * self.global_info.resolution
        H = self.global_info.height * self.global_info.resolution
        ox, oy = self.global_origin

        for _ in range(self.max_iter):
            # Sample
            if random.random() < self.goal_bias:
                tx, ty = gx, gy
            else:
                tx = ox + random.random() * W
                ty = oy + random.random() * H

            # Nearest
            nearest = nodes[0]
            min_d = math.hypot(nearest.x - tx, nearest.y - ty)
            for n in nodes:
                d = math.hypot(n.x - tx, n.y - ty)
                if d < min_d:
                    min_d = d
                    nearest = n

            # Steer
            angle = math.atan2(ty - nearest.y, tx - nearest.x)
            nx = nearest.x + self.step_size * math.cos(angle)
            ny = nearest.y + self.step_size * math.sin(angle)

            # Check & Add
            if self._line_of_sight(nearest.x, nearest.y, nx, ny):
                new_node = RRTNode(nx, ny)
                new_node.parent = nearest
                nodes.append(new_node)

                if math.hypot(nx - gx, ny - gy) < self.goal_reach_dist:
                    goal_node = RRTNode(gx, gy)
                    goal_node.parent = new_node
                    nodes.append(goal_node)
                    found = True
                    break

        dt = time.perf_counter() - t0
        self.get_logger().info(f'RRT search: {len(nodes)} nodes in {dt*1000:.1f}ms')

        self._publish_tree(nodes)

        if found:
            path = []
            curr = nodes[-1]
            while curr:
                path.append((curr.x, curr.y))
                curr = curr.parent
            path.reverse()
            path = self._smooth(path)
            self._publish_path(path)
        else:
            self.get_logger().warn('RRT failed to find path')

    def _smooth(self, path, iterations=30):
        if len(path) < 3: return path
        smoothed = [list(p) for p in path]
        for _ in range(iterations):
            for i in range(1, len(smoothed) - 1):
                # Get current smoothed position (not original)
                sx, sy = smoothed[i]
                prev_x, prev_y = smoothed[i-1]
                next_x, next_y = smoothed[i+1]
                
                # Laplacian smoothing: balance between staying on path and smoothness
                # Attraction to original path point
                px, py = path[i]
                attraction = 0.5 * (px - sx), 0.5 * (py - sy)
                
                # Smoothness from neighbors (Laplacian)
                laplacian = 0.2 * (prev_x + next_x - 2.0 * sx), 0.2 * (prev_y + next_y - 2.0 * sy)
                
                nx = sx + attraction[0] + laplacian[0]
                ny = sy + attraction[1] + laplacian[1]
                
                if self._line_of_sight(prev_x, prev_y, nx, ny) and self._line_of_sight(nx, ny, next_x, next_y):
                    smoothed[i] = [nx, ny]
        return smoothed

    def _publish_path(self, path_pts):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in path_pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)
        self.current_path = msg

    def _publish_tree(self, nodes):
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'tree'
        m.id = 0
        m.type = Marker.LINE_LIST
        m.action = Marker.ADD
        m.scale.x = 0.02
        m.color = ColorRGBA(r=0.0, g=0.8, b=0.2, a=0.4)
        
        for n in nodes:
            if n.parent:
                m.points.append(Point(x=n.parent.x, y=n.parent.y, z=0.05))
                m.points.append(Point(x=n.x, y=n.y, z=0.05))
        
        ma.markers.append(m)
        self.tree_pub.publish(ma)

def main(args=None):
    rclpy.init(args=args)
    node = RRTPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
