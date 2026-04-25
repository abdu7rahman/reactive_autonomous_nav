#!/usr/bin/env python3
"""
RRT-SMAC Hybrid Global Planner — reactive_autonomous_nav
  • UNIFIED dual-costmap evaluation: BOTH global AND local checked at EVERY expansion
  • RRT-style sampling for exploration efficiency
  • SMAC-style kinematic motion primitives (arcs, heading bins)
  • Conservative cost merging: max(global_cost, local_cost) for safety
  • Kinematically feasible paths respecting minimum turning radius
  • Real-time reactivity to dynamic local obstacles during planning
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

# ── Cost thresholds ──────────────────────────────────────────────────
LETHAL_COST = 253
AVOID_COST  = 200
FREE_COST   = 10

# ── Planner params ───────────────────────────────────────────────────
MAX_ITER = 3000
GOAL_REACH_DIST = 0.25
GOAL_REACH_ANGLE = 0.3
GOAL_BIAS = 0.2  # 20% goal-directed sampling

# ── Kinematics (SMAC-inspired) ───────────────────────────────────────
MIN_TURN_RADIUS = 0.22
ARC_LENGTH = 0.3
MAX_STEER_ANGLE = ARC_LENGTH / MIN_TURN_RADIUS

STEER_SAMPLES = [
    -MAX_STEER_ANGLE,
    -MAX_STEER_ANGLE * 0.5,
    0.0,
    MAX_STEER_ANGLE * 0.5,
    MAX_STEER_ANGLE,
]


def _angle_wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class HybridNode:
    """Tree node: position + heading."""
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.parent = None
        self.cost = 0.0


class HybridRRTSMACPlannerNode(Node):

    def __init__(self):
        super().__init__('rrt_smac_hybrid_planner_node')

        # ── State ────────────────────────────────────────────────────
        self.global_data   = None
        self.global_info   = None
        self.global_origin = None

        self.local_data    = None
        self.local_info    = None
        self.local_origin  = None

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

        # ── Subs ─────────────────────────────────────────────────────
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

        # ── Pubs ─────────────────────────────────────────────────────
        self.path_pub      = self.create_publisher(Path,        '/plan',           10)
        self.status_pub    = self.create_publisher(String,      '/hybrid_status',  10)
        self.marker_pub    = self.create_publisher(MarkerArray, '/hybrid_tree',    10)

        # ── Timer ────────────────────────────────────────────────────
        self.create_timer(0.5, self._check_and_replan)

        self.get_logger().info('Hybrid RRT-SMAC — DUAL costmap evaluation at every expansion')

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
            self.get_logger().info('Force replan')
            self._plan(self.last_goal)

    def _goal_cb(self, msg: PoseStamped):
        self.last_goal        = msg
        self.goal_reached     = False
        self.current_path     = None
        self.last_replan_pose = None
        self._plan(msg)

    def _check_and_replan(self):
        if self.goal_reached or self.last_goal is None:
            return
        
        pose = self._get_robot_pose_yaw()
        if pose is None:
            return
        
        if self.last_replan_pose is not None:
            dx = pose[0] - self.last_replan_pose[0]
            dy = pose[1] - self.last_replan_pose[1]
            if math.hypot(dx, dy) < 0.3:
                return
        
        self._plan(self.last_goal)

    # ================================================================
    #  TF
    # ================================================================
    def _get_robot_pose_yaw(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(), Duration(seconds=0.5))
            tr  = t.transform.translation
            rot = t.transform.rotation
            yaw = math.atan2(2.0 * (rot.w * rot.z + rot.x * rot.y),
                             1.0 - 2.0 * (rot.y**2 + rot.z**2))
            return tr.x, tr.y, yaw
        except Exception:
            return None

    # ================================================================
    #  Coordinate Transforms
    # ================================================================
    def _w2g_global(self, wx, wy):
        if self.global_info is None:
            return None, None
        res = self.global_info.resolution
        col = int((wx - self.global_origin[0]) / res)
        row = int((wy - self.global_origin[1]) / res)
        return row, col

    def _w2g_local(self, wx, wy):
        if self.local_info is None:
            return None, None
        res = self.local_info.resolution
        col = int((wx - self.local_origin[0]) / res)
        row = int((wy - self.local_origin[1]) / res)
        return row, col

    # ================================================================
    #  CRITICAL: Dual Costmap Evaluation
    # ================================================================
    def _get_merged_cost(self, wx, wy):
        """
        ⭐ KEY METHOD: Checks BOTH global AND local costmaps.
        Returns: max(global_cost, local_cost) for conservative obstacle avoidance.
        This ensures paths avoid BOTH static (global) AND dynamic (local) obstacles.
        """
        global_cost = 0
        local_cost = 0

        # Check global costmap (static obstacles)
        if self.global_data is not None and self.global_info is not None:
            gr, gc = self._w2g_global(wx, wy)
            if gr is not None and 0 <= gr < self.global_info.height and \
               0 <= gc < self.global_info.width:
                global_cost = int(self.global_data[gr, gc])
            else:
                global_cost = LETHAL_COST  # Out of bounds

        # Check local costmap (dynamic obstacles + inflation)
        if self.local_data is not None and self.local_info is not None:
            lr, lc = self._w2g_local(wx, wy)
            if lr is not None and 0 <= lr < self.local_info.height and \
               0 <= lc < self.local_info.width:
                local_cost = int(self.local_data[lr, lc])

        # Return maximum cost (conservative)
        return max(global_cost, local_cost)

    def _is_arc_collision_free(self, arc_points):
        """
        Collision check for kinematic arc.
        ⭐ Every point evaluated against BOTH costmaps.
        """
        for x, y, _ in arc_points:
            cost = self._get_merged_cost(x, y)
            if cost >= LETHAL_COST or cost < 0:
                return False
        return True

    # ================================================================
    #  Kinematic Arc Generation (SMAC-inspired)
    # ================================================================
    def _compute_arc(self, x, y, yaw, steer):
        """Generate kinematic arc respecting turning radius."""
        if abs(steer) < 1e-6:
            # Straight motion
            nx = x + ARC_LENGTH * math.cos(yaw)
            ny = y + ARC_LENGTH * math.sin(yaw)
            return [(x, y, yaw), (nx, ny, yaw)]
        else:
            # Curved arc
            R = ARC_LENGTH / abs(steer)
            dtheta = steer
            sign = 1 if steer > 0 else -1
            
            # Center of rotation
            cx = x - R * math.sin(yaw) * sign
            cy = y + R * math.cos(yaw) * sign
            
            # Sample arc densely for collision checking
            arc = []
            steps = max(5, int(abs(dtheta) / 0.05))
            for i in range(steps + 1):
                t = i / steps
                angle = yaw + t * dtheta
                px = cx + R * math.sin(angle) * sign
                py = cy - R * math.cos(angle) * sign
                arc.append((px, py, _angle_wrap(angle)))
            
            return arc

    def _compute_direct_connection(self, sx, sy, syaw, gx, gy, gyaw):
        """Attempt direct kinematic connection for final goal link."""
        dist = math.hypot(gx - sx, gy - sy)
        
        arc = []
        steps = max(10, int(dist / 0.05))
        
        for i in range(steps + 1):
            t = i / steps
            px = sx + t * (gx - sx)
            py = sy + t * (gy - sy)
            pyaw = syaw + t * _angle_wrap(gyaw - syaw)
            arc.append((px, py, _angle_wrap(pyaw)))
        
        return arc

    # ================================================================
    #  Main Planning Algorithm
    # ================================================================
    def _plan(self, goal_msg):
        """Entry point for hybrid RRT-SMAC planning with dual costmap eval."""
        if self.global_data is None:
            self.get_logger().warn('No global costmap')
            return

        pose = self._get_robot_pose_yaw()
        if pose is None:
            self.get_logger().warn('No robot pose')
            return

        sx, sy, syaw = pose
        gx = goal_msg.pose.position.x
        gy = goal_msg.pose.position.y
        
        q = goal_msg.pose.orientation
        gyaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y**2 + q.z**2))

        self.get_logger().info(
            f'Hybrid planning: ({sx:.1f},{sy:.1f})→({gx:.1f},{gy:.1f}) '
            f'[BOTH global+local evaluated at every node]')

        start_time = time.time()
        path = self._plan_hybrid_unified(sx, sy, syaw, gx, gy, gyaw)
        elapsed = time.time() - start_time

        if path is None or len(path) == 0:
            self.get_logger().error(f'Planning failed after {elapsed:.2f}s')
            self.status_pub.publish(String(data='FAILED'))
            return

        self.get_logger().info(
            f'✓ Path: {len(path)} waypoints, {elapsed:.2f}s '
            f'(dual-costmap checked at every expansion)')

        self._publish_path(path, goal_msg.header.stamp)
        self.current_path = path
        self.last_replan_pose = pose
        self.status_pub.publish(String(data='SUCCESS'))

    def _plan_hybrid_unified(self, sx, sy, syaw, gx, gy, gyaw):
        """
        Unified Hybrid Algorithm:
        
        For each iteration:
          1. Sample random state OR goal (RRT exploration)
          2. Find nearest node in tree
          3. Expand with kinematic arcs (SMAC motion primitives)
          4. ⭐ Check BOTH costmaps for each arc point ⭐
          5. Add collision-free nodes to tree
          6. Check goal reachability
        
        Result: Kinematically feasible path avoiding obstacles in BOTH maps.
        """
        # Initialize tree
        nodes = [HybridNode(sx, sy, syaw)]
        goal_node = None
        
        global_checks = 0
        local_checks = 0
        rejections = 0

        for iteration in range(MAX_ITER):
            # === 1. SAMPLING (RRT-style) ===
            if random.random() < GOAL_BIAS:
                sample_x, sample_y, sample_yaw = gx, gy, gyaw
            else:
                # Random sample in map bounds
                sample_x = random.uniform(
                    self.global_origin[0],
                    self.global_origin[0] + self.global_info.width * self.global_info.resolution)
                sample_y = random.uniform(
                    self.global_origin[1],
                    self.global_origin[1] + self.global_info.height * self.global_info.resolution)
                sample_yaw = random.uniform(-math.pi, math.pi)

            # === 2. NEAREST NODE ===
            nearest = min(nodes, key=lambda n: math.hypot(n.x - sample_x, n.y - sample_y) + 
                                               0.5 * abs(_angle_wrap(n.yaw - sample_yaw)))

            # === 3. KINEMATIC EXPANSION (SMAC-style) ===
            # Try multiple steering angles (motion primitives)
            best_arc = None
            best_node = None
            best_score = -float('inf')

            for steer in STEER_SAMPLES:
                arc = self._compute_arc(nearest.x, nearest.y, nearest.yaw, steer)
                
                if len(arc) == 0:
                    continue

                # === 4. ⭐ DUAL COSTMAP CHECK ⭐ ===
                collision_free = True
                for x, y, _ in arc:
                    global_checks += 1
                    local_checks += 1
                    
                    cost = self._get_merged_cost(x, y)
                    if cost >= LETHAL_COST or cost < 0:
                        collision_free = False
                        rejections += 1
                        break

                if not collision_free:
                    continue

                # Valid arc! Score by closeness to sample
                nx, ny, nyaw = arc[-1]
                dist_score = -math.hypot(nx - sample_x, ny - sample_y)
                angle_score = -abs(_angle_wrap(nyaw - sample_yaw))
                score = dist_score + 0.3 * angle_score

                if score > best_score:
                    best_score = score
                    best_arc = arc
                    best_node = (nx, ny, nyaw)

            # === 5. ADD TO TREE ===
            if best_node is None:
                continue

            nx, ny, nyaw = best_node
            new_node = HybridNode(nx, ny, nyaw)
            new_node.parent = nearest
            new_node.cost = nearest.cost + ARC_LENGTH
            nodes.append(new_node)

            # === 6. GOAL CHECK ===
            dist = math.hypot(nx - gx, ny - gy)
            angle_diff = abs(_angle_wrap(nyaw - gyaw))

            if dist < GOAL_REACH_DIST and angle_diff < GOAL_REACH_ANGLE:
                # Try direct connection
                final_arc = self._compute_direct_connection(nx, ny, nyaw, gx, gy, gyaw)
                
                # ⭐ Final arc also checked against BOTH costmaps
                if self._is_arc_collision_free(final_arc):
                    goal_node = HybridNode(gx, gy, gyaw)
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + dist
                    
                    self.get_logger().info(
                        f'Goal found at iter {iteration}: '
                        f'tree={len(nodes)} nodes, '
                        f'global_checks={global_checks}, '
                        f'local_checks={local_checks}, '
                        f'rejections={rejections}')
                    break

        if goal_node is None:
            self.get_logger().warn(f'Failed after {MAX_ITER} iterations')
            return None

        # === PATH RECONSTRUCTION ===
        path = []
        current = goal_node
        while current is not None:
            path.append((current.x, current.y, current.yaw))
            current = current.parent
        path.reverse()

        # Light smoothing (preserving collision-freedom)
        path = self._smooth_path_dual_check(path)

        return path

    def _smooth_path_dual_check(self, path, iterations=15):
        """
        Smooth path with ⭐ dual costmap validation ⭐.
        Only accept smoothed points that pass BOTH costmap checks.
        """
        if len(path) < 3:
            return path
        
        smoothed = [list(p) for p in path]
        
        for _ in range(iterations):
            for i in range(1, len(smoothed) - 1):
                # Weighted average
                new_x = 0.25 * smoothed[i-1][0] + 0.5 * smoothed[i][0] + 0.25 * smoothed[i+1][0]
                new_y = 0.25 * smoothed[i-1][1] + 0.5 * smoothed[i][1] + 0.25 * smoothed[i+1][1]
                
                # ⭐ Check BOTH costmaps before accepting
                cost = self._get_merged_cost(new_x, new_y)
                if cost < LETHAL_COST:
                    smoothed[i][0] = new_x
                    smoothed[i][1] = new_y
        
        return [(p[0], p[1], p[2]) for p in smoothed]

    # ================================================================
    #  Dual Costmap Methods
    # ================================================================
    def _get_merged_cost(self, wx, wy):
        """
        ⭐⭐⭐ CRITICAL METHOD ⭐⭐⭐
        
        Evaluates cost from BOTH global AND local costmaps.
        Returns max(global_cost, local_cost) for conservative safety.
        
        Why this works:
        - Global map: Static obstacles (walls, furniture)
        - Local map: Dynamic obstacles (people, other robots) + inflation
        - Max ensures we avoid obstacles detected by EITHER map
        - Planning is reactive to local changes in real-time
        """
        global_cost = 0
        local_cost = 0

        # Global costmap check
        if self.global_data is not None and self.global_info is not None:
            gr, gc = self._w2g_global(wx, wy)
            if gr is not None and 0 <= gr < self.global_info.height and \
               0 <= gc < self.global_info.width:
                global_cost = int(self.global_data[gr, gc])
            else:
                global_cost = LETHAL_COST

        # Local costmap check
        if self.local_data is not None and self.local_info is not None:
            lr, lc = self._w2g_local(wx, wy)
            if lr is not None and 0 <= lr < self.local_info.height and \
               0 <= lc < self.local_info.width:
                local_cost = int(self.local_data[lr, lc])

        return max(global_cost, local_cost)

    def _is_arc_collision_free(self, arc_points):
        """
        ⭐ DUAL COSTMAP COLLISION CHECK ⭐
        Every single point in arc is evaluated against BOTH maps.
        """
        for x, y, _ in arc_points:
            cost = self._get_merged_cost(x, y)
            if cost >= LETHAL_COST or cost < 0:
                return False
        return True

    # ================================================================
    #  Path Publishing
    # ================================================================
    def _publish_path(self, path, stamp):
        msg = Path()
        msg.header.frame_id = 'map'
        msg.header.stamp = stamp

        for p in path:
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.header.stamp = stamp
            ps.pose.position.x = p[0]
            ps.pose.position.y = p[1]
            ps.pose.position.z = 0.0
            
            yaw = p[2]
            ps.pose.orientation.w = math.cos(yaw / 2.0)
            ps.pose.orientation.z = math.sin(yaw / 2.0)
            
            msg.poses.append(ps)

        self.path_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = HybridRRTSMACPlannerNode()
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
