#!/usr/bin/env python3
"""
MPPI Local Controller — reactive_autonomous_nav (Nav2-inspired)
  • Model Predictive Path Integral control with proper path following
  • Time-correlated noise sampling with warm-starting (shift previous solution)
  • Reference trajectory generation with lookahead for smooth tracking
  • Critics: PathFollow, PathAngle, GoalAngle, GoalAlign, Obstacles, Smoothness
  • Proper costmap collision checking with inflation
  • RViz visualization: sampled trajectories + best trajectory
"""

import rclpy
import rclpy.time
import math
import numpy as np
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import Path, OccupancyGrid
from std_msgs.msg import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformListener, Buffer


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


class MPPIControllerNode(Node):

    def __init__(self):
        super().__init__('mppi_controller_node')

        # ── MPPI parameters (Nav2-inspired) ──────────────────────────
        self.time_steps   = 56          # Horizon length
        self.dt           = 0.05        # Time step (20 Hz internal model)
        self.num_samples  = 1000        # Number of trajectory samples
        self.temperature  = 0.3         # Softmax temperature (higher = more exploration)
        self.gamma        = 0.015       # Noise decay factor for time correlation

        # ── Robot kinematic limits ───────────────────────────────────
        self.max_vel       = 0.5
        self.min_vel       = -0.35
        self.max_yawrate   = 1.9
        self.max_accel     = 3.0
        self.max_yaw_accel = 3.2

        # ── Noise standard deviations ────────────────────────────────
        self.std_vel      = 0.25
        self.std_yawrate  = 0.4

        # ── Cost weights (tuned for path following) ──────────────────
        self.w_reference_cost   = 6.0    # Distance to reference trajectory
        self.w_path_angle_cost  = 2.0    # Alignment with path direction
        self.w_goal_cost        = 5.0    # Distance to goal
        self.w_goal_angle_cost  = 3.0    # Heading towards goal
        self.w_obstacle_cost    = 10.0   # Obstacle avoidance
        self.w_prefer_forward   = 5.0    # Prefer forward motion
        self.w_smoothness       = 2.0    # Control smoothness

        # ── Path following parameters ────────────────────────────────
        self.lookahead_dist      = 0.8   # How far ahead to look on path
        self.goal_tol            = 0.15  # Goal tolerance
        self.path_prune_dist     = 0.5   # Prune path points behind robot

        # ── Collision parameters ─────────────────────────────────────
        self.collision_cost      = 10000.0
        self.near_goal_threshold = 0.5
        self.lethal_cost         = 253
        self.inflation_cost      = 200

        # ── State ────────────────────────────────────────────────────
        self.current_path   = None
        self.path_xy        = None        # Cached numpy array of path points
        self.path_yaw       = None        # Cached path headings
        self.local_costmap  = None
        self.costmap_data   = None        # Cached numpy array
        self.goal_reached   = False

        # ── Control sequence (warm start) ────────────────────────────
        self.control_sequence = np.zeros((self.time_steps, 2))
        self.prev_cmd         = np.array([0.0, 0.0])

        # ── TF ───────────────────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── ROS interfaces ───────────────────────────────────────────
        self.create_subscription(Path, '/plan', self._path_cb, 10)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap',
                                 self._costmap_cb, 10)

        self.cmd_pub    = self.create_publisher(Twist,       '/cmd_vel_unstamped', 10)
        self.status_pub = self.create_publisher(String,      '/dwa_status',        10)
        self.marker_pub = self.create_publisher(MarkerArray, '/mppi_trajectories', 10)

        self.control_timer = self.create_timer(0.05, self._control_loop)  # 20 Hz
        self.get_logger().info('MPPI Controller (Nav2-style) — ready')

    # ================================================================
    #  Callbacks
    # ================================================================
    def _path_cb(self, msg: Path):
        if len(msg.poses) < 2:
            return
        self.current_path = msg
        self.goal_reached = False

        # Pre-compute path as numpy arrays
        self.path_xy = np.array([[p.pose.position.x, p.pose.position.y]
                                  for p in msg.poses])
        
        # Compute path headings (direction from each point to next)
        diffs = np.diff(self.path_xy, axis=0)
        self.path_yaw = np.arctan2(diffs[:, 1], diffs[:, 0])
        # Append last heading for the goal point
        self.path_yaw = np.append(self.path_yaw, self.path_yaw[-1])

        # Reset control sequence on new path
        self.control_sequence = np.zeros((self.time_steps, 2))
        self.get_logger().info(f'New path: {len(msg.poses)} waypoints')

    def _costmap_cb(self, msg: OccupancyGrid):
        self.local_costmap = msg
        self.costmap_data = np.array(msg.data, dtype=np.int16).reshape(
            (msg.info.height, msg.info.width))

    # ================================================================
    #  TF
    # ================================================================
    def _get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(), Duration(seconds=0.1))
            tr  = t.transform.translation
            rot = t.transform.rotation
            yaw = math.atan2(2.0 * (rot.w * rot.z + rot.x * rot.y),
                             1.0 - 2.0 * (rot.y**2 + rot.z**2))
            return np.array([tr.x, tr.y, yaw])
        except Exception:
            return None

    # ================================================================
    #  Main control loop
    # ================================================================
    def _control_loop(self):
        if self.current_path is None or self.goal_reached:
            return
        if self.path_xy is None or len(self.path_xy) < 2:
            return

        pose = self._get_robot_pose()
        if pose is None:
            return

        # Check goal reached
        goal = self.path_xy[-1]
        dist_to_goal = np.hypot(goal[0] - pose[0], goal[1] - pose[1])

        if dist_to_goal < self.goal_tol:
            self.get_logger().info('Goal reached!')
            self.goal_reached = True
            self._stop()
            self.status_pub.publish(String(data='REACHED'))
            return

        # Prune passed waypoints from path
        self._prune_path(pose)

        if len(self.path_xy) < 2:
            self._stop()
            return

        # Run MPPI optimization
        optimal_cmd = self._mppi_optimize(pose)
        if optimal_cmd is None:
            self._stop()
            return

        # Publish command
        cmd = Twist()
        cmd.linear.x  = float(optimal_cmd[0])
        cmd.angular.z = float(optimal_cmd[1])
        self.cmd_pub.publish(cmd)
        self.prev_cmd = optimal_cmd

    def _prune_path(self, pose):
        """Remove path points that are behind the robot."""
        if self.path_xy is None or len(self.path_xy) < 2:
            return

        robot_pos = pose[:2]
        dists = np.linalg.norm(self.path_xy - robot_pos, axis=1)
        closest_idx = np.argmin(dists)

        # Keep at least the closest point and everything after
        if closest_idx > 0 and dists[closest_idx] < self.path_prune_dist:
            self.path_xy  = self.path_xy[closest_idx:]
            self.path_yaw = self.path_yaw[closest_idx:]

    # ================================================================
    #  MPPI Optimization
    # ================================================================
    def _mppi_optimize(self, pose):
        K = self.num_samples
        T = self.time_steps

        # Generate control samples with time-correlated noise
        controls = self._sample_controls(K, T)

        # Rollout trajectories
        trajectories = self._rollout_trajectories(pose, controls)

        # Compute costs for each trajectory
        costs = self._compute_all_costs(trajectories, controls, pose)

        # MPPI weighting
        min_cost = np.min(costs)
        costs_shifted = costs - min_cost
        
        # Numerical stability for softmax
        weights = np.exp(-costs_shifted / self.temperature)
        weight_sum = np.sum(weights)
        
        if weight_sum < 1e-10:
            # All trajectories have very high cost
            self.get_logger().warn('All MPPI trajectories have high cost')
            return None

        weights /= weight_sum

        # Weighted average of control sequences
        weighted_controls = np.sum(weights[:, None, None] * controls, axis=0)

        # Update stored control sequence (warm start for next iteration)
        self.control_sequence = np.roll(weighted_controls, -1, axis=0)
        self.control_sequence[-1] = self.control_sequence[-2]  # Repeat last

        # Publish visualization
        self._publish_trajectories(trajectories, costs, weights)

        # Return first control
        return weighted_controls[0]

    def _sample_controls(self, K, T):
        """Generate K control sequences with time-correlated noise."""
        controls = np.zeros((K, T, 2))

        # Start from previous optimal sequence (warm start)
        baseline = self.control_sequence.copy()

        # Generate time-correlated noise
        for k in range(K):
            noise_v = np.zeros(T)
            noise_w = np.zeros(T)

            # First timestep: pure random
            noise_v[0] = np.random.normal(0, self.std_vel)
            noise_w[0] = np.random.normal(0, self.std_yawrate)

            # Subsequent timesteps: correlated with previous
            for t in range(1, T):
                noise_v[t] = self.gamma * noise_v[t-1] + np.sqrt(1 - self.gamma**2) * np.random.normal(0, self.std_vel)
                noise_w[t] = self.gamma * noise_w[t-1] + np.sqrt(1 - self.gamma**2) * np.random.normal(0, self.std_yawrate)

            controls[k, :, 0] = baseline[:, 0] + noise_v
            controls[k, :, 1] = baseline[:, 1] + noise_w

        # Keep one sample as pure baseline (no noise)
        controls[0] = baseline

        # Apply kinematic constraints
        controls = self._apply_constraints(controls)

        return controls

    def _apply_constraints(self, controls):
        """Apply velocity limits and acceleration limits."""
        K, T, _ = controls.shape

        # Enforce acceleration limits
        for t in range(1, T):
            dv = controls[:, t, 0] - controls[:, t-1, 0]
            dw = controls[:, t, 1] - controls[:, t-1, 1]

            dv = np.clip(dv, -self.max_accel * self.dt, self.max_accel * self.dt)
            dw = np.clip(dw, -self.max_yaw_accel * self.dt, self.max_yaw_accel * self.dt)

            controls[:, t, 0] = controls[:, t-1, 0] + dv
            controls[:, t, 1] = controls[:, t-1, 1] + dw

        # Enforce velocity limits
        controls[:, :, 0] = np.clip(controls[:, :, 0], self.min_vel, self.max_vel)
        controls[:, :, 1] = np.clip(controls[:, :, 1], -self.max_yawrate, self.max_yawrate)

        return controls

    def _rollout_trajectories(self, init_pose, controls):
        """Rollout trajectories using unicycle model."""
        K, T, _ = controls.shape
        trajs = np.zeros((K, T + 1, 3))
        trajs[:, 0] = init_pose

        for t in range(T):
            v = controls[:, t, 0]
            w = controls[:, t, 1]
            theta = trajs[:, t, 2]

            # Unicycle model integration
            trajs[:, t+1, 0] = trajs[:, t, 0] + v * np.cos(theta) * self.dt
            trajs[:, t+1, 1] = trajs[:, t, 1] + v * np.sin(theta) * self.dt
            trajs[:, t+1, 2] = normalize_angle(theta + w * self.dt)

        return trajs

    # ================================================================
    #  Cost Functions (Nav2-inspired critics)
    # ================================================================
    def _compute_all_costs(self, trajectories, controls, pose):
        """Compute total cost for each trajectory."""
        K = trajectories.shape[0]
        costs = np.zeros(K)

        # Generate reference trajectory on path
        ref_traj = self._generate_reference_trajectory(pose)

        # Reference trajectory following (PathFollow critic)
        costs += self.w_reference_cost * self._reference_cost(trajectories, ref_traj)

        # Path angle alignment (PathAngle critic)
        costs += self.w_path_angle_cost * self._path_angle_cost(trajectories)

        # Goal costs (only when near goal)
        goal = self.path_xy[-1]
        dist_to_goal = np.hypot(goal[0] - pose[0], goal[1] - pose[1])
        
        if dist_to_goal < self.near_goal_threshold * 3:
            costs += self.w_goal_cost * self._goal_dist_cost(trajectories, goal)
            costs += self.w_goal_angle_cost * self._goal_angle_cost(trajectories, goal)

        # Obstacle avoidance
        costs += self.w_obstacle_cost * self._obstacle_cost(trajectories)

        # Prefer forward motion
        costs += self.w_prefer_forward * self._prefer_forward_cost(controls)

        # Smoothness
        costs += self.w_smoothness * self._smoothness_cost(controls)

        return costs

    def _generate_reference_trajectory(self, pose):
        """Generate reference points along path for each timestep."""
        T = self.time_steps + 1
        ref = np.zeros((T, 2))

        # Current position and velocity estimate
        robot_pos = pose[:2]
        
        # Find closest point on path
        dists = np.linalg.norm(self.path_xy - robot_pos, axis=1)
        closest_idx = np.argmin(dists)

        # Generate reference points at each timestep
        current_dist = 0.0
        path_idx = closest_idx
        
        for t in range(T):
            # Distance we expect to travel by this timestep
            target_dist = current_dist + self.max_vel * self.dt * t * 0.7  # 70% of max speed

            # Find point on path at that distance
            accumulated_dist = 0.0
            for i in range(closest_idx, len(self.path_xy) - 1):
                seg_len = np.linalg.norm(self.path_xy[i+1] - self.path_xy[i])
                if accumulated_dist + seg_len >= target_dist:
                    # Interpolate along this segment
                    ratio = (target_dist - accumulated_dist) / (seg_len + 1e-6)
                    ratio = np.clip(ratio, 0, 1)
                    ref[t] = self.path_xy[i] + ratio * (self.path_xy[i+1] - self.path_xy[i])
                    break
                accumulated_dist += seg_len
            else:
                # Beyond end of path, use goal
                ref[t] = self.path_xy[-1]

        return ref

    def _reference_cost(self, trajectories, ref_traj):
        """Cost for deviation from reference trajectory."""
        # trajectories: (K, T+1, 3), ref_traj: (T+1, 2)
        traj_xy = trajectories[:, :, :2]  # (K, T+1, 2)
        
        # Squared distance to reference at each timestep
        diffs = traj_xy - ref_traj[None, :, :]  # (K, T+1, 2)
        dists = np.sum(diffs ** 2, axis=-1)  # (K, T+1)
        
        return np.sum(dists, axis=1)  # (K,)

    def _path_angle_cost(self, trajectories):
        """Cost for heading deviation from path direction."""
        K, T_plus_1, _ = trajectories.shape
        costs = np.zeros(K)

        # For each trajectory point, find closest path point and compare angles
        for t in range(T_plus_1):
            traj_xy = trajectories[:, t, :2]  # (K, 2)
            traj_yaw = trajectories[:, t, 2]  # (K,)

            # Find closest path point for each trajectory
            # Vectorized: (K, 1, 2) - (1, N, 2) -> (K, N, 2) -> (K, N)
            dists = np.linalg.norm(traj_xy[:, None, :] - self.path_xy[None, :, :], axis=-1)
            closest_idx = np.argmin(dists, axis=1)  # (K,)

            # Get path heading at closest point
            path_heading = self.path_yaw[closest_idx]  # (K,)

            # Angular difference cost
            angle_diff = normalize_angle(traj_yaw - path_heading)
            costs += np.abs(angle_diff)

        return costs

    def _goal_dist_cost(self, trajectories, goal):
        """Terminal distance to goal."""
        terminal_xy = trajectories[:, -1, :2]  # (K, 2)
        return np.linalg.norm(terminal_xy - goal, axis=1)

    def _goal_angle_cost(self, trajectories, goal):
        """Heading alignment towards goal."""
        K = trajectories.shape[0]
        costs = np.zeros(K)

        for t in range(trajectories.shape[1]):
            traj_xy = trajectories[:, t, :2]
            traj_yaw = trajectories[:, t, 2]

            # Angle to goal
            diff = goal - traj_xy
            angle_to_goal = np.arctan2(diff[:, 1], diff[:, 0])

            # Angular difference
            angle_diff = normalize_angle(traj_yaw - angle_to_goal)
            costs += np.abs(angle_diff)

        return costs

    def _obstacle_cost(self, trajectories):
        """Obstacle avoidance using local costmap."""
        if self.local_costmap is None or self.costmap_data is None:
            return np.zeros(trajectories.shape[0])

        cm = self.local_costmap
        ox = cm.info.origin.position.x
        oy = cm.info.origin.position.y
        res = cm.info.resolution
        W = cm.info.width
        H = cm.info.height

        K, T_plus_1, _ = trajectories.shape

        all_x = trajectories[:, :, 0]  # (K, T+1)
        all_y = trajectories[:, :, 1]

        # Convert to costmap coordinates
        mx = ((all_x - ox) / res).astype(np.int32)
        my = ((all_y - oy) / res).astype(np.int32)

        # Valid mask
        valid = (mx >= 0) & (mx < W) & (my >= 0) & (my < H)

        mx_safe = np.clip(mx, 0, W - 1)
        my_safe = np.clip(my, 0, H - 1)

        cell_costs = self.costmap_data[my_safe, mx_safe].astype(np.float64)
        
        # Out of bounds = high cost
        cell_costs[~valid] = self.lethal_cost

        # Collision = very high cost
        collision_mask = (cell_costs >= self.lethal_cost) | (cell_costs < 0)
        
        # Inflation zone = proportional cost
        inflation_mask = (cell_costs >= self.inflation_cost) & ~collision_mask

        costs = np.zeros((K, T_plus_1))
        costs[collision_mask] = self.collision_cost
        costs[inflation_mask] = (cell_costs[inflation_mask] / self.lethal_cost) * 50.0

        return np.sum(costs, axis=1)

    def _prefer_forward_cost(self, controls):
        """Penalize backward motion."""
        v = controls[:, :, 0]  # (K, T)
        backward_mask = v < 0
        cost = np.zeros_like(v)
        cost[backward_mask] = np.abs(v[backward_mask]) * 5.0
        return np.sum(cost, axis=1)

    def _smoothness_cost(self, controls):
        """Penalize jerky controls."""
        dv = np.diff(controls[:, :, 0], axis=1)
        dw = np.diff(controls[:, :, 1], axis=1)
        return np.sum(dv**2 + dw**2, axis=1)

    # ================================================================
    #  Visualization
    # ================================================================
    def _publish_trajectories(self, trajectories, costs, weights):
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()

        # Sample some trajectories to display
        K = trajectories.shape[0]
        num_display = min(30, K)
        indices = np.random.choice(K, num_display, replace=False)

        cost_min = np.min(costs)
        cost_max = np.max(costs)
        cost_range = cost_max - cost_min + 1e-6

        for i, idx in enumerate(indices):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = now
            m.ns = 'mppi_samples'
            m.id = i
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.01
            m.pose.orientation.w = 1.0

            # Color by cost (green = low, red = high)
            c = (costs[idx] - cost_min) / cost_range
            m.color = ColorRGBA(r=c, g=1.0-c, b=0.0, a=0.4)

            for t in range(0, trajectories.shape[1], 2):  # Skip every other point
                pt = Point()
                pt.x = float(trajectories[idx, t, 0])
                pt.y = float(trajectories[idx, t, 1])
                pt.z = 0.02
                m.points.append(pt)

            ma.markers.append(m)

        # Best trajectory (highest weight)
        best_idx = np.argmax(weights)
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = now
        m.ns = 'mppi_best'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.04
        m.pose.orientation.w = 1.0
        m.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0)

        for t in range(trajectories.shape[1]):
            pt = Point()
            pt.x = float(trajectories[best_idx, t, 0])
            pt.y = float(trajectories[best_idx, t, 1])
            pt.z = 0.03
            m.points.append(pt)

        ma.markers.append(m)
        self.marker_pub.publish(ma)

    def _stop(self):
        self.cmd_pub.publish(Twist())
        self.control_sequence = np.zeros((self.time_steps, 2))


def main(args=None):
    rclpy.init(args=args)
    node = MPPIControllerNode()
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