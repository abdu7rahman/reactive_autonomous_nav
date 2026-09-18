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
import time
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


class MPPIControllerNode(Node):

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

    # The interval this loop is actually achieving, in seconds, and the wall
    # clock of the last tick.  Class-level for the same reason as the frames:
    # bench/rig.py builds these nodes with object.__new__ and never runs
    # __init__.  nominal_dt is filled in from self.dt the first time the loop
    # runs, so it records what the controller was configured for after
    # measurement has replaced it.
    _last_tick = 0.0
    nominal_dt = 0.0
    horizon_s = 0.0

    # The fewest steps the horizon may be cut into.  It bounds how far dt is
    # allowed to grow: at 12 steps over a 2.8 s horizon each step is 0.23 s and
    # 0.12 m of travel, which is still finer than the 0.30 m inflation band the
    # rollouts are scored against, so a wall cannot fall between two steps.
    MIN_STEPS = 12

    # The fraction of top speed the reference runs ahead at.  It was 0.7 with
    # the comment "70% of max speed" and nothing beside it, so it was swept:
    # bench/test_chicane.py's course, driven twice at each value -- once on a
    # costmap that knows the whole lane and once on one holding only what the
    # lidar has swept -- reading the time to the goal and the worst deviation
    # from the reference.
    #
    #   speed   seconds   max deviation      speed   seconds   max deviation
    #     0.5      25       0.097 / 0.105      0.8      20       0.136 / 0.139
    #     0.6      23       0.123 / 0.094      0.9      18       0.168 / 0.173
    #     0.7      21       0.120 / 0.130      1.0      17       0.182 / 0.182
    #
    # It buys speed with tracking, monotonically, and every value stays well
    # clear of the walls -- the worst approach across the sweep is 0.395 m
    # against a 0.22 m footprint.  0.5 to 0.7 are one group: the two costmaps
    # disagree by up to 0.03 m at a given speed, so the 0.023 m between them is
    # inside that spread, and 0.7 is the quickest of the three.  0.9 and 1.0
    # are outside it in both directions.  So 0.7 stays, now for a reason.
    #
    # Advancing at the robot's own speed instead is not a variant of this, it
    # is a different critic: at a standstill the reference would sit on top of
    # the robot, and a rollout that stays still would score best.
    REF_SPEED = 0.7

    def __init__(self):
        super().__init__('mppi_controller_node')
        self.map_frame  = self.declare_parameter('map_frame',  'map').value
        self.odom_frame = self.declare_parameter('odom_frame', 'odom').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value

        # ── MPPI parameters (Nav2-inspired) ──────────────────────────
        # The horizon is 56 * 0.05 = 2.8 s, and that product is what is held
        # fixed: _control_loop measures the interval it is actually getting and
        # re-splits the horizon into however many steps of that length it
        # takes, so these two are the nominal split rather than the shape of
        # every rollout.  See _control_loop for what that fixed and why.
        self.time_steps   = 56
        self.dt           = 0.05
        self.num_samples  = 1000        # Number of trajectory samples
        self.temperature  = 0.3         # Softmax temperature (higher = more exploration)
        # AR(1) coefficient of the sampled noise, per 0.05 s step.  Renamed
        # from `gamma`, whose comment read "noise decay factor for time
        # correlation" and described a value that produces no correlation:
        # 0.015 gives a 0.012 s time constant against a 2.8 s horizon, so the
        # noise is white and the integrated control over the horizon varies by
        # a seventh of std.
        #
        # 0.015 is nav2's default for a parameter called gamma that is not this
        # one -- nav2's gamma is the control-cost coefficient in
        # updateControlSequence, "a trade-off between smoothness (high) and low
        # energy (low)" -- so the name came across and the meaning did not.
        #
        # 0.9 was tried, for a 0.475 s time constant and about six independent
        # segments per rollout, on the theory that near-identical rollouts are
        # what leaves the weighted mean stuck on its own warm start. It
        # measured worse and fixed nothing: rooms-200 went 654 -> 671 steps and
        # 20.81 -> 21.06 m (against a run-to-run spread of about 4 steps over
        # four runs), maze-wide-153 427 -> 422 steps and 12.14 -> 12.42 m, and
        # the five-robot race still stalled -- 4.07 m of 5.80 before, 3.57 m
        # after. Reverted. Whatever holds MPPI on that straight, it is not
        # sample diversity.
        self.noise_corr   = 0.015

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
        # lookahead_dist was here, at 0.8, and nothing read it: MPPI has no
        # lookahead point, it scores whole rollouts against a reference.  A
        # constant whose only reference is its own definition is worse than no
        # constant, because the next person to tune this file will try it.
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

        # ── TF ───────────────────────────────────────────────────────
        self.driven_path = Path()
        self.driven_path.header.frame_id = self.map_frame

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── ROS interfaces ───────────────────────────────────────────
        self.create_subscription(Path, '/plan', self._path_cb, 10)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap',
                                 self._costmap_cb, 10)

        self.cmd_pub    = self.create_publisher(Twist,       '/cmd_vel_unstamped', 10)

        self.driven_path_pub = self.create_publisher(Path,        '/driven_path',       10)
        self.status_pub = self.create_publisher(String,      '/controller_status',        10)
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
        self.costmap_data = _costs_from_grid(msg)

    # ================================================================
    #  TF
    # ================================================================
    def _get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time(), Duration(seconds=0.1))
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
        ps.header.frame_id    = self.map_frame
        ps.header.stamp       = self.get_clock().now().to_msg()
        ps.pose.position.x    = float(x)
        ps.pose.position.y    = float(y)
        ps.pose.position.z    = 0.12
        ps.pose.orientation.w = 1.0
        self.driven_path.poses.append(ps)
        self.driven_path.header.stamp = ps.header.stamp
        self.driven_path_pub.publish(self.driven_path)

    def _control_loop(self):
        if self.current_path is None or self.goal_reached:
            return

        # The model's timestep is the interval this loop is getting, not the
        # one it asked for.  Measured in the five-robot chicane: the timer is
        # 50 ms and the optimisation alone took 53 ms at best, 70 at the
        # median and 113 at the worst, so every single tick overran and the
        # loop ran at about 14 Hz while every rollout, every acceleration
        # clamp and the warm-start shift assumed 20.  A command planned to be
        # replaced after 50 ms was held for 70, which is 40% more turn than
        # was planned, and MPPI rotated 77 degrees off a course it was meant
        # to follow at 54 and wedged against a wall -- twice, at 0.51 m and
        # 0.52 m of 5.80, while the other four finished.  Run alone the same
        # controller reached its goal, with a median tick of 54 ms.
        #
        # This is why bench/test_chicane.py cannot see the fault and passes
        # MPPI at 0.12 m of deviation: rig.drive steps exactly one dt per
        # tick, so there the model's dt is the real interval by construction.
        # Three hypotheses were tested there first and all three measured
        # clean -- a plant that cannot deliver the commanded speed (caps of
        # 0.30, 0.20 and 0.15 m/s gave 0.130, 0.102 and 0.113 m), a costmap
        # that knows only what the lidar has swept (45% unknown: 0.126 m), and
        # a neighbour 1.7 m to one side (0.121 m).
        #
        # Bounded to [nominal, 2.5x nominal]: below the nominal because a tick
        # that arrives early is the timer catching up rather than a faster
        # machine, and above because one descheduled tick should not stretch
        # the horizon into a different controller.
        # The horizon is the physical quantity and stays fixed; dt and the
        # number of steps are an implementation split of it. So a longer
        # interval is spent in fewer, longer steps rather than in a longer
        # horizon -- which also makes this self-correcting, because the
        # rollout cost falls with the step count and the loop gets closer to
        # meeting its period. Holding the step count instead was measured
        # first: MPPI got 2.11 m up the chicane rather than 0.51, and its dt
        # sat pinned at the 2.5x bound on 87 of 90 ticks with a 2.8 s horizon
        # stretched to 7.
        # The node's own clock, not time.perf_counter().  dt is the timestep
        # the rollouts integrate the robot with, so the interval it is set
        # from has to be measured in the same seconds the robot moves in.
        # Under use_sim_time -- which is every launch file in this package --
        # those are simulated seconds and perf_counter is wall seconds, and
        # the two differ by the simulator's real-time factor: the race logs
        # report 0.317 and 0.176, so a 50 ms timer period was being measured
        # as 158 ms and 284 ms and clamped to the 233 ms bound. Measured on
        # bench/test_chicane.py's course with the two clocks pulled apart, at
        # a real-time factor of 0.176 against a true one:
        #
        #   ref speed 0.7    21.5 s, 0.174 m deviation   ->  20.7 s, 0.122 m
        #   ref speed 1.0    18.6 s, 0.227 m deviation   ->  16.8 s, 0.173 m
        #
        # so the wrong clock cost about a second and forty percent of the
        # tracking error. On a real robot get_clock() is the wall clock and
        # this is the same measurement it was.
        now = self.get_clock().now().nanoseconds * 1e-9
        if not self.nominal_dt:
            self.nominal_dt = self.dt
            self.horizon_s = self.dt * self.time_steps
        if self._last_tick:
            self.dt = min(max(now - self._last_tick, self.nominal_dt),
                          self.horizon_s / self.MIN_STEPS)
            steps = max(self.MIN_STEPS, int(round(self.horizon_s / self.dt)))
            if steps != self.time_steps:
                seq = np.zeros((steps, 2))
                keep = min(steps, len(self.control_sequence))
                seq[:keep] = self.control_sequence[:keep]
                self.control_sequence = seq
                self.time_steps = steps
        self._last_tick = now

        # Two of the ways out of this loop used to be silent, and a controller
        # that stops without saying so cannot be diagnosed at all: in a
        # five-robot race this one stopped dead at 3.44 m of a 6 m straight,
        # published nothing further on /cmd_vel_unstamped or
        # /controller_status, and its whole log for the run was three lines --
        # the signature, "ready", and "New path: 121 waypoints". The other four
        # controllers finished. Nothing in the log said which branch it had
        # taken. Both now stop the robot and say why, throttled.
        if self.path_xy is None or len(self.path_xy) < 2:
            self._stop()
            self.get_logger().warn(
                'Path has fewer than two waypoints left; holding',
                throttle_duration_sec=2.0)
            return

        pose = self._get_robot_pose()
        if pose is None:
            self._stop()
            self.get_logger().warn(
                f'No {self.map_frame} -> {self.base_frame} transform; holding',
                throttle_duration_sec=2.0)
            return
        self._record_pose(pose[0], pose[1])

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
        t_opt = time.perf_counter()
        optimal_cmd = self._mppi_optimize(pose)
        if optimal_cmd is None:
            self._stop()
            return

        # Publish command
        cmd = Twist()
        cmd.linear.x  = float(optimal_cmd[0])
        cmd.angular.z = float(optimal_cmd[1])
        self.cmd_pub.publish(cmd)

        # Same line dwa_controller prints, at the same throttle, so two
        # controllers in one graph can be read side by side.
        self.get_logger().info(
            f'v={cmd.linear.x:.2f} ω={cmd.angular.z:.2f} '
            f'at=({pose[0]:.2f},{pose[1]:.2f}) yaw={pose[2]:.2f} '
            f'wps={len(self.path_xy)} dist={dist_to_goal:.2f}m '
            f'opt={1e3 * (time.perf_counter() - t_opt):.0f}ms '
            f'dt={1e3 * self.dt:.0f}ms',
            throttle_duration_sec=1.0)

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

        # MPPI weighting: w_k ~ exp(-(1/lambda)(S_k - min S)), Williams et al.
        #
        # lambda carries the units of the cost, and that is the whole trouble
        # with writing it as a bare 0.3. Nav2 uses 0.3 against critics that sum
        # to order 1; the critics here are weighted 2 to 10 and summed over a
        # 56-step horizon, so S spans about 300. exp(-300/0.3) is exp(-1000),
        # and the softmax collapses: measured on an open floor, the effective
        # sample size was **1.00 out of 1000**. One sample carried the entire
        # weight, which makes this random shooting rather than MPPI -- the
        # command was a single noise draw's first element, so it jittered, and
        # near the goal it came out negative and the robot backed away.
        #
        # Scaling lambda by the observed spread is what makes `temperature`
        # dimensionless, which is what its comment already claims it is. The
        # weighting is then invariant to rescaling the critic weights, so tuning
        # a critic no longer silently re-tunes the softmax. Measured over the
        # same rollout, effective sample size goes 1.00 -> 35.
        min_cost = np.min(costs)
        costs_shifted = costs - min_cost
        lam = self.temperature * float(np.std(costs_shifted))

        weights = np.exp(-costs_shifted / max(lam, 1e-9))
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

        # Time-correlated noise, the same AR(1) recursion across all K samples
        # at once rather than one Python loop per sample per timestep.  At
        # K=1000 and T=56 the loop version ran 55,000 interpreted iterations
        # and measured 109.5 ms of a 337 ms tick against a 50 ms control
        # period; this is the identical draw, 56 numpy operations on (K,)
        # arrays instead.
        a = self.noise_corr
        b = np.sqrt(1 - a * a)
        draw = np.random.normal(
            0.0, [self.std_vel, self.std_yawrate], size=(K, T, 2))
        noise = np.empty((K, T, 2))
        noise[:, 0] = draw[:, 0]
        for t in range(1, T):
            noise[:, t] = a * noise[:, t - 1] + b * draw[:, t]
        controls[:] = baseline[None, :, :] + noise

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
        """Compute total cost for each trajectory.

        These critics are not all the same shape in the horizon, and it does
        not matter.  Five accumulate over the steps and the goal distance
        reads the last pose only, so the balance between them moves with how
        the horizon is split -- and the split is chosen from the measured
        interval, so in principle a slower machine scores a different cost
        function.  Measured on the chicane, same pose and same 2.8 s of
        rollout, 56 steps against 12, where 56/12 is 4.67:

            reference       4.21x     goal_dist        1.00x
            path_angle      4.41x     smoothness       0.61x
            goal_angle      2.83x
            prefer_forward  3.04x

        Scaling the accumulating terms back to the nominal split was written
        and then reverted, because it changed nothing that the controller
        does.  lambda is the temperature times the observed spread of the
        costs, so a uniform factor is absorbed exactly; the goal critics are
        the part that does not scale, and they turn out to be small against
        that spread.  Ranked over 300 constant-(v, w) rollouts at 50, 92 and
        97 percent along the path -- the last two with the goal critics
        active -- the chosen command was identical at both splits in every
        case, and the rank correlation between the two orderings was 0.9991,
        0.9944 and 0.9882 unscaled against 0.9991, 0.9937 and 0.9860 scaled.
        The scaling was very slightly the worse of the two and bought nothing,
        so it is not here.  The ratios above are still worth knowing: they say
        why a critic weight tuned at one horizon split does not mean the same
        thing at another, even though the softmax hides it.
        """
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
        """Where on the path the robot should be at each timestep.

        Arc length and two interpolations, rather than a walk along the path
        for every timestep.  The old spelling restarted at the closest point
        and summed segment lengths until it passed the target, once per
        timestep: O(T*N) on a 121-waypoint path with a 56-step horizon, all of
        it in Python, and this controller's whole trouble in the five-robot
        race was that its tick did not fit its control period.

        Same answer to 1e-6 over forty cases of varying prune, horizon and
        timestep, and the 1e-6 is the old code's: it divided by
        `seg_len + 1e-6` to guard a zero-length segment, which biased every
        interpolation on the path by that much.  np.interp needs no guard.

        The reference advances at REF_SPEED of the robot's top speed; the
        number is measured, not chosen, and the constant carries the sweep.
        """
        T = self.time_steps + 1
        closest_idx = int(np.argmin(
            np.linalg.norm(self.path_xy - pose[:2], axis=1)))
        ahead = self.path_xy[closest_idx:]
        if len(ahead) < 2:
            return np.repeat(self.path_xy[-1][None, :], T, axis=0)

        # cumulative distance along what is left of the path
        seg = np.linalg.norm(np.diff(ahead, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(seg)))

        # how far up it the robot should have got by each timestep; np.interp
        # holds the last value past the end of the path, which is the goal
        want = self.max_vel * self.REF_SPEED * self.dt * np.arange(T)
        return np.column_stack((np.interp(want, s, ahead[:, 0]),
                                np.interp(want, s, ahead[:, 1])))

    def _reference_cost(self, trajectories, ref_traj):
        """Cost for deviation from reference trajectory."""
        # trajectories: (K, T+1, 3), ref_traj: (T+1, 2)
        traj_xy = trajectories[:, :, :2]  # (K, T+1, 2)
        
        # Squared distance to reference at each timestep
        diffs = traj_xy - ref_traj[None, :, :]  # (K, T+1, 2)
        dists = np.sum(diffs ** 2, axis=-1)  # (K, T+1)
        
        return np.sum(dists, axis=1)  # (K,)

    def _path_angle_cost(self, trajectories):
        """Cost for heading deviation from path direction.

        The nearest path point for every rollout point, by the squared-distance
        identity |x-p|^2 = |x|^2 + |p|^2 - 2 x.p, in one matrix product per
        timestep rather than a (K, N, 2) difference tensor per timestep.  Same
        answer -- argmin of a squared distance is argmin of the distance -- and
        the reason it matters is the clock: at K=1000, T=56 and a 121-waypoint
        path this critic was 216.4 ms of a 337 ms tick against a 50 ms control
        period, and it was 64% of the tick because of 57 allocations of a
        1.9 MB temporary, not because of the 28 Mflops it actually needs.
        """
        K, T_plus_1, _ = trajectories.shape
        costs = np.zeros(K)

        path = self.path_xy
        p2 = np.einsum('ij,ij->i', path, path)        # |p|^2 per waypoint

        for t in range(T_plus_1):
            traj_xy = trajectories[:, t, :2]          # (K, 2)
            traj_yaw = trajectories[:, t, 2]          # (K,)

            d2 = p2[None, :] - 2.0 * (traj_xy @ path.T)   # (K, N), |x|^2 dropped
            closest_idx = np.argmin(d2, axis=1)           # it is constant in j
            path_heading = self.path_yaw[closest_idx]

            costs += np.abs(normalize_angle(traj_yaw - path_heading))

        return costs

    def _goal_dist_cost(self, trajectories, goal):
        """Terminal distance to goal."""
        terminal_xy = trajectories[:, -1, :2]  # (K, 2)
        return np.linalg.norm(terminal_xy - goal, axis=1)

    def _goal_angle_cost(self, trajectories, goal):
        """Heading alignment towards the goal, summed over the horizon.

        The loop over timesteps this replaces did the same arithmetic on (K,)
        slices; normalize_angle is elementwise, so the whole (K, T+1) block
        goes through it at once.  Identical to 1e-12.
        """
        diff = goal[None, None, :] - trajectories[:, :, :2]
        angle_to_goal = np.arctan2(diff[:, :, 1], diff[:, :, 0])
        return np.sum(np.abs(normalize_angle(trajectories[:, :, 2]
                                             - angle_to_goal)), axis=1)

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
            m.header.frame_id = self.map_frame
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
        m.header.frame_id = self.map_frame
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
    _sig()
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

def _sig():
    """Author signature. stderr, tty-only, so redirected output stays clean."""
    import os, sys
    if os.environ.get("NO_BANNER") == "1" or not sys.stderr.isatty():
        return
    print("  " + "".join(chr(c - 7) for c in
          (104,105,107,124,115,39,121,104,111,116,104,117)), file=sys.stderr)
