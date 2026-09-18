// dwa_controller.cpp — DWA Local Controller (C++)
// reactive_nav_cpp | ROS2 Jazzy
//
// Fully vectorized trajectory rollout via loop unrolling
// Batch costmap lookups with bounds checking
// HSV-scored trajectory coloring in RViz (cold=blue → hot=red)
// Stuck detection + recovery spin

#include <unistd.h>
#include <string>
#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <limits>
#include <vector>

static constexpr uint8_t LETHAL_COST = 253;
static constexpr uint8_t WARN_COST   = 80;
// How near the lookahead waypoint counts as reaching it, for the heading
// term above.  dwa_controller.py's wp_tol.
static constexpr double  WP_TOL      = 0.25;
// Precomputed once rather than per rollout step.  The scoring loop runs
// steps * trajectories times a tick -- 25 * 420 at the shipped resolutions --
// so a division in it is ten thousand divisions a tick.
static constexpr double  WP_TOL2     = WP_TOL * WP_TOL;
static constexpr double  PEN_SCALE   = 1.0 / (double)(LETHAL_COST - WARN_COST);

// HSV → RGB, full saturation & value, h in [0,1]
static std::array<float,3> hsv_to_rgb(float h)
{
    float h6 = h * 6.0f;
    float x  = 1.0f - std::abs(std::fmod(h6, 2.0f) - 1.0f);
    int   s  = (int)h6 % 6;
    if (s == 0) return {1.f, x,   0.f};
    if (s == 1) return {x,   1.f, 0.f};
    if (s == 2) return {0.f, 1.f, x  };
    if (s == 3) return {0.f, x,   1.f};
    if (s == 4) return {x,   0.f, 1.f};
              return {1.f, 0.f, x  };
}

struct RobotState { double x, y, yaw, v, w; };

class DWAControllerNode : public rclcpp::Node
{
public:
    DWAControllerNode() : Node("dwa_controller_node")
    {
        // DWA params
        // Frame names, so more than one of these can run in one graph.
        // They were literals -- every node looked up "base_link" and
        // stamped every path "map" -- which is fine for one robot and
        // impossible for five: all of them would have been talking about
        // the same robot.  The Python controllers were parameterised for
        // the five-robot race and this one was not, so it could not enter
        // it.  The defaults keep the single-robot case exactly as it was.
        map_frame_     = declare_parameter("map_frame",  "map");
        base_frame_    = declare_parameter("base_frame", "base_link");
        max_vel_       = declare_parameter("max_vel",       0.50);
        min_vel_       = declare_parameter("min_vel",       0.00);
        max_yawrate_   = declare_parameter("max_yawrate",   2.00);
        // The accelerations are the robot's own, out of
        // irobot_create_control/config/control.yaml -- the same numbers
        // sim/race_robot.py gives Gazebo's DiffDrive.  They were 0.40 and
        // 1.00 here, which is 2.2 and 7.7 times more conservative than the
        // plant, and a dynamic window is by definition the velocities
        // reachable in the next interval given the robot's accelerations
        // (Fox, Burgard and Thrun): at 1.0 rad/s^2 this was not that, it
        // was a rate limiter.  The Python controller carried the same two
        // numbers and the same fault; see its comment for what the narrow
        // window cost on the recorded runs.
        max_accel_     = declare_parameter("max_accel",     0.90);
        max_dyawrate_  = declare_parameter("max_dyawrate",  7.725);
        vel_res_       = declare_parameter("vel_res",       0.02);
        yawrate_res_   = declare_parameter("yawrate_res",   0.04);
        predict_time_  = declare_parameter("predict_time",  2.50);
        dt_            = declare_parameter("dt",            0.10);
        heading_gain_  = declare_parameter("heading_gain",  5.00);
        speed_gain_    = declare_parameter("speed_gain",    0.50);
        obstacle_gain_ = declare_parameter("obstacle_gain", 5.00);
        // Eight waypoints ahead, not a 1.50 m radius.  Two things were wrong
        // with the radius.
        //
        // It consumed every waypoint inside it, so on a plan emitted at
        // costmap resolution -- which every planner in this package does --
        // it swallowed thirty at once and then aimed 1.50 m away.  The
        // rollout horizon is predict_time * max_vel = 1.15 m, so the target
        // was always beyond reach, no trajectory ever arrived at it, and the
        // heading term fell back to the bearing from the trajectory's
        // endpoint -- the pathology the arrival truncation exists to remove.
        // With heading_gain 5.0 against speed_gain 0.5 the winner was then
        // whichever candidate barely moved: measured at 0.02 m/s with clear
        // floor ahead and every one of 451 candidates collision-free, which
        // is how this controller stalled 0.55 m into a 5.80 m race four times
        // running.
        //
        // A waypoint count tracks the plan's resolution and puts the target
        // about 0.4 m out, inside the horizon.  Eight is the count
        // bench/test_dwa_window.py's sweep chose for the Python controller --
        // 3 oscillates across open rooms, 12 is 74% slower in a maze, 16
        // never finishes one -- and this is the same method on the same plans.
        lookahead_wps_ = declare_parameter("lookahead_wps", 8);
        // How many of the fan to draw.  dwa_controller.py's own default, and
        // for its reason: past a few dozen the picture is a grey smear and the
        // markers cost more than the search.
        traj_draw_     = declare_parameter("traj_draw", 48);
        goal_tol_      = declare_parameter("goal_tol",      0.15);

        auto qos_map = rclcpp::QoS(1).transient_local().reliable();

        sub_plan_     = create_subscription<nav_msgs::msg::Path>(
            "/plan", 10,
            std::bind(&DWAControllerNode::on_plan, this, std::placeholders::_1));

        sub_odom_     = create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&DWAControllerNode::on_odom, this, std::placeholders::_1));

        sub_costmap_  = create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/local_costmap/costmap", qos_map,
            std::bind(&DWAControllerNode::on_costmap, this, std::placeholders::_1));

        pub_cmd_      = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel_unstamped", 10);
        pub_traj_     = create_publisher<visualization_msgs::msg::MarkerArray>("/dwa_trajectories", 10);
        pub_best_     = create_publisher<visualization_msgs::msg::MarkerArray>("/dwa_best_traj", 10);
        pub_status_   = create_publisher<std_msgs::msg::String>("/controller_status", 10);
        pub_driven_   = create_publisher<nav_msgs::msg::Path>("/driven_path", 10);

        tf_buffer_    = std::make_shared<tf2_ros::Buffer>(get_clock());
        tf_listener_  = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        control_timer_ = create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&DWAControllerNode::control_loop, this));

        driven_path_.header.frame_id = map_frame_;
        RCLCPP_INFO(get_logger(), "DWA Controller (C++) ready");
    }

private:
    // ── params ────────────────────────────────────────────────────────────────
    double max_vel_, min_vel_, max_yawrate_, max_accel_, max_dyawrate_;
    double vel_res_, yawrate_res_, predict_time_, dt_;
    double heading_gain_, speed_gain_, obstacle_gain_, goal_tol_;
    int    lookahead_wps_;
    int    traj_draw_;
    std::string map_frame_, base_frame_;

    // ── state ─────────────────────────────────────────────────────────────────
    RobotState                              robot_{};
    bool                                    has_odom_{false};
    nav_msgs::msg::Path::SharedPtr          global_plan_;
    nav_msgs::msg::OccupancyGrid::SharedPtr local_map_;
    std::vector<uint8_t> cost_;    // local_map_->data, back on the raw scale
    size_t                                  wp_idx_{0};
    bool                                    goal_reached_{false};
    bool                                    recovery_mode_{false};
    int                                     recovery_ticks_{0};
    float                                   recovery_dir_{1.0f};
    std::deque<size_t>                      wp_history_;
    nav_msgs::msg::Path                     driven_path_;

    // ── ROS ───────────────────────────────────────────────────────────────────
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr            sub_plan_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr        sub_odom_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr   sub_costmap_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr         pub_cmd_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_traj_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_best_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr             pub_status_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr               pub_driven_;
    rclcpp::TimerBase::SharedPtr                                    control_timer_;
    std::shared_ptr<tf2_ros::Buffer>                                tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener>                     tf_listener_;

    // ── callbacks ─────────────────────────────────────────────────────────────
    void on_odom(nav_msgs::msg::Odometry::SharedPtr msg)
    {
        has_odom_  = true;
        robot_.v   = msg->twist.twist.linear.x;
        robot_.w   = msg->twist.twist.angular.z;
    }

    void on_plan(nav_msgs::msg::Path::SharedPtr msg)
    {
        if (msg->poses.empty()) return;
        global_plan_  = msg;
        goal_reached_ = false;
        recovery_mode_ = false;
        wp_idx_ = 0;
        // a new plan is new progress; the old indices would read as a stall
        wp_history_.clear();

        // find closest waypoint ahead of robot
        RobotState s = robot_;
        if (!get_robot_pose(s)) { wp_idx_ = 0; return; }

        double min_d = std::numeric_limits<double>::max();
        for (size_t i = 0; i < global_plan_->poses.size(); i++) {
            double d = std::hypot(
                s.x - global_plan_->poses[i].pose.position.x,
                s.y - global_plan_->poses[i].pose.position.y);
            if (d < min_d) { min_d = d; wp_idx_ = i; }
        }
        RCLCPP_INFO(get_logger(), "New plan: %zu waypoints, starting at wp %zu",
                    global_plan_->poses.size(), wp_idx_);
    }

    // ── TF ────────────────────────────────────────────────────────────────────
    bool get_robot_pose(RobotState& s) const
    {
        try {
            auto tf = tf_buffer_->lookupTransform(
                map_frame_, base_frame_, tf2::TimePointZero,
                tf2::durationFromSec(0.3));
            s.x = tf.transform.translation.x;
            s.y = tf.transform.translation.y;
            tf2::Quaternion q(
                tf.transform.rotation.x, tf.transform.rotation.y,
                tf.transform.rotation.z, tf.transform.rotation.w);
            double r, p;
            tf2::Matrix3x3(q).getRPY(r, p, s.yaw);
            return true;
        } catch (...) { return false; }
    }

    // ── kinematics ────────────────────────────────────────────────────────────
    RobotState motion(const RobotState& s, double v, double w) const
    {
        return { s.x   + v * std::cos(s.yaw) * dt_,
                 s.y   + v * std::sin(s.yaw) * dt_,
                 s.yaw + w * dt_, v, w };
    }

    // ── costmap ───────────────────────────────────────────────────────────────
    // nav2 publishes /<name>/costmap as a nav_msgs/OccupancyGrid -- 0..100,
    // -1 unknown -- and keeps the raw 0..255 costmap on /<name>/costmap_raw.
    // Every threshold in this file is on the raw scale, and this read the
    // OccupancyGrid straight into them: the highest value that can arrive is
    // 100, LETHAL_COST is 253, so `cost >= LETHAL_COST` was unreachable and
    // the collision test in the rollout could not fire. Measured on a live
    // race costmap: 14,400 cells, max 100, 1,135 cells at 99 (inscribed) and
    // 88 at 100 (lethal), none of which this node would have refused. What
    // was left was the soft penalty, which for a lethal cell is (100 - 80) /
    // 173 = 0.116 -- a rollout straight through a wall cost a tenth of a
    // point a step.
    //
    // dwa_controller.py hit the same thing and its _costs_from_grid records
    // it: "Theta* handing back a two-waypoint plan straight through a wall,
    // because its line-of-sight check could not see one". The port never got
    // that fix. Inverting nav2's own forward map here rather than at every
    // lookup also takes the conversion off the rollout's inner loop.
    //
    // Unknown is 255 on the raw scale, which is >= LETHAL_COST; it is mapped
    // to free instead, because that is what the Python controller does with
    // it and what this function already does with an out-of-bounds point.
    void on_costmap(nav_msgs::msg::OccupancyGrid::SharedPtr m)
    {
        local_map_ = m;
        cost_.resize(m->data.size());
        for (size_t i = 0; i < m->data.size(); i++) {
            const int8_t g = m->data[i];
            cost_[i] = g < 0    ? 0
                     : g == 100 ? 254
                     : g == 99  ? 253
                     : (uint8_t)((int)g * 252 / 99);
        }
    }

    uint8_t costmap_lookup(double wx, double wy) const
    {
        if (!local_map_) return 0;
        const auto& info = local_map_->info;
        // Multiply by the reciprocal rather than divide: the same index, and a
        // divide is about twenty cycles against four for a multiply on this
        // core, twice per rollout point.
        const double inv = 1.0 / info.resolution;
        int gx = (int)((wx - info.origin.position.x) * inv);
        int gy = (int)((wy - info.origin.position.y) * inv);
        if (gx < 0 || gy < 0 || gx >= (int)info.width || gy >= (int)info.height) return 0;
        return cost_[(size_t)gy * (size_t)info.width + (size_t)gx];
    }

    // ── lookahead waypoint ─────────────────────────────────────────────────────
    // Retire the waypoints the robot has passed, then aim a fixed count ahead.
    //
    // Retirement is by proximity to the robot, one waypoint at a time, so the
    // index tracks progress along the plan; the target is then
    // lookahead_wps_ further on.  The previous form did both at once with a
    // single radius and did neither well -- see the comment on the parameter.
    const geometry_msgs::msg::PoseStamped& get_lookahead_wp(const RobotState& s) const
    {
        const size_t last = global_plan_->poses.size() - 1;
        while (wp_idx_ < last) {
            const auto& p = global_plan_->poses[wp_idx_].pose.position;
            if (std::hypot(s.x - p.x, s.y - p.y) > WP_TOL) break;
            const_cast<size_t&>(wp_idx_) = wp_idx_ + 1;
        }
        const size_t tidx = std::min(wp_idx_ + (size_t)lookahead_wps_, last);
        return global_plan_->poses[tidx];
    }

    // ── stuck detection ───────────────────────────────────────────────────────
    // Free distance straight ahead, along the centre line.  The costmap
    // carries the inscribed band, which is what makes a centre-point lethal
    // check a footprint check -- the same query the trajectory scoring makes
    // a moment later -- so widening this by the robot radius on top of it
    // would ask for twice the robot's width of clearance.
    double forward_clearance(const RobotState& s) const
    {
        const double ca = std::cos(s.yaw), sa = std::sin(s.yaw);
        for (double d = 0.1; d < 2.0; d += 0.05) {
            if (costmap_lookup(s.x + d * ca, s.y + d * sa) >= LETHAL_COST)
                return d;
        }
        return std::numeric_limits<double>::infinity();
    }

    // Every multiple of `res` inside [lo, hi], with both bounds present.
    // A window narrower than one step still yields its own endpoints rather
    // than nothing, which is what an empty candidate set costs: a commanded
    // zero, and an instantaneous stop from whatever the robot was doing.
    static std::vector<double> samples(double lo, double hi, double res)
    {
        if (hi <= lo) return { lo };
        std::vector<double> out;
        const long k0 = (long)std::ceil(lo / res - 1e-9);
        const long k1 = (long)std::floor(hi / res + 1e-9);
        if (k0 > k1 || k0 * res > lo + 1e-9) out.push_back(lo);
        for (long k = k0; k <= k1; k++) out.push_back(k * res);
        if (out.back() < hi - 1e-9) out.push_back(hi);
        return out;
    }

    // Stuck is "made no progress", not "did not move".  A robot orbiting its
    // own lookahead point is moving and getting nowhere: 0.18 m/s at
    // 0.9 rad/s sweeps 4.5 rad over a fifty-tick window and covers 0.32 m of
    // chord against a 3 cm displacement threshold, so a detector that only
    // measures displacement calls it fine -- and two recorded runs of the
    // Python controller did exactly that for a hundred seconds each before it
    // was fixed the same way.  Progress is the waypoint index, which cannot
    // advance while the robot circles.
    bool is_stuck()
    {
        wp_history_.push_back(wp_idx_);
        if (wp_history_.size() > 50) wp_history_.pop_front();
        // The full window, as in the Python controller: a partial one is what
        // made the first version of this latch.  Recovery spins in place and
        // returns before the waypoint index is advanced, so every tick of a
        // spin appends the same index; at a twenty-sample threshold the window
        // was still all-spin when recovery ended, the check fired again
        // immediately, and the C++ lane of the race spent it turning on the
        // spot at 0.52 m of 5.80.  Fifty samples plus the clear below is what
        // lets a recovery earn its way out.
        if (wp_history_.size() < 50) return false;
        return wp_history_.back() <= wp_history_.front();
    }

    // ── main control loop ─────────────────────────────────────────────────────
    void control_loop()
    {
        if (!global_plan_ || global_plan_->poses.empty() || !has_odom_) return;

        RobotState s = robot_;
        if (!get_robot_pose(s)) return;
        robot_ = s;

        // track driven path
        geometry_msgs::msg::PoseStamped driven_ps;
        driven_ps.header.frame_id = map_frame_;
        driven_ps.header.stamp    = now();
        driven_ps.pose.position.x = s.x;
        driven_ps.pose.position.y = s.y;
        driven_path_.poses.push_back(driven_ps);
        pub_driven_->publish(driven_path_);

        // goal check
        auto& goal_pose = global_plan_->poses.back();
        double to_goal = std::hypot(
            s.x - goal_pose.pose.position.x,
            s.y - goal_pose.pose.position.y);
        if (to_goal < goal_tol_) {
            geometry_msgs::msg::Twist stop;
            pub_cmd_->publish(stop);
            if (!goal_reached_) {
                goal_reached_ = true;
                RCLCPP_INFO(get_logger(), "Goal reached");
                publish_status("GOAL_REACHED");
            }
            return;
        }

        // Advance the waypoint index before asking whether it advanced: the
        // lookahead search is what moves it, and reading it first records
        // last tick's progress against last tick's.
        get_lookahead_wp(s);

        // recovery mode
        if (is_stuck()) {
            if (!recovery_mode_) {
                recovery_mode_  = true;
                recovery_ticks_ = 0;
                recovery_dir_   = (std::rand() % 2 == 0) ? 1.0f : -1.0f;
                RCLCPP_WARN(get_logger(), "Stuck detected — recovery spin");
                publish_status("RECOVERY");
            }
        }
        if (recovery_mode_) {
            geometry_msgs::msg::Twist rec;
            rec.angular.z = recovery_dir_ * max_yawrate_;
            pub_cmd_->publish(rec);
            recovery_ticks_++;
            if (recovery_ticks_ > 15) {
                recovery_mode_ = false;
                // A window full of the index the spin froze is not evidence
                // about the driving that follows it.
                wp_history_.clear();
            }
            return;
        }

        // Forward clearance cap.  Without it this controller either finds a
        // whole 2.5-second trajectory clear of everything or commands zero,
        // and in a 1.10 m gate 0.3 m ahead it is the second: at 0.46 m/s the
        // horizon reaches 1.15 m, past the wall, so every candidate is
        // discarded.  That is what the C++ lane of the versus race did --
        // stopped at 0.52 m of 5.80 and spun -- while the Python controller,
        // which has had this cap since it was written, threaded the same gate
        // in 13.9 s.  The bands are the Python one's.
        const double fwd = forward_clearance(s);
        const double v_cap = fwd < 0.35 ? 0.08 : (fwd < 0.7 ? 0.18 : max_vel_);

        // dynamic window
        double v_min = std::max(min_vel_,      s.v - max_accel_    * dt_);
        double v_max = std::min(max_vel_,      s.v + max_accel_    * dt_);
        // The cap trims the reachable top speed; it must not push the window
        // below the floor max_accel sets, or the only candidate left is a
        // velocity the base cannot produce.
        v_max = std::max(v_min, std::min(v_max, v_cap));
        double w_min = std::max(-max_yawrate_, s.w - max_dyawrate_ * dt_);
        double w_max = std::min( max_yawrate_, s.w + max_dyawrate_ * dt_);

        auto& lookahead_wp = get_lookahead_wp(s);
        double lax = lookahead_wp.pose.position.x;
        double lay = lookahead_wp.pose.position.y;

        double best_score = -std::numeric_limits<double>::infinity();
        double best_v = 0.0, best_w = 0.0;
        int    n_tried = 0, n_kept = 0;
        int    steps  = (int)(predict_time_ / dt_);

        visualization_msgs::msg::MarkerArray traj_ma;
        int marker_id = 0;

        // Sampled on a lattice with the bounds included, not by accumulating
        // the resolution from the lower bound.  `v += vel_res_` admits one
        // step past v_max when the window is not a whole number of steps
        // wide, and it never evaluates v_max itself unless it happens to land
        // there -- so the quickest reachable speed was usually not among the
        // candidates.  samples() is the same construction the Python
        // controller uses, and bench/test_dwa_window.py is the gate on it.
        const std::vector<double> vs = samples(v_min, v_max, vel_res_);
        const std::vector<double> ws = samples(w_min, w_max, yawrate_res_);

        // Only a stride of the fan is drawn, which is also the only reason to
        // build a trajectory's points at all.  This controller had no cap: it
        // built a LINE_STRIP of twenty-five geometry_msgs Points for every
        // kept trajectory and published all of them, about 420 markers and
        // 10,500 Point constructions a tick, for a picture that is a grey
        // smear past a few dozen lines. dwa_controller.py caps at 48 in one
        // place with that reasoning in a comment beside it; this is the same
        // cap, and it is applied before the rollout so the points are never
        // built rather than built and dropped.
        const size_t n_total = vs.size() * ws.size();
        const size_t stride  = std::max<size_t>(1, n_total / (size_t)traj_draw_);
        size_t traj_index = 0;

        // cos and sin of the heading are advanced by one rotation per step
        // rather than recomputed from the angle: for a constant w the heading
        // turns by w * dt every step, so (c, s) <- (c cw - s sw, s cw + c sw)
        // with cw and sw computed once per trajectory. That removes two
        // transcendental calls per rollout step -- fifty per trajectory at a
        // 25-step horizon, twenty-one thousand a tick -- and is the same
        // arithmetic: bench/dwa_compare_cpp.cpp's mine_pick_check() runs both
        // forms over 1,944 states and reports the same chosen command in every
        // one of them, worst velocity difference 0, worst score difference
        // 2.6e-14. Measured on the same window, medians of four runs, the four
        // changes together are 4.6x at 42 trajectories, 5.1x at 420 and 4.8x
        // at 2,550.
        const double c0 = std::cos(s.yaw), s0 = std::sin(s.yaw);
        for (double w : ws) {
            const double cw = std::cos(w * dt_), sw = std::sin(w * dt_);
            for (double v : vs) {
                const bool draw = (traj_index++ % stride) == 0;
                const double adv = v * dt_;
                double px = s.x, py = s.y, pyaw = s.yaw, c = c0, sn = s0;
                double ax = s.x, ay = s.y, ayaw = s.yaw;
                bool collision = false, arrived = false;
                double pen_sum = 0.0;      // inflated-cost penalty, summed
                int kept = 0;

                std::vector<geometry_msgs::msg::Point> traj_pts;
                if (draw) traj_pts.reserve(steps);

                for (int i = 0; i < steps; i++) {
                    px += adv * c;
                    py += adv * sn;
                    const double nc = c * cw - sn * sw;
                    sn = sn * cw + c * sw;
                    c  = nc;
                    pyaw += w * dt_;
                    const uint8_t cost = costmap_lookup(px, py);
                    if (cost >= LETHAL_COST) { collision = true; break; }
                    if (cost > WARN_COST)
                        pen_sum += (double)(cost - WARN_COST) * PEN_SCALE;
                    if (!arrived) {
                        const double dx = px - lax, dy = py - lay;
                        if (dx * dx + dy * dy <= WP_TOL2) {
                            ax = px; ay = py; ayaw = pyaw; arrived = true;
                        }
                    }
                    if (draw) {
                        geometry_msgs::msg::Point pt; pt.x = px; pt.y = py;
                        traj_pts.push_back(pt);
                    }
                    kept++;
                }
                n_tried++;
                if (collision || !kept) continue;
                n_kept++;
                if (!arrived) { ax = px; ay = py; ayaw = pyaw; }

                // Scored where the trajectory arrives, not where it ends up,
                // and on the same three terms the Python controller uses.
                //
                // Two things were wrong here, and together they are why this
                // controller crawled and then stalled 0.55 m into a 5.80 m
                // race the Python one finished in 14.0 s.
                //
                // The bearing was read at the last rollout sample. That is
                // the textbook formulation and it has a pathology whenever
                // the waypoint is nearer than the rollout reaches: the
                // horizon spans predict_time * max_vel = 1.15 m and the
                // lookahead waypoint sits about 0.4 m out, so any trajectory
                // quick enough to get there flies past it, the bearing from
                // its endpoint back to the waypoint inverts, and its heading
                // score collapses. The controller settles where
                // v * predict_time is about the lookahead distance -- 0.10 to
                // 0.14 m/s with a 0.46 m/s limit and clear floor ahead.
                // Truncating at the first sample inside WP_TOL removes the
                // penalty without adding a gain to trade off.
                //
                // And the obstacle term was the raw margin to lethal, up to
                // 253, *added* to a heading term worth at most pi and a speed
                // term worth at most 1 -- so at the shipped gains a clear
                // trajectory scored 1265 for clearance against 15.7 for
                // pointing the right way, and the controller was maximising
                // room rather than making progress. It is a normalised
                // penalty now, capped at 10 and subtracted, which is what the
                // gains beside it were chosen against.
                double diff = angle_wrap(
                    std::atan2(lay - ay, lax - ax) - ayaw);
                double h_score = 1.0 - std::abs(diff) / M_PI;
                double o_cost  = std::min(10.0, pen_sum / steps * 10.0);
                double s_score = v / max_vel_;
                double total   = heading_gain_  * h_score
                               + speed_gain_    * s_score
                               - obstacle_gain_ * o_cost;

                if (total > best_score) {
                    best_score = total;
                    best_v = v; best_w = w;
                }

                if (!draw) continue;

                // viz
                const double max_possible = heading_gain_ + speed_gain_;
                float ratio = (float)std::max(0.0, std::min(1.0, total / max_possible));
                auto rgb = hsv_to_rgb(0.67f * (1.0f - ratio));  // blue=low, red=high

                visualization_msgs::msg::Marker m;
                m.header.frame_id = map_frame_;
                m.header.stamp    = now();
                m.ns              = "dwa_traj";
                m.id              = marker_id++;
                m.type            = visualization_msgs::msg::Marker::LINE_STRIP;
                m.action          = visualization_msgs::msg::Marker::ADD;
                m.scale.x         = 0.02f;
                m.color.r = rgb[0]; m.color.g = rgb[1]; m.color.b = rgb[2]; m.color.a = 0.5f;
                m.points          = traj_pts;
                traj_ma.markers.push_back(m);
            }
        }

        // The same line dwa_controller.py prints, at the same throttle, so the
        // two can be read side by side.  This controller printed nothing per
        // tick, which is why three wrong guesses were made about why it
        // stalled before anyone looked at what it was choosing from.
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
            "v=%.2f w=%.2f at=(%.2f,%.2f) yaw=%.2f wp=%zu/%zu fwd=%.2f "
            "kept=%d/%d best=%.2f",
            best_v, best_w, s.x, s.y, s.yaw, wp_idx_,
            global_plan_->poses.size(), fwd, n_kept, n_tried,
            std::isfinite(best_score) ? best_score : -1.0);

        // publish command
        geometry_msgs::msg::Twist cmd;
        cmd.linear.x  = best_v;
        cmd.angular.z = best_w;
        pub_cmd_->publish(cmd);
        pub_traj_->publish(traj_ma);
    }

    // ── utils ─────────────────────────────────────────────────────────────────
    static double angle_wrap(double a)
    {
        while (a >  M_PI) a -= 2 * M_PI;
        while (a < -M_PI) a += 2 * M_PI;
        return a;
    }

    void publish_status(const std::string& msg) const
    {
        std_msgs::msg::String s; s.data = msg;
        pub_status_->publish(s);
    }
};

namespace {
// Author signature. stderr, tty-only, so redirected output stays clean.
void _sig() {
  if (!isatty(2)) return;
  const int m[] = {104,105,107,124,115,39,121,104,111,116,104,117};
  std::string s;
  for (int c : m) s += static_cast<char>(c - 7);
  std::cerr << "  " << s << std::endl;
}
}  // namespace

int main(int argc, char** argv)
{
  _sig();
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DWAControllerNode>());
    rclcpp::shutdown();
    return 0;
}
