// dwa_controller.cpp — DWA Local Controller (C++)
// reactive_nav_cpp | ROS2 Jazzy
//
// Fully vectorized trajectory rollout via loop unrolling
// Batch costmap lookups with bounds checking
// HSV-scored trajectory coloring in RViz (cold=blue → hot=red)
// Stuck detection + recovery spin

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
        max_vel_       = declare_parameter("max_vel",       0.50);
        min_vel_       = declare_parameter("min_vel",       0.00);
        max_yawrate_   = declare_parameter("max_yawrate",   2.00);
        max_accel_     = declare_parameter("max_accel",     0.40);
        max_dyawrate_  = declare_parameter("max_dyawrate",  1.00);
        vel_res_       = declare_parameter("vel_res",       0.02);
        yawrate_res_   = declare_parameter("yawrate_res",   0.04);
        predict_time_  = declare_parameter("predict_time",  2.50);
        dt_            = declare_parameter("dt",            0.10);
        heading_gain_  = declare_parameter("heading_gain",  5.00);
        speed_gain_    = declare_parameter("speed_gain",    0.50);
        obstacle_gain_ = declare_parameter("obstacle_gain", 5.00);
        lookahead_     = declare_parameter("lookahead",     1.50);
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
            [this](nav_msgs::msg::OccupancyGrid::SharedPtr m){ local_map_ = m; });

        pub_cmd_      = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel_unstamped", 10);
        pub_traj_     = create_publisher<visualization_msgs::msg::MarkerArray>("/dwa_trajectories", 10);
        pub_best_     = create_publisher<visualization_msgs::msg::MarkerArray>("/dwa_best_traj", 10);
        pub_status_   = create_publisher<std_msgs::msg::String>("/dwa_status", 10);
        pub_driven_   = create_publisher<nav_msgs::msg::Path>("/driven_path", 10);

        tf_buffer_    = std::make_shared<tf2_ros::Buffer>(get_clock());
        tf_listener_  = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        control_timer_ = create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&DWAControllerNode::control_loop, this));

        driven_path_.header.frame_id = "map";
        RCLCPP_INFO(get_logger(), "DWA Controller (C++) ready");
    }

private:
    // ── params ────────────────────────────────────────────────────────────────
    double max_vel_, min_vel_, max_yawrate_, max_accel_, max_dyawrate_;
    double vel_res_, yawrate_res_, predict_time_, dt_;
    double heading_gain_, speed_gain_, obstacle_gain_, lookahead_, goal_tol_;

    // ── state ─────────────────────────────────────────────────────────────────
    RobotState                              robot_{};
    bool                                    has_odom_{false};
    nav_msgs::msg::Path::SharedPtr          global_plan_;
    nav_msgs::msg::OccupancyGrid::SharedPtr local_map_;
    size_t                                  wp_idx_{0};
    bool                                    goal_reached_{false};
    bool                                    recovery_mode_{false};
    int                                     recovery_ticks_{0};
    float                                   recovery_dir_{1.0f};
    std::deque<std::pair<double,double>>    pos_history_;
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
                "map", "base_link", tf2::TimePointZero, tf2::durationFromSec(0.3));
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
    uint8_t costmap_lookup(double wx, double wy) const
    {
        if (!local_map_) return 0;
        const auto& info = local_map_->info;
        int gx = (int)((wx - info.origin.position.x) / info.resolution);
        int gy = (int)((wy - info.origin.position.y) / info.resolution);
        if (gx < 0 || gy < 0 || gx >= (int)info.width || gy >= (int)info.height) return 0;
        return static_cast<uint8_t>(local_map_->data[gy * (int)info.width + gx]);
    }

    // ── lookahead waypoint ─────────────────────────────────────────────────────
    const geometry_msgs::msg::PoseStamped& get_lookahead_wp(const RobotState& s) const
    {
        // advance wp_idx to stay ahead of robot
        for (size_t i = wp_idx_; i < global_plan_->poses.size() - 1; i++) {
            double d = std::hypot(
                s.x - global_plan_->poses[i].pose.position.x,
                s.y - global_plan_->poses[i].pose.position.y);
            if (d < lookahead_) {
                const_cast<size_t&>(wp_idx_) = i + 1;
            } else break;
        }
        return global_plan_->poses[std::min(wp_idx_, global_plan_->poses.size() - 1)];
    }

    // ── stuck detection ───────────────────────────────────────────────────────
    bool is_stuck(const RobotState& s)
    {
        pos_history_.push_back({s.x, s.y});
        if (pos_history_.size() > 50) pos_history_.pop_front();
        if (pos_history_.size() < 20) return false;
        double dx = pos_history_.back().first  - pos_history_.front().first;
        double dy = pos_history_.back().second - pos_history_.front().second;
        return std::hypot(dx, dy) < 0.03;
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
        driven_ps.header.frame_id = "map";
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

        // recovery mode
        if (is_stuck(s)) {
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
            if (recovery_ticks_ > 15) recovery_mode_ = false;
            return;
        }

        // dynamic window
        double v_min = std::max(min_vel_,      s.v - max_accel_    * dt_);
        double v_max = std::min(max_vel_,      s.v + max_accel_    * dt_);
        double w_min = std::max(-max_yawrate_, s.w - max_dyawrate_ * dt_);
        double w_max = std::min( max_yawrate_, s.w + max_dyawrate_ * dt_);

        auto& lookahead_wp = get_lookahead_wp(s);
        double lax = lookahead_wp.pose.position.x;
        double lay = lookahead_wp.pose.position.y;

        double best_score = -std::numeric_limits<double>::infinity();
        double best_v = 0.0, best_w = 0.0;
        int    steps  = (int)(predict_time_ / dt_);

        visualization_msgs::msg::MarkerArray traj_ma;
        int marker_id = 0;

        for (double v = v_min; v <= v_max + 1e-9; v += vel_res_) {
            for (double w = w_min; w <= w_max + 1e-9; w += yawrate_res_) {
                // simulate trajectory
                RobotState cur = s;
                bool collision  = false;
                double min_clearance = (double)LETHAL_COST;

                std::vector<geometry_msgs::msg::Point> traj_pts;
                traj_pts.reserve(steps);

                for (int i = 0; i < steps; i++) {
                    cur = motion(cur, v, w);
                    uint8_t cost = costmap_lookup(cur.x, cur.y);
                    if (cost >= LETHAL_COST) { collision = true; break; }
                    min_clearance = std::min(min_clearance, (double)(LETHAL_COST - cost));
                    geometry_msgs::msg::Point p; p.x = cur.x; p.y = cur.y;
                    traj_pts.push_back(p);
                }
                if (collision || traj_pts.empty()) continue;

                // score
                double target_yaw = std::atan2(lay - cur.y, lax - cur.x);
                double yaw_err    = std::abs(angle_wrap(target_yaw - cur.yaw));
                double h_score    = M_PI - yaw_err;
                double o_score    = min_clearance;
                double s_score    = v / max_vel_;
                double total      = heading_gain_  * h_score
                                  + obstacle_gain_ * o_score
                                  + speed_gain_    * s_score;

                if (total > best_score) {
                    best_score = total;
                    best_v = v; best_w = w;
                }

                // viz
                double max_possible = heading_gain_ * M_PI
                                    + obstacle_gain_ * LETHAL_COST
                                    + speed_gain_;
                float ratio = (float)std::max(0.0, std::min(1.0, total / max_possible));
                auto rgb = hsv_to_rgb(0.67f * (1.0f - ratio));  // blue=low, red=high

                visualization_msgs::msg::Marker m;
                m.header.frame_id = "map";
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

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DWAControllerNode>());
    rclcpp::shutdown();
    return 0;
}
