// Races somebody else's DWA on a real robot, in the live graph -- the C and
// C++ half of it.
//
// reactive_autonomous_nav/baseline_controller.py is the same node for the two
// reference Python implementations, and its header carries the two decisions
// both of them rest on: obstacles are the local costmap's own lethal cells,
// and the goal handed to a goal seeker is the same lookahead point this
// repo's DWA steers at, because the chicane is a tracking course. They are
// not repeated here.
//
// What is here rather than there is why this file exists at all: these three
// are C and C++, they already compile into bench/dwa_compare_cpp, and their
// per-tick scoring is bench/baseline_*.cpp's step_* -- the same function the
// figure's traces loop over. Nothing about their planners is reimplemented
// for the robot. If the clip and the figure ever disagree it is the robot or
// the costmap that differs, never the controller.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "trace.h"

namespace {

// The plant's own limits and this repo's own sampling, so a lane differs by
// its scoring function and nothing else. Same source as dwa_controller.cpp:
// irobot_create_control/config/control.yaml by way of sim/race_robot.py.
constexpr double MAX_V = 0.50, MAX_W = 2.0, ACC_V = 0.9, ACC_W = 7.725;
constexpr double VEL_RES = 0.02, YAW_RES = 0.04, DT = 0.1;
constexpr int    HORIZON = 25;
constexpr int    LOOKAHEAD_WPS = 8;
constexpr double WP_TOL = 0.25, GOAL_TOL = 0.15;
constexpr uint8_t LETHAL_COST = 253;

// The costmap's own inscribed radius, out of config/race_costmap_params.yaml:
// a cell is 253 when the robot's centre there would be within 0.22 m of
// something occupied, so refusing a rollout point within 0.22 m of an
// occupied cell is this repo's lethal test written their way. The Python
// host's header records what handing them the inscribed ring instead
// measured.
constexpr double INSCRIBED_R = 0.22;

// 0.5 m/s for a 2.5 s rollout is 1.25 m, and turning makes that a disc rather
// than a line, so only the lethal cells inside it can change a decision.
constexpr double NEAR_R = 1.6;

double yaw_of(const geometry_msgs::msg::Quaternion& q) {
    return std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

class BaselineControllerNode : public rclcpp::Node {
public:
    BaselineControllerNode() : Node("baseline_controller_node") {
        impl_ = declare_parameter<std::string>("impl", "cpprobotics");
        map_frame_ = declare_parameter<std::string>("map_frame", "map");
        declare_parameter<std::string>("odom_frame", "odom");
        base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
        if (impl_ != "cpprobotics" && impl_ != "goktug97" &&
            impl_ != "amslabtech") {
            RCLCPP_FATAL(get_logger(), "impl=%s is not one of cpprobotics, "
                         "goktug97, amslabtech", impl_.c_str());
            throw std::runtime_error("unknown impl");
        }

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        rclcpp::QoS qos_map(1);
        qos_map.reliable().transient_local();
        pub_cmd_ = create_publisher<geometry_msgs::msg::Twist>(
            "/cmd_vel_unstamped", 10);
        pub_status_ = create_publisher<std_msgs::msg::String>(
            "/controller_status", 10);
        sub_plan_ = create_subscription<nav_msgs::msg::Path>(
            "/plan", 10, [this](nav_msgs::msg::Path::SharedPtr m) {
                if (m->poses.empty()) return;
                plan_ = m; wp_ = 0; reached_ = false; });
        sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, [this](nav_msgs::msg::Odometry::SharedPtr m) {
                v_ = m->twist.twist.linear.x; w_ = m->twist.twist.angular.z; });
        sub_map_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/local_costmap/costmap", qos_map,
            std::bind(&BaselineControllerNode::on_costmap, this,
                      std::placeholders::_1));
        timer_ = create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&BaselineControllerNode::tick, this));
        RCLCPP_INFO(get_logger(), "baseline controller: %s -- ready",
                    impl_.c_str());
    }

private:
    // nav2 publishes the 0..100 OccupancyGrid, not the raw scale; the same
    // inverse map dwa_controller.cpp applies, and for the same reason its
    // comment records. Only the occupied cells are kept, as their centres --
    // 100 here is 254 raw, LETHAL_OBSTACLE -- because that is the whole
    // obstacle list these three are given, and the inscribed ring at 99 is
    // the same information already counted by their radius.
    void on_costmap(nav_msgs::msg::OccupancyGrid::SharedPtr m) {
        map_ = m;
        obx_.clear(); oby_.clear();
        const auto& info = m->info;
        for (size_t i = 0; i < m->data.size(); i++) {
            if (m->data[i] != 100) continue;      // 254 raw, LETHAL_OBSTACLE
            const size_t gx = i % info.width, gy = i / info.width;
            obx_.push_back(info.origin.position.x + (gx + 0.5) * info.resolution);
            oby_.push_back(info.origin.position.y + (gy + 0.5) * info.resolution);
        }
    }

    uint8_t cost_at(double wx, double wy) const {
        if (!map_) return 0;
        const auto& info = map_->info;
        const int gx = (int)((wx - info.origin.position.x) / info.resolution);
        const int gy = (int)((wy - info.origin.position.y) / info.resolution);
        if (gx < 0 || gy < 0 || gx >= (int)info.width || gy >= (int)info.height)
            return 0;
        const int8_t g = map_->data[(size_t)gy * info.width + (size_t)gx];
        return g < 0 ? 0 : g == 100 ? 254 : g == 99 ? 253 : (uint8_t)(g * 252 / 99);
    }

    bool visible(double x0, double y0, double x1, double y1) const {
        const double d = std::hypot(x1 - x0, y1 - y0);
        const int n = std::max(2, (int)(d / (map_ ? map_->info.resolution : 0.05)));
        for (int i = 0; i <= n; i++) {
            const double t = (double)i / n;
            if (cost_at(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t) >= LETHAL_COST)
                return false;
        }
        return true;
    }

    void tick() {
        if (!plan_ || !map_) return;
        geometry_msgs::msg::TransformStamped tf;
        try {
            tf = tf_buffer_->lookupTransform(map_frame_, base_frame_,
                                             tf2::TimePointZero);
        } catch (const std::exception&) { return; }
        const double rx = tf.transform.translation.x;
        const double ry = tf.transform.translation.y;
        const double ryaw = yaw_of(tf.transform.rotation);

        const auto& last = plan_->poses.back().pose.position;
        if (std::hypot(last.x - rx, last.y - ry) < GOAL_TOL) {
            if (!reached_) {
                RCLCPP_INFO(get_logger(), "Goal REACHED");
                std_msgs::msg::String s; s.data = "REACHED";
                pub_status_->publish(s);
                reached_ = true;
            }
            pub_cmd_->publish(geometry_msgs::msg::Twist());
            return;
        }

        // The same lookahead this repo's own DWA steers at.
        while (wp_ + 1 < (int)plan_->poses.size()) {
            const auto& cur = plan_->poses[wp_].pose.position;
            const auto& nxt = plan_->poses[wp_ + 1].pose.position;
            const double d = std::hypot(cur.x - rx, cur.y - ry);
            if (d < WP_TOL || std::hypot(nxt.x - rx, nxt.y - ry) < d) wp_++;
            else break;
        }
        int tidx = std::min(wp_ + LOOKAHEAD_WPS, (int)plan_->poses.size() - 1);
        while (tidx > wp_) {
            const auto& p = plan_->poses[tidx].pose.position;
            if (visible(rx, ry, p.x, p.y)) break;
            tidx--;
        }
        const auto& carrot = plan_->poses[tidx].pose.position;

        // An empty obstacle list divides by zero in two of the three and
        // takes min() of nothing in the third. One cell far enough away to
        // change no decision costs a branch and removes that whole class of
        // crash.
        std::vector<double> nx, ny;
        for (size_t i = 0; i < obx_.size(); i++) {
            const double dx = obx_[i] - rx, dy = oby_[i] - ry;
            if (dx * dx + dy * dy <= NEAR_R * NEAR_R) {
                nx.push_back(obx_[i]); ny.push_back(oby_[i]);
            }
        }
        if (nx.empty()) { nx.push_back(rx + 1e3); ny.push_back(ry + 1e3); }

        StepIn in;
        in.obx = nx.data(); in.oby = ny.data(); in.nob = (int)nx.size();
        in.x = rx; in.y = ry; in.yaw = ryaw; in.v = v_; in.w = w_;
        in.gx = carrot.x; in.gy = carrot.y;
        in.dt = DT; in.steps = HORIZON;
        in.top_speed = MAX_V; in.max_yaw = MAX_W;
        in.acc_v = ACC_V; in.acc_w = ACC_W;
        in.vres = VEL_RES; in.wres = YAW_RES;
        in.radius = INSCRIBED_R;

        StepOut out;
        if (impl_ == "cpprobotics") step_cpprobotics(in, out);
        else if (impl_ == "goktug97") step_goktug(in, out);
        else step_amslabtech(in, out);

        // The same clamp bench/trace.h applies: goktug97's and amslabtech's
        // recoveries both ask for a fixed turn rate the base could not do.
        geometry_msgs::msg::Twist cmd;
        cmd.linear.x = std::fmin(std::fmax(out.v, -MAX_V), MAX_V);
        cmd.angular.z = std::fmin(std::fmax(out.w, -MAX_W), MAX_W);
        pub_cmd_->publish(cmd);

        if (++ticks_ % 50 == 0) {
            std_msgs::msg::String s;
            s.data = impl_ + " wp=" + std::to_string(wp_) + "/" +
                     std::to_string(plan_->poses.size()) + " obs=" +
                     std::to_string(nx.size()) + " " +
                     std::to_string(out.ms) + " ms";
            pub_status_->publish(s);
        }
    }

    std::string impl_, map_frame_, base_frame_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_cmd_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_status_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_plan_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr sub_map_;
    rclcpp::TimerBase::SharedPtr timer_;
    nav_msgs::msg::Path::SharedPtr plan_;
    nav_msgs::msg::OccupancyGrid::SharedPtr map_;
    std::vector<double> obx_, oby_;
    double v_ = 0.0, w_ = 0.0;
    int wp_ = 0, ticks_ = 0;
    bool reached_ = false;
};

}  // namespace

int main(int argc, char** argv) {
    if (isatty(2)) std::fprintf(stderr, "  %s\n", trace_author().c_str());
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<BaselineControllerNode>());
    rclcpp::shutdown();
    return 0;
}
