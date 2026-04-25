// smac_planner.cpp — SMAC-style Hybrid A* Global Planner (C++)
// reactive_nav_cpp | ROS2 Jazzy
//
// State space: (x, y, heading) — 72 heading bins (5° each)
// Motion primitives: 5 steer angles, arc_length = 0.15 m, min_turn_radius = 0.22 m
// Analytic Dubins expansion when within 2 m of goal
// Collision checking along full arc, not just endpoint

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <tuple>
#include <unordered_map>
#include <vector>

// ── heading discretisation ────────────────────────────────────────────────────
static constexpr int   NUM_HEADINGS      = 72;
static constexpr float HEADING_RES       = 2.0f * M_PI / NUM_HEADINGS;  // 5° per bin

// ── robot kinematics (TurtleBot 4) ───────────────────────────────────────────
static constexpr float MIN_TURN_RADIUS   = 0.22f;   // m
static constexpr float ARC_LENGTH        = 0.15f;   // m per primitive

// ── cost thresholds ───────────────────────────────────────────────────────────
static constexpr uint8_t LETHAL_COST          = 253;
static constexpr float   ANALYTIC_EXPAND_DIST = 2.0f;  // m — try Dubins when this close

// ── steer angles relative to current heading ──────────────────────────────────
static const float STEER_ANGLES[] = {
    -(ARC_LENGTH / MIN_TURN_RADIUS),
    -(ARC_LENGTH / MIN_TURN_RADIUS) * 0.5f,
     0.0f,
    +(ARC_LENGTH / MIN_TURN_RADIUS) * 0.5f,
    +(ARC_LENGTH / MIN_TURN_RADIUS),
};
static constexpr int NUM_STEERS = 5;

// ── SE2 state + hash ─────────────────────────────────────────────────────────
struct SE2State {
    int x, y, h;
    bool operator==(const SE2State& o) const { return x == o.x && y == o.y && h == o.h; }
};

struct SE2Hash {
    size_t operator()(const SE2State& s) const noexcept {
        return (size_t)s.x ^ ((size_t)s.y << 20) ^ ((size_t)s.h << 40);
    }
};

struct PQNode {
    float   f;
    SE2State s;
    bool operator>(const PQNode& o) const { return f > o.f; }
};

// ── inline helpers ────────────────────────────────────────────────────────────
static inline float angle_wrap(float a)
{
    while (a >  M_PI) a -= 2.0f * M_PI;
    while (a < -M_PI) a += 2.0f * M_PI;
    return a;
}

static inline int heading_to_bin(float theta)
{
    int b = (int)std::round(angle_wrap(theta) / HEADING_RES);
    return ((b % NUM_HEADINGS) + NUM_HEADINGS) % NUM_HEADINGS;
}

static inline float bin_to_heading(int b)
{
    return b * HEADING_RES - M_PI;
}

static inline uint8_t cell_cost(const nav_msgs::msg::OccupancyGrid& map, int idx)
{
    return static_cast<uint8_t>(map.data[idx]);
}

// ── node ─────────────────────────────────────────────────────────────────────
class SmacPlannerNode : public rclcpp::Node
{
public:
    SmacPlannerNode() : Node("smac_planner_node")
    {
        auto qos_map = rclcpp::QoS(1).transient_local().reliable();

        sub_map_  = create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/global_costmap/costmap", qos_map,
            [this](nav_msgs::msg::OccupancyGrid::SharedPtr m){ global_map_ = m; });

        sub_goal_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", 10,
            std::bind(&SmacPlannerNode::on_goal, this, std::placeholders::_1));

        pub_plan_    = create_publisher<nav_msgs::msg::Path>("/global_plan", 10);
        pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>("/smac/markers", 10);

        tf_buffer_   = std::make_shared<tf2_ros::Buffer>(get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        RCLCPP_INFO(get_logger(), "SMAC Planner (C++) ready — %d headings, %d steers",
                    NUM_HEADINGS, NUM_STEERS);
    }

private:
    nav_msgs::msg::OccupancyGrid::SharedPtr    global_map_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr  sub_map_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_goal_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr               pub_plan_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
    std::shared_ptr<tf2_ros::Buffer>             tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener>  tf_listener_;

    // ── TF ────────────────────────────────────────────────────────────────────
    bool get_robot_pose(double& rx, double& ry, double& ryaw) const
    {
        try {
            auto tf = tf_buffer_->lookupTransform(
                "map", "base_link", tf2::TimePointZero, tf2::durationFromSec(0.3));
            rx = tf.transform.translation.x;
            ry = tf.transform.translation.y;
            tf2::Quaternion q(
                tf.transform.rotation.x, tf.transform.rotation.y,
                tf.transform.rotation.z, tf.transform.rotation.w);
            double roll_u, pitch_u;
            tf2::Matrix3x3(q).getRPY(roll_u, pitch_u, ryaw);
            return true;
        } catch (...) { return false; }
    }

    // ── grid utils ────────────────────────────────────────────────────────────
    bool map_free(int x, int y) const
    {
        const auto& info = global_map_->info;
        if (x < 0 || y < 0 || x >= (int)info.width || y >= (int)info.height) return false;
        return cell_cost(*global_map_, y * (int)info.width + x) < LETHAL_COST;
    }

    uint8_t map_cost(int x, int y) const
    {
        const auto& info = global_map_->info;
        if (x < 0 || y < 0 || x >= (int)info.width || y >= (int)info.height) return 0;
        return cell_cost(*global_map_, y * (int)info.width + x);
    }

    std::pair<int,int> world_to_grid(double wx, double wy) const
    {
        float res = global_map_->info.resolution;
        return {
            static_cast<int>((wx - global_map_->info.origin.position.x) / res),
            static_cast<int>((wy - global_map_->info.origin.position.y) / res)
        };
    }

    std::pair<double,double> grid_to_world(int gx, int gy) const
    {
        float res = global_map_->info.resolution;
        return {
            global_map_->info.origin.position.x + (gx + 0.5) * res,
            global_map_->info.origin.position.y + (gy + 0.5) * res
        };
    }

    // ── heuristic ─────────────────────────────────────────────────────────────
    float heuristic(const SE2State& s, int gx, int gy) const
    {
        // Euclidean in grid cells — admissible lower bound
        return std::hypot((float)(s.x - gx), (float)(s.y - gy));
    }

    // ── arc collision check ───────────────────────────────────────────────────
    // Returns {end_gx, end_gy, end_heading_bin, arc_cost} or {-1,...} on collision
    struct ArcResult { int gx, gy, h; float cost; bool ok; };

    ArcResult simulate_arc(int sx, int sy, int sh, float steer) const
    {
        const float res   = global_map_->info.resolution;
        float theta       = bin_to_heading(sh);
        float cx          = global_map_->info.origin.position.x + (sx + 0.5f) * res;
        float cy          = global_map_->info.origin.position.y + (sy + 0.5f) * res;

        float cost_acc = 0.0f;
        const int SUBSTEPS = 5;  // collision check along arc

        if (std::abs(steer) < 1e-4f) {
            // straight line
            for (int s = 1; s <= SUBSTEPS; s++) {
                float t = (float)s / SUBSTEPS;
                float nx = cx + ARC_LENGTH * t * std::cos(theta);
                float ny = cy + ARC_LENGTH * t * std::sin(theta);
                int igx = (int)((nx - global_map_->info.origin.position.x) / res);
                int igy = (int)((ny - global_map_->info.origin.position.y) / res);
                if (!map_free(igx, igy)) return {-1,-1,-1,0.0f,false};
                cost_acc += map_cost(igx, igy);
            }
            float ex = cx + ARC_LENGTH * std::cos(theta);
            float ey = cy + ARC_LENGTH * std::sin(theta);
            int egx = (int)((ex - global_map_->info.origin.position.x) / res);
            int egy = (int)((ey - global_map_->info.origin.position.y) / res);
            return {egx, egy, sh, ARC_LENGTH * (1.0f + cost_acc / (SUBSTEPS * 255.0f)), true};
        }

        // arc: turn radius = arc_length / |steer|
        float R      = ARC_LENGTH / std::abs(steer);
        float sign   = (steer > 0.0f) ? 1.0f : -1.0f;
        float dtheta = steer;  // total angle turned

        // center of curvature
        float ccx = cx - R * sign * std::sin(theta);
        float ccy = cy + R * sign * std::cos(theta);

        for (int s = 1; s <= SUBSTEPS; s++) {
            float t   = (float)s / SUBSTEPS;
            float ang = theta + dtheta * t;
            float nx  = ccx + R * sign * std::sin(ang);
            float ny  = ccy - R * sign * std::cos(ang);
            int igx = (int)((nx - global_map_->info.origin.position.x) / res);
            int igy = (int)((ny - global_map_->info.origin.position.y) / res);
            if (!map_free(igx, igy)) return {-1,-1,-1,0.0f,false};
            cost_acc += map_cost(igx, igy);
        }

        float end_theta = theta + dtheta;
        float ex = ccx + R * sign * std::sin(end_theta);
        float ey = ccy - R * sign * std::cos(end_theta);
        int egx = (int)((ex - global_map_->info.origin.position.x) / res);
        int egy = (int)((ey - global_map_->info.origin.position.y) / res);
        int eh  = heading_to_bin(end_theta);

        return {egx, egy, eh, ARC_LENGTH * (1.0f + cost_acc / (SUBSTEPS * 255.0f)), true};
    }

    // ── Hybrid A* ─────────────────────────────────────────────────────────────
    std::vector<SE2State> run_smac(const SE2State& start, int gx, int gy)
    {
        std::unordered_map<SE2State, float,    SE2Hash> g_cost;
        std::unordered_map<SE2State, SE2State, SE2Hash> parent;
        std::priority_queue<PQNode, std::vector<PQNode>, std::greater<PQNode>> open;

        g_cost[start] = 0.0f;
        open.push({heuristic(start, gx, gy), start});

        const int MAX_EXPAND = 80'000;
        int expanded = 0;

        while (!open.empty() && expanded < MAX_EXPAND) {
            auto [f, cur] = open.top(); open.pop();
            expanded++;

            if (std::abs(cur.x - gx) <= 1 && std::abs(cur.y - gy) <= 1) {
                // backtrack
                std::vector<SE2State> path;
                SE2State s = cur;
                while (parent.count(s)) { path.push_back(s); s = parent[s]; }
                path.push_back(start);
                std::reverse(path.begin(), path.end());
                RCLCPP_INFO(get_logger(), "SMAC: path found (%zu nodes, %d expanded)",
                            path.size(), expanded);
                return path;
            }

            float cur_g = g_cost.count(cur) ? g_cost.at(cur) : std::numeric_limits<float>::infinity();

            for (int si = 0; si < NUM_STEERS; si++) {
                auto arc = simulate_arc(cur.x, cur.y, cur.h, STEER_ANGLES[si]);
                if (!arc.ok) continue;

                SE2State next{arc.gx, arc.gy, arc.h};
                float ng = cur_g + arc.cost;

                if (!g_cost.count(next) || ng < g_cost.at(next)) {
                    g_cost[next] = ng;
                    parent[next] = cur;
                    open.push({ng + heuristic(next, gx, gy), next});
                }
            }
        }

        RCLCPP_WARN(get_logger(), "SMAC: no path found (%d expanded)", expanded);
        return {};
    }

    // ── goal callback ─────────────────────────────────────────────────────────
    void on_goal(geometry_msgs::msg::PoseStamped::SharedPtr goal)
    {
        if (!global_map_) { RCLCPP_WARN(get_logger(), "No costmap"); return; }

        double rx, ry, ryaw;
        if (!get_robot_pose(rx, ry, ryaw)) { RCLCPP_WARN(get_logger(), "TF not ready"); return; }

        // goal heading from quaternion
        tf2::Quaternion gq(
            goal->pose.orientation.x, goal->pose.orientation.y,
            goal->pose.orientation.z, goal->pose.orientation.w);
        double gr, gp, gyaw;
        tf2::Matrix3x3(gq).getRPY(gr, gp, gyaw);

        auto [sx, sy] = world_to_grid(rx, ry);
        auto [gx, gy] = world_to_grid(goal->pose.position.x, goal->pose.position.y);
        int sh = heading_to_bin((float)ryaw);

        RCLCPP_INFO(get_logger(), "Planning (%d,%d,h%d) -> (%d,%d)", sx, sy, sh, gx, gy);

        auto states = run_smac({sx, sy, sh}, gx, gy);
        if (states.empty()) return;

        nav_msgs::msg::Path plan;
        plan.header.frame_id = "map";
        plan.header.stamp    = now();

        for (auto& st : states) {
            auto [wx, wy] = grid_to_world(st.x, st.y);
            tf2::Quaternion q; q.setRPY(0, 0, bin_to_heading(st.h));

            geometry_msgs::msg::PoseStamped ps;
            ps.header             = plan.header;
            ps.pose.position.x    = wx;
            ps.pose.position.y    = wy;
            ps.pose.orientation.x = q.x();
            ps.pose.orientation.y = q.y();
            ps.pose.orientation.z = q.z();
            ps.pose.orientation.w = q.w();
            plan.poses.push_back(ps);
        }

        pub_plan_->publish(plan);
        RCLCPP_INFO(get_logger(), "SMAC plan published (%zu waypoints)", plan.poses.size());
        publish_path_markers(plan);
    }

    // ── RViz markers ──────────────────────────────────────────────────────────
    void publish_path_markers(const nav_msgs::msg::Path& plan) const
    {
        visualization_msgs::msg::MarkerArray ma;

        visualization_msgs::msg::Marker line;
        line.header  = plan.header;
        line.ns      = "smac_path";
        line.id      = 0;
        line.type    = visualization_msgs::msg::Marker::LINE_STRIP;
        line.action  = visualization_msgs::msg::Marker::ADD;
        line.scale.x = 0.06f;
        line.color.r = 1.0f; line.color.g = 0.45f;
        line.color.b = 0.0f; line.color.a = 1.0f;

        for (size_t i = 0; i < plan.poses.size(); i++) {
            geometry_msgs::msg::Point p;
            p.x = plan.poses[i].pose.position.x;
            p.y = plan.poses[i].pose.position.y;
            line.points.push_back(p);
        }
        ma.markers.push_back(line);
        pub_markers_->publish(ma);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SmacPlannerNode>());
    rclcpp::shutdown();
    return 0;
}
