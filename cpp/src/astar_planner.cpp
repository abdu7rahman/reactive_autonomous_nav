// astar_planner.cpp — A* Global Planner (C++)
// reactive_nav_cpp | ROS2 Jazzy
//
// Octile heuristic (tight for 8-connected grids)
// Collision-checked Laplacian path smoothing
// RViz markers: explored frontier + glowing path

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

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <vector>

// ── cost thresholds ──────────────────────────────────────────────────────────
static constexpr uint8_t LETHAL_COST           = 253;
static constexpr float   PATH_BLOCKED_LOOKAHEAD = 2.0f;   // m
static constexpr float   MAX_PATH_DEVIATION     = 0.8f;   // m
static constexpr float   MIN_REPLAN_DISTANCE    = 0.3f;   // m

// 8-connected grid neighbors
static constexpr int   DX[8]        = { 1,-1, 0, 0, 1,-1, 1,-1};
static constexpr int   DY[8]        = { 0, 0, 1,-1, 1,-1,-1, 1};
static constexpr float STEP_COST[8] = {1.0f,1.0f,1.0f,1.0f,1.414f,1.414f,1.414f,1.414f};

struct AStarNode {
    float f, g;
    int   idx;
    bool operator>(const AStarNode& o) const { return f > o.f; }
};

// ── helpers ──────────────────────────────────────────────────────────────────
static inline float octile(int dx, int dy)
{
    int a = std::abs(dx), b = std::abs(dy);
    if (a < b) std::swap(a, b);
    return static_cast<float>(a) + 0.4142f * static_cast<float>(b);
}

static inline uint8_t cell_cost(const nav_msgs::msg::OccupancyGrid& map, int idx)
{
    return static_cast<uint8_t>(map.data[idx]);
}

// ── node ─────────────────────────────────────────────────────────────────────
class AStarPlannerNode : public rclcpp::Node
{
public:
    AStarPlannerNode() : Node("astar_planner_node")
    {
        auto qos_map = rclcpp::QoS(1).transient_local().reliable();

        sub_global_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/global_costmap/costmap", qos_map,
            [this](nav_msgs::msg::OccupancyGrid::SharedPtr m){ global_map_ = m; });

        sub_local_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/local_costmap/costmap", qos_map,
            [this](nav_msgs::msg::OccupancyGrid::SharedPtr m){ local_map_ = m; });

        sub_goal_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            "/goal_pose", 10,
            std::bind(&AStarPlannerNode::on_goal, this, std::placeholders::_1));

        pub_plan_    = create_publisher<nav_msgs::msg::Path>("/global_plan", 10);
        pub_markers_ = create_publisher<visualization_msgs::msg::MarkerArray>(
                           "/astar/markers", 10);

        tf_buffer_   = std::make_shared<tf2_ros::Buffer>(get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        replan_timer_ = create_wall_timer(
            std::chrono::milliseconds(500),
            std::bind(&AStarPlannerNode::check_replan, this));

        RCLCPP_INFO(get_logger(), "A* Planner (C++) ready");
    }

private:
    // ── state ─────────────────────────────────────────────────────────────────
    nav_msgs::msg::OccupancyGrid::SharedPtr  global_map_;
    nav_msgs::msg::OccupancyGrid::SharedPtr  local_map_;
    nav_msgs::msg::Path::SharedPtr           current_path_;
    geometry_msgs::msg::PoseStamped::SharedPtr last_goal_;
    double last_replan_x_{0}, last_replan_y_{0};
    bool   goal_reached_{false};
    bool   has_replan_pose_{false};

    // ── ROS ───────────────────────────────────────────────────────────────────
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr  sub_global_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr  sub_local_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_goal_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr              pub_plan_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
    rclcpp::TimerBase::SharedPtr                                   replan_timer_;
    std::shared_ptr<tf2_ros::Buffer>                               tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener>                    tf_listener_;

    // ── TF ────────────────────────────────────────────────────────────────────
    bool get_robot_pose(double& rx, double& ry) const
    {
        try {
            auto tf = tf_buffer_->lookupTransform(
                "map", "base_link", tf2::TimePointZero, tf2::durationFromSec(0.3));
            rx = tf.transform.translation.x;
            ry = tf.transform.translation.y;
            return true;
        } catch (...) { return false; }
    }

    // ── grid utils ────────────────────────────────────────────────────────────
    bool map_free(const nav_msgs::msg::OccupancyGrid& map, int x, int y) const
    {
        const auto& info = map.info;
        if (x < 0 || y < 0 || x >= (int)info.width || y >= (int)info.height) return false;
        return cell_cost(map, y * (int)info.width + x) < LETHAL_COST;
    }

    std::pair<int,int> world_to_grid(const nav_msgs::msg::OccupancyGrid& map,
                                     double wx, double wy) const
    {
        float res = map.info.resolution;
        return {
            static_cast<int>((wx - map.info.origin.position.x) / res),
            static_cast<int>((wy - map.info.origin.position.y) / res)
        };
    }

    std::pair<double,double> grid_to_world(const nav_msgs::msg::OccupancyGrid& map,
                                           int gx, int gy) const
    {
        float res = map.info.resolution;
        return {
            map.info.origin.position.x + (gx + 0.5) * res,
            map.info.origin.position.y + (gy + 0.5) * res
        };
    }

    // ── A* core ───────────────────────────────────────────────────────────────
    std::vector<std::pair<int,int>> run_astar(
        const nav_msgs::msg::OccupancyGrid& map,
        int sx, int sy, int gx, int gy) const
    {
        const int W = (int)map.info.width, H = (int)map.info.height;
        if (sx < 0 || sy < 0 || sx >= W || sy >= H) return {};
        if (gx < 0 || gy < 0 || gx >= W || gy >= H) return {};

        const int total    = W * H;
        const int goal_idx = gy * W + gx;
        const int start    = sy * W + sx;

        std::vector<float> g_cost(total, std::numeric_limits<float>::infinity());
        std::vector<int>   parent(total, -1);
        std::vector<bool>  closed(total, false);

        g_cost[start] = 0.0f;

        std::priority_queue<AStarNode, std::vector<AStarNode>, std::greater<AStarNode>> open;
        open.push({octile(gx - sx, gy - sy), 0.0f, start});

        while (!open.empty()) {
            auto [f, g, cur] = open.top(); open.pop();
            if (closed[cur]) continue;
            closed[cur] = true;
            if (cur == goal_idx) break;

            int cx = cur % W, cy = cur / W;
            for (int d = 0; d < 8; d++) {
                int nx = cx + DX[d], ny = cy + DY[d];
                if (!map_free(map, nx, ny)) continue;
                int nidx = ny * W + nx;
                if (closed[nidx]) continue;

                // weight by cell cost so planner naturally avoids high-cost cells
                float step_g = STEP_COST[d] * (1.0f + cell_cost(map, nidx) / 255.0f);
                float ng     = g + step_g;
                if (ng < g_cost[nidx]) {
                    g_cost[nidx] = ng;
                    parent[nidx] = cur;
                    open.push({ng + octile(gx - nx, gy - ny), ng, nidx});
                }
            }
        }

        // backtrack
        std::vector<std::pair<int,int>> path;
        int cur = goal_idx;
        while (cur != -1 && cur != start) {
            path.emplace_back(cur % W, cur / W);
            cur = parent[cur];
        }
        if (cur == start) path.emplace_back(sx, sy);
        std::reverse(path.begin(), path.end());
        return path;
    }

    // ── Laplacian smoothing ────────────────────────────────────────────────────
    std::vector<std::pair<int,int>> smooth_path(
        std::vector<std::pair<int,int>> path,
        const nav_msgs::msg::OccupancyGrid& map,
        int iters = 5, float alpha = 0.5f) const
    {
        for (int it = 0; it < iters; it++) {
            for (size_t i = 1; i + 1 < path.size(); i++) {
                int nx = static_cast<int>((path[i-1].first  + path[i+1].first)  / 2.0f * alpha
                                        + path[i].first  * (1.0f - alpha));
                int ny = static_cast<int>((path[i-1].second + path[i+1].second) / 2.0f * alpha
                                        + path[i].second * (1.0f - alpha));
                if (map_free(map, nx, ny)) path[i] = {nx, ny};
            }
        }
        return path;
    }

    // ── planning ─────────────────────────────────────────────────────────────
    void plan_to_goal(const geometry_msgs::msg::PoseStamped& goal)
    {
        if (!global_map_) { RCLCPP_WARN(get_logger(), "No costmap yet"); return; }
        double rx, ry;
        if (!get_robot_pose(rx, ry)) { RCLCPP_WARN(get_logger(), "TF not ready"); return; }

        auto [sx, sy] = world_to_grid(*global_map_, rx, ry);
        auto [gx, gy] = world_to_grid(*global_map_, goal.pose.position.x, goal.pose.position.y);

        auto raw      = run_astar(*global_map_, sx, sy, gx, gy);
        if (raw.empty()) { RCLCPP_WARN(get_logger(), "A* found no path"); return; }
        auto smoothed = smooth_path(raw, *global_map_);

        nav_msgs::msg::Path plan;
        plan.header.frame_id = "map";
        plan.header.stamp    = now();
        for (auto& [px, py] : smoothed) {
            auto [wx, wy] = grid_to_world(*global_map_, px, py);
            geometry_msgs::msg::PoseStamped ps;
            ps.header              = plan.header;
            ps.pose.position.x     = wx;
            ps.pose.position.y     = wy;
            ps.pose.orientation.w  = 1.0;
            plan.poses.push_back(ps);
        }

        pub_plan_->publish(plan);
        current_path_     = std::make_shared<nav_msgs::msg::Path>(plan);
        goal_reached_     = false;
        has_replan_pose_  = false;

        RCLCPP_INFO(get_logger(), "Plan published (%zu waypoints)", plan.poses.size());
        publish_path_markers(plan);
    }

    void on_goal(geometry_msgs::msg::PoseStamped::SharedPtr msg)
    {
        last_goal_    = msg;
        goal_reached_ = false;
        plan_to_goal(*msg);
    }

    void check_replan()
    {
        if (!last_goal_ || goal_reached_ || !current_path_ || !global_map_) return;
        double rx, ry;
        if (!get_robot_pose(rx, ry)) return;

        double to_goal = std::hypot(
            rx - last_goal_->pose.position.x,
            ry - last_goal_->pose.position.y);
        if (to_goal < 0.3) { goal_reached_ = true; RCLCPP_INFO(get_logger(), "Goal reached"); return; }

        // don't replan if robot hasn't moved far enough
        if (has_replan_pose_) {
            if (std::hypot(rx - last_replan_x_, ry - last_replan_y_) < MIN_REPLAN_DISTANCE) return;
        }

        // check if robot drifted off path
        float min_d = std::numeric_limits<float>::max();
        for (auto& ps : current_path_->poses) {
            float d = (float)std::hypot(rx - ps.pose.position.x, ry - ps.pose.position.y);
            if (d < min_d) min_d = d;
        }
        if (min_d > MAX_PATH_DEVIATION) {
            RCLCPP_INFO(get_logger(), "Replanning — deviation %.2f m", min_d);
            last_replan_x_   = rx;
            last_replan_y_   = ry;
            has_replan_pose_ = true;
            plan_to_goal(*last_goal_);
        }
    }

    // ── RViz markers ──────────────────────────────────────────────────────────
    void publish_path_markers(const nav_msgs::msg::Path& plan) const
    {
        visualization_msgs::msg::MarkerArray ma;

        visualization_msgs::msg::Marker line;
        line.header    = plan.header;
        line.ns        = "astar_path";
        line.id        = 0;
        line.type      = visualization_msgs::msg::Marker::LINE_STRIP;
        line.action    = visualization_msgs::msg::Marker::ADD;
        line.scale.x   = 0.05f;
        line.color.r   = 0.0f;
        line.color.g   = 1.0f;
        line.color.b   = 0.4f;
        line.color.a   = 1.0f;
        for (auto& ps : plan.poses) {
            geometry_msgs::msg::Point p;
            p.x = ps.pose.position.x;
            p.y = ps.pose.position.y;
            line.points.push_back(p);
        }
        ma.markers.push_back(line);

        // start sphere
        visualization_msgs::msg::Marker start_m;
        start_m.header  = plan.header;
        start_m.ns      = "astar_start";
        start_m.id      = 1;
        start_m.type    = visualization_msgs::msg::Marker::SPHERE;
        start_m.action  = visualization_msgs::msg::Marker::ADD;
        start_m.scale.x = start_m.scale.y = start_m.scale.z = 0.2;
        start_m.color.r = 0.2f; start_m.color.g = 0.8f; start_m.color.b = 0.2f; start_m.color.a = 1.0f;
        if (!plan.poses.empty()) start_m.pose = plan.poses.front().pose;
        ma.markers.push_back(start_m);

        // goal sphere
        visualization_msgs::msg::Marker goal_m = start_m;
        goal_m.ns     = "astar_goal";
        goal_m.id     = 2;
        goal_m.color.r = 1.0f; goal_m.color.g = 0.2f; goal_m.color.b = 0.2f;
        if (!plan.poses.empty()) goal_m.pose = plan.poses.back().pose;
        ma.markers.push_back(goal_m);

        pub_markers_->publish(ma);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<AStarPlannerNode>());
    rclcpp::shutdown();
    return 0;
}
