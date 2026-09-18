// Baseline: amslabtech/dwa_planner, the most-used standalone ROS DWA package.
//
// Its scoring core is lifted verbatim from src/dwa_planner.cpp -- motion,
// generate_trajectory, calc_dynamic_window, calc_to_goal_cost, calc_obs_cost,
// calc_speed_cost and evaluate_trajectory are byte-for-byte what that file
// contains, with `DWAPlanner::` dropped and the members that back them made
// file-scope. Nothing about how it searches or scores has been touched.
//
// What is stubbed is only the ROS boundary the node sits behind: Eigen's
// Vector3d, geometry_msgs::PoseArray for the obstacle list, and the members
// the node reads out of parameters. The parts that make it a node -- the
// subscribers, the costmap and scan callbacks, the visualisation -- are not
// part of the planning work and are not timed here or in this repo's numbers.
//
// The structural difference is the same one the other baselines show: this
// implementation walks the whole obstacle list for every rollout point, which
// is O(samples * horizon * obstacles), while this repo reads a costmap cell,
// which is O(samples * horizon).
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "trace.h"

namespace ams {

// ── the ROS boundary, stubbed ───────────────────────────────────────
struct Vec2 {
    double x_, y_;
    double norm() const { return std::hypot(x_, y_); }
};
struct Vector3d {
    double x_, y_, z_;
    Vector3d(double a = 0, double b = 0, double c = 0) : x_(a), y_(b), z_(c) {}
    double x() const { return x_; }
    double y() const { return y_; }
    Vec2 segment(int, int) const { return Vec2{x_, y_}; }
};
static inline Vec2 operator-(const Vec2& a, const Vec2& b) { return Vec2{a.x_ - b.x_, a.y_ - b.y_}; }

struct Point { double x, y, z; };
struct Pose { Point position; };
struct PoseArray { std::vector<Pose> poses; };
struct Twist { struct { double x; } linear; struct { double z; } angular; };

// ── the node's own classes, from include/dwa_planner/dwa_planner.h ──
class State {
public:
    State() : x_(0.0), y_(0.0), yaw_(0.0), velocity_(0.0), yawrate_(0.0) {}
    double x_, y_, yaw_, velocity_, yawrate_;
};
class Window {
public:
    double min_velocity_, max_velocity_, min_yawrate_, max_yawrate_;
};
class Cost {
public:
    void calc_total_cost(void) { total_cost_ = obs_cost_ + to_goal_cost_ + speed_cost_ + path_cost_; }
    float obs_cost_ = 0, to_goal_cost_ = 0, speed_cost_ = 0, path_cost_ = 0, total_cost_ = 0;
};

// ── the members the scoring functions read ──────────────────────────
static double max_deceleration_ = 4.0, max_acceleration_ = 0.4, sim_period_ = 0.1;
static double min_velocity_ = 0.0, target_velocity_ = 0.5;
static double max_yawrate_ = 2.0, max_d_yawrate_ = 1.0;
static double obs_range_ = 2.5, robot_radius_ = 0.17, footprint_padding_ = 0.01;
static double predict_time_ = 2.5;
static int sim_time_samples_ = 25;
static bool use_footprint_ = false, use_speed_cost_ = true, use_path_cost_ = false;
static PoseArray obs_list_;
static Twist current_cmd_vel_;

// ── verbatim from src/dwa_planner.cpp ───────────────────────────────
static void motion(State& state, const double velocity, const double yawrate) {
    const double sim_time_step = predict_time_ / static_cast<double>(sim_time_samples_);
    state.yaw_ += yawrate * sim_time_step;
    state.x_ += velocity * std::cos(state.yaw_) * sim_time_step;
    state.y_ += velocity * std::sin(state.yaw_) * sim_time_step;
    state.velocity_ = velocity;
    state.yawrate_ = yawrate;
}

static Window calc_dynamic_window(void) {
    Window window;
    window.min_velocity_ = std::max((current_cmd_vel_.linear.x - max_deceleration_ * sim_period_), min_velocity_);
    window.max_velocity_ = std::min((current_cmd_vel_.linear.x + max_acceleration_ * sim_period_), target_velocity_);
    window.min_yawrate_ = std::max((current_cmd_vel_.angular.z - max_d_yawrate_ * sim_period_), -max_yawrate_);
    window.max_yawrate_ = std::min((current_cmd_vel_.angular.z + max_d_yawrate_ * sim_period_), max_yawrate_);
    return window;
}

static float calc_to_goal_cost(const std::vector<State>& traj, const Vector3d& goal) {
    Vector3d last_position(traj.back().x_, traj.back().y_, traj.back().yaw_);
    return (last_position.segment(0, 2) - goal.segment(0, 2)).norm();
}

static float calc_obs_cost(const std::vector<State>& traj) {
    float min_dist = obs_range_;
    for (const auto& state : traj) {
        for (const auto& obs : obs_list_.poses) {
            float dist;
            dist = hypot((state.x_ - obs.position.x), (state.y_ - obs.position.y)) - robot_radius_ - footprint_padding_;
            if (dist < DBL_EPSILON) return 1e6;
            min_dist = std::min(min_dist, dist);
        }
    }
    return obs_range_ - min_dist;
}

static float calc_speed_cost(const std::vector<State>& traj) {
    if (!use_speed_cost_) return 0.0;
    const Window dynamic_window = calc_dynamic_window();
    return dynamic_window.max_velocity_ - traj.front().velocity_;
}

static std::vector<State> generate_trajectory(const double velocity, const double yawrate) {
    std::vector<State> trajectory;
    trajectory.resize(sim_time_samples_);
    State state;
    for (int i = 0; i < sim_time_samples_; i++) {
        motion(state, velocity, yawrate);
        trajectory[i] = state;
    }
    return trajectory;
}

static Cost evaluate_trajectory(const std::vector<State>& trajectory, const Vector3d& goal) {
    Cost cost;
    cost.to_goal_cost_ = calc_to_goal_cost(trajectory, goal);
    cost.obs_cost_ = calc_obs_cost(trajectory);
    cost.speed_cost_ = calc_speed_cost(trajectory);
    cost.path_cost_ = 0.0;
    return cost;
}

}  // namespace ams

// ── the harness ─────────────────────────────────────────────────────
double bench_amslabtech(int side, int reps, int n_obstacles) {
    using namespace ams;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(0.0, 6.0);
    obs_list_.poses.clear();
    for (int i = 0; i < n_obstacles; i++) obs_list_.poses.push_back(Pose{Point{U(rng), U(rng), 0.0}});
    current_cmd_vel_.linear.x = 0.25;
    current_cmd_vel_.angular.z = 0.0;
    sim_time_samples_ = (int)(predict_time_ / 0.1);

    // the same window this repo is given, sampled the same number of times
    const Vector3d goal(5.0, 5.0, 0.0);
    double best_total = 0.0;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) {
        double best = 1e18;
        for (int i = 0; i < side; i++) {
            const double v = target_velocity_ * i / std::max(1, side - 1);
            for (int j = 0; j < side; j++) {
                const double w = -max_yawrate_ + 2.0 * max_yawrate_ * j / std::max(1, side - 1);
                std::vector<State> traj = generate_trajectory(v, w);
                Cost c = evaluate_trajectory(traj, goal);
                c.calc_total_cost();
                if (c.total_cost_ < best) best = c.total_cost_;
            }
        }
        best_total += best;
    }
    auto t1 = std::chrono::steady_clock::now();
    (void)best_total;
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

// Its own generate_trajectory and evaluate_trajectory in closed loop, with
// the field transformed into the robot frame each tick.
//
// The transform is not a liberty: generate_trajectory starts every rollout
// from a zero State and calc_obs_cost measures against obs_list_, so its
// trajectories, its obstacles and its goal are all in the base frame -- which
// is where its node gets them, from a scan.  Feeding it map-frame obstacles
// would have it avoiding things thirteen metres behind itself.
//
// Held to the same plant as the others: max_acceleration_ and max_d_yawrate_
// are its own 0.4 and 1.0, and target_velocity_ its own 0.5, and all three are
// overwritten here, or its window would be a fifth as wide as everyone
// else's.  robot_radius_ likewise: its 0.17 plus 0.01 of padding against the
// 0.47 the others are given.
void trace_amslabtech(const TraceIn& in, TraceOut& out) {
    using namespace ams;
    sim_time_samples_ = in.steps;
    predict_time_ = in.steps * in.dt;
    sim_period_ = in.dt;
    target_velocity_ = in.top_speed;
    min_velocity_ = 0.0;
    max_yawrate_ = in.max_yaw;
    max_acceleration_ = in.acc_v;
    max_deceleration_ = in.acc_v;
    max_d_yawrate_ = in.acc_w;
    robot_radius_ = in.radius - footprint_padding_;

    TraceState s{in.sx, in.sy, in.syaw, 0.0, 0.0};
    std::vector<double> ms, rolls;
    out.xy = { s.x, s.y };
    for (int k = 0; k < in.max_steps; k++) {
        const double c = std::cos(-s.yaw), sn = std::sin(-s.yaw);
        obs_list_.poses.clear();
        for (int i = 0; i < in.nob; i++) {
            const double dx = in.obx[i] - s.x, dy = in.oby[i] - s.y;
            obs_list_.poses.push_back(
                Pose{Point{dx * c - dy * sn, dx * sn + dy * c, 0.0}});
        }
        const double gdx = in.gx - s.x, gdy = in.gy - s.y;
        const Vector3d goal(gdx * c - gdy * sn, gdx * sn + gdy * c, 0.0);
        current_cmd_vel_.linear.x = s.v;
        current_cmd_vel_.angular.z = s.w;

        const Window dw = calc_dynamic_window();
        double bv = 0.0, bw = 0.0, best = 1e18;
        int n = 0;
        auto t0 = std::chrono::steady_clock::now();
        for (double v = dw.min_velocity_; v <= dw.max_velocity_; v += in.vres) {
            for (double w = dw.min_yawrate_; w <= dw.max_yawrate_; w += in.wres) {
                std::vector<State> traj = generate_trajectory(v, w);
                Cost cost = evaluate_trajectory(traj, goal);
                cost.calc_total_cost();
                n++;
                if (cost.total_cost_ < best) { best = cost.total_cost_; bv = v; bw = w; }
            }
        }
        ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count());
        rolls.push_back((double)n);
        // Its node's own recovery when nothing is admissible: turn in place.
        // Without it a run that meets an obstacle head on stops there, and
        // the trail would report a stall that is the harness leaving out half
        // the node rather than anything about its scoring.
        if (best >= 1e6) { bv = 0.0; bw = in.max_yaw; }
        s = trace_step(s, bv, bw, in);
        out.xy.push_back(s.x);
        out.xy.push_back(s.y);
        if (trace_arrived(s, in)) break;
    }
    out.ms = trace_median(ms);
    out.rolls = trace_median(rolls);
}
