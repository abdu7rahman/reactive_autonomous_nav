// One closed loop, one plant, for the four C++ implementations, so that
// bench/gif_compare.py --cpp draws what each of them does rather than only how
// long each of them takes.
//
// The timing table next door hands every implementation the same window and
// measures one call.  This drives them: same field, same start and goal, same
// plant and the same clamp on it, each one's own scoring function choosing its
// own commands.  Each baseline exposes one trace_* function from its own
// translation unit because their Config and Point types collide -- the same
// reason the bench_* functions live there.
#ifndef BENCH_TRACE_H
#define BENCH_TRACE_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

struct TraceIn {
    const double* obx = nullptr;   // obstacle centres
    const double* oby = nullptr;
    int nob = 0;
    const double* rx = nullptr;    // this repo's route; the baselines get the
    const double* ry = nullptr;    // goal, which is what each is written for
    int nroute = 0;
    const int8_t* map = nullptr;   // the inflated costmap, this repo's input
    int nw = 0, nh = 0;
    double res = 0.05, ox = 0.0, oy = 0.0;
    double sx = 0, sy = 0, syaw = 0, gx = 0, gy = 0;
    double dt = 0.1;
    int steps = 25;                // rollout samples per trajectory
    double top_speed = 0.5, max_yaw = 2.0, acc_v = 0.9, acc_w = 7.725;
    double vres = 0.02, wres = 0.04;
    double radius = 0.47;          // clearance, obstacle included
    double carrot = 0.4, wp_tol = 0.25, goal_tol = 0.4;
    int max_steps = 900;
};

struct TraceOut {
    std::vector<double> xy;        // x0,y0,x1,y1,... in the map frame
    double ms = 0.0;               // median per tick
    double rolls = 0.0;            // median trajectories scored per tick
};

// The plant, clamped.  Every implementation gets the same one, and the clamp
// is what keeps a recovery behaviour from commanding something the base could
// not do: goktug97's and amslabtech's both ask for a fixed turn rate when they
// stall, and PythonRobotics asks for its whole max_delta_yaw_rate.
struct TraceState { double x, y, yaw, v, w; };

inline TraceState trace_step(const TraceState& s, double v, double w,
                             const TraceIn& in) {
    v = std::fmin(std::fmax(v, -in.top_speed), in.top_speed);
    w = std::fmin(std::fmax(w, -in.max_yaw), in.max_yaw);
    return TraceState{ s.x + v * std::cos(s.yaw) * in.dt,
                       s.y + v * std::sin(s.yaw) * in.dt,
                       s.yaw + w * in.dt, v, w };
}

inline bool trace_arrived(const TraceState& s, const TraceIn& in) {
    return std::hypot(s.x - in.gx, s.y - in.gy) < in.goal_tol;
}

inline double trace_median(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    std::size_t k = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + k, v.end());
    return v[k];
}

void trace_cpprobotics(const TraceIn& in, TraceOut& out);
// goktug97 ships no default gains, so its clearance gain is a harness choice
// and is swept rather than assumed; see the note above trace_goktug.
void trace_goktug_gain(float gain);
void trace_goktug_footprint(float half_width);
void trace_goktug(const TraceIn& in, TraceOut& out);
void trace_amslabtech(const TraceIn& in, TraceOut& out);

#endif
