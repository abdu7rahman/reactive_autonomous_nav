// Times this repo's C++ DWA rollout against CppRobotics' implementation.
//
// Baseline: onlytailei/CppRobotics src/dynamic_window_approach.cpp, the C++
// counterpart of PythonRobotics. Its planner functions are used verbatim --
// only the OpenCV visualisation and main() are stripped, since neither is part
// of the planning work.
//
// Both are given the same dynamic window, the same sample count and the same
// horizon. This repo evaluates obstacles through an O(1) costmap lookup; the
// baseline measures every rollout point against an explicit obstacle list.

int count_cpprobotics(int side);
int count_goktug(int side);
int count_amslabtech(int side);

#include "trace.h"

#include <unistd.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static constexpr uint8_t LETHAL_COST = 253;
static constexpr uint8_t WARN_COST   = 80;
static constexpr double  WP_TOL      = 0.25;
static int NW = 400, NH = 400;
static double RES = 0.05, OX = 0.0, OY = 0.0;
static std::vector<int8_t> MAP;

struct RS { double x, y, yaw, v, w; };
static double dt_ = 0.1, predict_time_ = 2.5, max_vel_ = 0.5, max_yawrate_ = 2.0;
static double heading_gain_ = 5.0, obstacle_gain_ = 5.0, speed_gain_ = 0.5;

static inline RS mine_motion(const RS& s, double v, double w) {
    return { s.x + v * std::cos(s.yaw) * dt_, s.y + v * std::sin(s.yaw) * dt_,
             s.yaw + w * dt_, v, w };
}
static inline uint8_t costmap_lookup(double wx, double wy) {
    int gx = (int)((wx - OX) / RES), gy = (int)((wy - OY) / RES);
    if (gx < 0 || gy < 0 || gx >= NW || gy >= NH) return 0;
    return (uint8_t)MAP[(size_t)gy * NW + gx];
}
static inline double wrap(double a) {
    while (a >  M_PI) a -= 2 * M_PI;
    while (a < -M_PI) a += 2 * M_PI;
    return a;
}

// The sampler from cpp/src/dwa_controller.cpp, verbatim: a lattice of
// multiples of res with both bounds included, rather than accumulating the
// resolution from the lower bound.
static std::vector<double> samples(double lo, double hi, double res) {
    if (hi <= lo) return { lo };
    std::vector<double> out;
    const long k0 = (long)std::ceil(lo / res - 1e-9);
    const long k1 = (long)std::floor(hi / res + 1e-9);
    if (k0 > k1 || k0 * res > lo + 1e-9) out.push_back(lo);
    for (long k = k0; k <= k1; k++) out.push_back(k * res);
    if (out.back() < hi - 1e-9) out.push_back(hi);
    return out;
}

// The scoring from cpp/src/dwa_controller.cpp.  It used to be the older form
// -- heading_gain * (pi - yaw_err) + obstacle_gain * clear + speed_gain *
// (v / max_vel), read at the last rollout sample -- which the controller
// replaced with the three normalised terms below when it was found crawling at
// 0.10 to 0.14 m/s with a 0.46 m/s limit: the raw clearance margin, up to 253
// and added, outweighed a heading term worth at most pi, and the bearing read
// at the endpoint inverts whenever the waypoint is nearer than the rollout
// reaches.  This file kept timing the old form, so the "this repo" column was
// timing a function the controller no longer has.  Per trajectory the new one
// does more work, not less: a per-step penalty accumulation and a WP_TOL
// distance check against the old one's single min().
// Precomputed once rather than per step.
static constexpr double PEN_SCALE = 1.0 / (double)(LETHAL_COST - WARN_COST);
static constexpr double WP_TOL2   = WP_TOL * WP_TOL;
static double INV_RES = 20.0;      // 1 / RES, set in main and trace_main

static inline uint8_t costmap_lookup_fast(double wx, double wy) {
    const int gx = (int)((wx - OX) * INV_RES), gy = (int)((wy - OY) * INV_RES);
    if (gx < 0 || gy < 0 || gx >= NW || gy >= NH) return 0;
    return (uint8_t)MAP[(size_t)gy * NW + gx];
}

// The same scoring, with the arithmetic that does not have to be in the loop
// taken out of it.  Four changes, all of them exact or measured:
//
//   - cos and sin of the heading are advanced by one rotation rather than
//     recomputed: for a constant w the heading turns by w * dt every step, so
//     (c, s) <- (c cw - s sw, s cw + c sw) with cw and sw computed once per
//     trajectory.  That removes 2 transcendental calls per step -- 50 per
//     trajectory at a 25-step horizon -- and is algebraically the same number;
//     mine_pick_check() measures the drift it actually costs.
//   - the arrival test compares squared distances instead of calling hypot.
//   - the inflation penalty multiplies by 1 / (LETHAL - WARN) instead of
//     dividing by it per step.
//   - the costmap index multiplies by 1 / resolution instead of dividing.
static double mine_pick_fast(const RS& s, double lax, double lay,
                             const std::vector<double>& vs,
                             const std::vector<double>& ws,
                             double* out_v, double* out_w) {
    const int steps = (int)(predict_time_ / dt_);
    double best = -1e18, bv = 0.0, bw = 0.0;
    const double c0 = std::cos(s.yaw), s0 = std::sin(s.yaw);
    for (double w : ws) {
        const double cw = std::cos(w * dt_), sw = std::sin(w * dt_);
        for (double v : vs) {
            const double step = v * dt_;
            double x = s.x, y = s.y, yaw = s.yaw, c = c0, sn = s0;
            double ax = s.x, ay = s.y, ayaw = s.yaw;
            bool collision = false, arrived = false;
            double pen_sum = 0.0;
            int kept = 0;
            for (int i = 0; i < steps; i++) {
                x += step * c;
                y += step * sn;
                const double nc = c * cw - sn * sw;
                sn = sn * cw + c * sw;
                c = nc;
                yaw += w * dt_;
                const uint8_t cost = costmap_lookup_fast(x, y);
                if (cost >= LETHAL_COST) { collision = true; break; }
                if (cost > WARN_COST)
                    pen_sum += (double)(cost - WARN_COST) * PEN_SCALE;
                if (!arrived) {
                    const double dx = x - lax, dy = y - lay;
                    if (dx * dx + dy * dy <= WP_TOL2) {
                        ax = x; ay = y; ayaw = yaw; arrived = true;
                    }
                }
                kept++;
            }
            if (collision || !kept) continue;
            if (!arrived) { ax = x; ay = y; ayaw = yaw; }
            const double diff = wrap(std::atan2(lay - ay, lax - ax) - ayaw);
            const double total = heading_gain_ * (1.0 - std::abs(diff) / M_PI)
                               + speed_gain_ * (v / max_vel_)
                               - obstacle_gain_ * std::min(10.0,
                                     pen_sum / steps * 10.0);
            if (total > best) { best = total; bv = v; bw = w; }
        }
    }
    if (out_v) *out_v = bv;
    if (out_w) *out_w = bw;
    return best;
}

static double mine_pick(const RS& s, double lax, double lay,
                        const std::vector<double>& vs,
                        const std::vector<double>& ws,
                        double* out_v, double* out_w) {
    const int steps = (int)(predict_time_ / dt_);
    double best = -1e18;
    double bv = 0.0, bw = 0.0;
    for (double v : vs) {
        for (double w : ws) {
            RS cur = s, arrive = s;
            bool collision = false, arrived = false;
            double pen_sum = 0.0;
            int kept = 0;
            for (int i = 0; i < steps; i++) {
                cur = mine_motion(cur, v, w);
                uint8_t cost = costmap_lookup(cur.x, cur.y);
                if (cost >= LETHAL_COST) { collision = true; break; }
                if (cost > WARN_COST)
                    pen_sum += (double)(cost - WARN_COST)
                             / (double)(LETHAL_COST - WARN_COST);
                if (!arrived && std::hypot(cur.x - lax, cur.y - lay) <= WP_TOL) {
                    arrive = cur; arrived = true;
                }
                kept++;
            }
            if (collision || !kept) continue;
            if (!arrived) arrive = cur;
            double diff = wrap(std::atan2(lay - arrive.y, lax - arrive.x)
                               - arrive.yaw);
            double h_score = 1.0 - std::abs(diff) / M_PI;
            double o_cost  = std::min(10.0, pen_sum / steps * 10.0);
            double s_score = v / max_vel_;
            double total   = heading_gain_ * h_score + speed_gain_ * s_score
                           - obstacle_gain_ * o_cost;
            if (total > best) { best = total; bv = v; bw = w; }
        }
    }
    if (out_v) *out_v = bv;
    if (out_w) *out_w = bw;
    return best;
}

// The optimised sweep has to choose what the plain one chooses.  Run over a
// grid of states and lookahead points, reporting the worst disagreement in the
// chosen command and in the score, so the rotation-increment drift is a
// measured number rather than an argument that it must be small.
static int mine_pick_check() {
    const std::vector<double> vs = samples(0.0, max_vel_, max_vel_ / 19);
    const std::vector<double> ws = samples(-max_yawrate_, max_yawrate_,
                                           2 * max_yawrate_ / 19);
    double worst_v = 0.0, worst_w = 0.0, worst_score = 0.0;
    int cases = 0, disagree = 0;
    for (double x = 1.0; x <= 17.0; x += 2.0)
      for (double y = 1.0; y <= 17.0; y += 2.0)
        for (double yaw = -M_PI; yaw < M_PI; yaw += M_PI / 4)
          for (double d = 0.4; d <= 2.0; d += 0.8) {
            const RS s{x, y, yaw, 0.25, 0.0};
            const double lax = x + d * std::cos(yaw + 0.3);
            const double lay = y + d * std::sin(yaw + 0.3);
            double av = 0, aw = 0, bv = 0, bw = 0;
            const double sa = mine_pick(s, lax, lay, vs, ws, &av, &aw);
            const double sb = mine_pick_fast(s, lax, lay, vs, ws, &bv, &bw);
            cases++;
            if (av != bv || aw != bw) disagree++;
            worst_v = std::max(worst_v, std::abs(av - bv));
            worst_w = std::max(worst_w, std::abs(aw - bw));
            if (std::isfinite(sa) && std::isfinite(sb))
                worst_score = std::max(worst_score, std::abs(sa - sb));
          }
    printf("\n  optimised sweep against the plain one, %d states:\n", cases);
    printf("    commands differing %d, worst dv %.3g m/s, worst dw %.3g rad/s, "
           "worst score %.3g\n", disagree, worst_v, worst_w, worst_score);
    return disagree;
}

// What this repo's own sweep evaluates, for the same report the baselines give.
static void mine_work(int side, BenchWork* w) {
    RS s{1.0, 1.0, 0.0, 0.0, 0.0};
    const int steps = (int)(predict_time_ / dt_);
    const std::vector<double> vs = samples(0.0, max_vel_, max_vel_ / (side - 1));
    const std::vector<double> ws = samples(-max_yawrate_, max_yawrate_,
                                           2 * max_yawrate_ / (side - 1));
    long long pts = 0, bail = 0, traj = 0;
    for (double v : vs) for (double wr : ws) {
        RS cur = s;
        traj++;
        for (int i = 0; i < steps; i++) {
            cur = mine_motion(cur, v, wr);
            pts++;
            if (costmap_lookup(cur.x, cur.y) >= LETHAL_COST) { bail++; break; }
        }
    }
    w->points = (double)pts / traj;
    w->checks = (double)pts / traj;     // one costmap cell per point
    w->bailed = (double)bail / traj;
}

// The table's entry point: one full window, scored, with the trajectory count
// returned so the row is labelled by what was actually evaluated.
static int mine_sweep(int side, double lax, double lay) {
    RS s{1.0, 1.0, 0.0, 0.25, 0.0};
    const std::vector<double> vs = samples(0.0, max_vel_,
                                           max_vel_ / (side - 1));
    const std::vector<double> ws = samples(-max_yawrate_, max_yawrate_,
                                           2 * max_yawrate_ / (side - 1));
    mine_pick_fast(s, lax, lay, vs, ws, nullptr, nullptr);
    return (int)(vs.size() * ws.size());
}

// The form this replaced, kept so the table can show what the four changes
// bought rather than asserting it.
static int mine_sweep_plain(int side, double lax, double lay) {
    RS s{1.0, 1.0, 0.0, 0.25, 0.0};
    const std::vector<double> vs = samples(0.0, max_vel_,
                                           max_vel_ / (side - 1));
    const std::vector<double> ws = samples(-max_yawrate_, max_yawrate_,
                                           2 * max_yawrate_ / (side - 1));
    mine_pick(s, lax, lay, vs, ws, nullptr, nullptr);
    return (int)(vs.size() * ws.size());
}

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

// ── the trace mode ─────────────────────────────────────────────────────────
//
// bench/gif_compare.py writes the field, the route and the costmap and calls
// this; one file both sides read beats two random number generators agreeing.
static bool load_field(const char* path, TraceIn& in,
                       std::vector<double>& obx, std::vector<double>& oby,
                       std::vector<double>& rx, std::vector<double>& ry,
                       std::vector<int8_t>& map) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot read %s\n", path); return false; }
    std::string key;
    while (f >> key) {
        if (key == "obstacles") {
            int n; f >> n;
            obx.resize(n); oby.resize(n);
            for (int i = 0; i < n; i++) f >> obx[i] >> oby[i];
        } else if (key == "route") {
            int n; f >> n;
            rx.resize(n); ry.resize(n);
            for (int i = 0; i < n; i++) f >> rx[i] >> ry[i];
        } else if (key == "map") {
            std::string bin; f >> in.nw >> in.nh >> bin;
            map.assign((size_t)in.nw * in.nh, 0);
            std::ifstream b(bin, std::ios::binary);
            if (!b) { fprintf(stderr, "cannot read %s\n", bin.c_str()); return false; }
            b.read((char*)map.data(), (std::streamsize)map.size());
        } else {
            double v; f >> v;
            if (key == "dt") in.dt = v;
            else if (key == "horizon") in.steps = (int)v;
            else if (key == "res") in.res = v;
            else if (key == "radius") in.radius = v;
            else if (key == "top_speed") in.top_speed = v;
            else if (key == "max_yaw") in.max_yaw = v;
            else if (key == "acc_v") in.acc_v = v;
            else if (key == "acc_w") in.acc_w = v;
            else if (key == "vel_res") in.vres = v;
            else if (key == "yaw_res") in.wres = v;
            else if (key == "carrot") in.carrot = v;
            else if (key == "wp_tol") in.wp_tol = v;
            else if (key == "goal_tol") in.goal_tol = v;
            else if (key == "max_steps") in.max_steps = (int)v;
            else if (key == "start_x") in.sx = v;
            else if (key == "start_y") in.sy = v;
            else if (key == "start_yaw") in.syaw = v;
            else if (key == "goal_x") in.gx = v;
            else if (key == "goal_y") in.gy = v;
        }
    }
    in.obx = obx.data(); in.oby = oby.data(); in.nob = (int)obx.size();
    in.rx = rx.data();   in.ry = ry.data();   in.nroute = (int)rx.size();
    in.map = map.data();
    return in.nob > 0 && in.nroute > 0 && !map.empty();
}

// This repo, in closed loop on its own route: the same lookahead rule as
// cpp/src/dwa_controller.cpp's control_loop -- a monotonic waypoint index,
// the target carrot metres along by arc length, walked back to the furthest
// one whose straight segment is clear -- and mine_pick for the scoring.
static bool segment_clear(double x0, double y0, double x1, double y1,
                          const TraceIn& in) {
    const double step = in.res * 0.5;
    const int n = std::max(2, (int)(std::hypot(x1 - x0, y1 - y0) / step) + 1);
    for (int i = 0; i <= n; i++) {
        const double t = (double)i / n;
        if (costmap_lookup(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t) >= LETHAL_COST)
            return false;
    }
    return true;
}

static void trace_mine(const TraceIn& in, TraceOut& out) {
    std::vector<double> run(in.nroute, 0.0);
    for (int i = 1; i < in.nroute; i++)
        run[i] = run[i - 1] + std::hypot(in.rx[i] - in.rx[i - 1],
                                         in.ry[i] - in.ry[i - 1]);
    TraceState s{in.sx, in.sy, in.syaw, 0.0, 0.0};
    std::vector<double> ms, rolls;
    out.xy = { s.x, s.y };
    int wp = 0;
    for (int k = 0; k < in.max_steps; k++) {
        while (wp < in.nroute - 1) {
            const double d = std::hypot(in.rx[wp] - s.x, in.ry[wp] - s.y);
            if (d < in.wp_tol ||
                std::hypot(in.rx[wp + 1] - s.x, in.ry[wp + 1] - s.y) < d) wp++;
            else break;
        }
        int j = wp;
        while (j < in.nroute - 1 && run[j] < run[wp] + in.carrot) j++;
        while (j > wp && !segment_clear(s.x, s.y, in.rx[j], in.ry[j], in)) j--;

        const double v_min = std::max(0.0, s.v - in.acc_v * in.dt);
        const double v_max = std::min(in.top_speed, s.v + in.acc_v * in.dt);
        const double w_min = std::max(-in.max_yaw, s.w - in.acc_w * in.dt);
        const double w_max = std::min( in.max_yaw, s.w + in.acc_w * in.dt);
        const std::vector<double> vs = samples(v_min, v_max, in.vres);
        const std::vector<double> ws = samples(w_min, w_max, in.wres);
        rolls.push_back((double)(vs.size() * ws.size()));
        RS rs{s.x, s.y, s.yaw, s.v, s.w};
        double bv = 0.0, bw = 0.0;
        auto t0 = std::chrono::steady_clock::now();
        // The optimised form, which is what cpp/src/dwa_controller.cpp runs;
        // the plain one is kept only to measure what the change bought.
        mine_pick_fast(rs, in.rx[j], in.ry[j], vs, ws, &bv, &bw);
        ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count());
        s = trace_step(s, bv, bw, in);
        out.xy.push_back(s.x);
        out.xy.push_back(s.y);
        if (trace_arrived(s, in)) break;
    }
    out.ms = trace_median(ms);
    out.rolls = trace_median(rolls);
}

static int sweep_goktug_main(const char* path) {
    TraceIn in;
    std::vector<double> obx, oby, rx, ry;
    std::vector<int8_t> map;
    if (!load_field(path, in, obx, oby, rx, ry, map)) return 1;
    printf("  %9s %10s %10s %10s %s\n", "clearance", "halfwidth", "driven",
           "ms/tick", "outcome");
    for (float half : {(float)in.radius, (float)(in.radius * 0.886),
                       (float)(in.radius / std::sqrt(2.0))})
    for (float g : {5.0f, 1.0f, 0.5f, 0.2f, 0.1f, 0.05f}) {
        trace_goktug_gain(g);
        trace_goktug_footprint(half);
        TraceOut out;
        trace_goktug(in, out);
        double drove = 0.0;
        for (size_t i = 2; i + 1 < out.xy.size(); i += 2)
            drove += std::hypot(out.xy[i] - out.xy[i - 2],
                                out.xy[i + 1] - out.xy[i - 1]);
        const double left = std::hypot(out.xy[out.xy.size() - 2] - in.gx,
                                       out.xy[out.xy.size() - 1] - in.gy);
        printf("  %9.2f %9.3f m %8.2f m %8.3f ms  %s\n", g, half, drove,
               out.ms, left < in.goal_tol ? "arrived" : "did not arrive");
    }
    trace_goktug_gain(0.2f);
    trace_goktug_footprint(-1.0f);
    return 0;
}

static int trace_main(const char* path) {
    TraceIn in;
    std::vector<double> obx, oby, rx, ry;
    std::vector<int8_t> map;
    if (!load_field(path, in, obx, oby, rx, ry, map)) return 1;

    // The globals mine_pick reads, from the field rather than from this
    // file's own defaults.
    NW = in.nw; NH = in.nh; RES = in.res; OX = 0.0; OY = 0.0;
    INV_RES = 1.0 / in.res;
    MAP = map;
    dt_ = in.dt; predict_time_ = in.steps * in.dt;
    max_vel_ = in.top_speed; max_yawrate_ = in.max_yaw;

    struct Entry { const char* name; void (*fn)(const TraceIn&, TraceOut&); };
    const Entry entries[] = {{"this_repo",   trace_mine},
                             {"CppRobotics", trace_cpprobotics},
                             {"goktug97",    trace_goktug},
                             {"amslabtech",  trace_amslabtech}};
    for (const Entry& e : entries) {
        TraceOut out;
        e.fn(in, out);
        printf("trace %s %.4f %.0f ", e.name, out.ms, out.rolls);
        for (size_t i = 0; i < out.xy.size(); i++)
            printf("%s%.4f", i ? "," : "", out.xy[i]);
        printf("\n");
    }
    return 0;
}

int main(int argc, char** argv) {
  _sig();
    if (argc > 2 && std::string(argv[1]) == "--trace") return trace_main(argv[2]);
    if (argc > 2 && std::string(argv[1]) == "--sweep-goktug")
        return sweep_goktug_main(argv[2]);
    const int reps = argc > 1 ? atoi(argv[1]) : 15;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    // One field for everyone.  60 blocks of 10 x 10 cells -- 0.5 m square at
    // 0.05 m -- in the 20 x 20 m map, rasterised for this repo and handed to
    // the baselines as their centres with a radius of half a block.  The start
    // pose is kept clear of all of them, because a start inside an obstacle is
    // what made every baseline row a first-point collision.
    const int N_BLOCKS = 60;
    const double BLOCK = 0.25;                   // half a block, in metres
    std::vector<double> obx, oby;
    MAP.assign((size_t)NW * NH, 0);
    while ((int)obx.size() < N_BLOCKS) {
        const int r = (int)(U(rng) * (NH - 20)), col = (int)(U(rng) * (NW - 20));
        const double cx = (col + 5) * RES, cy = (r + 5) * RES;
        if (std::hypot(cx - 1.0, cy - 1.0) < 1.5) continue;   // clear of start
        for (int a = 0; a < 10; a++) for (int b = 0; b < 10; b++)
            MAP[(size_t)(r + a) * NW + (col + b)] = (int8_t)254;
        obx.push_back(cx);
        oby.push_back(cy);
    }
    BenchField field;
    field.obx = obx.data(); field.oby = oby.data();
    field.nob = (int)obx.size();
    field.radius = BLOCK;
    field.sx = 1.0; field.sy = 1.0; field.syaw = 0.0;
    field.gx = 15.0; field.gy = 15.0;
    field.max_speed = max_vel_; field.max_yaw = max_yawrate_;
    field.dt = dt_; field.predict_time = predict_time_;

    INV_RES = 1.0 / RES;
    if (mine_pick_check() != 0) {
        printf("  the optimised sweep disagrees with the plain one; "
               "not publishing a table off it\n");
        return 1;
    }

    // What the four changes bought, on the same window and the same field.
    printf("\n  %13s %16s %16s %10s\n", "trajectories", "plain", "optimised",
           "speedup");
    for (int side : {6, 20, 50}) {
        std::vector<double> p, q;
        int n = 0;
        mine_sweep_plain(side, 5.0, 5.0);
        mine_sweep(side, 5.0, 5.0);
        for (int r = 0; r < reps; r++) {
            auto t0 = std::chrono::steady_clock::now();
            n = mine_sweep_plain(side, 5.0, 5.0);
            auto t1 = std::chrono::steady_clock::now();
            mine_sweep(side, 5.0, 5.0);
            auto t2 = std::chrono::steady_clock::now();
            p.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            q.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        }
        std::sort(p.begin(), p.end());
        std::sort(q.begin(), q.end());
        printf("  %13d %13.3fms %13.3fms %9.2fx\n", n, p[p.size()/2],
               q[q.size()/2], p[p.size()/2] / q[q.size()/2]);
    }

    // What each one evaluates per trajectory.  This is printed rather than
    // assumed because the comparison was lopsided for exactly this reason and
    // nothing in the table would have shown it.
    {
        BenchWork a, b, cc, d;
        mine_work(20, &a);
        bench_cpprobotics(20, 1, field, &b);
        bench_goktug(20, 1, field, &cc);
        bench_amslabtech(20, 1, field, &d);
        printf("\n  work per trajectory, side 20, %d obstacles, "
               "%.2f m clearance\n", field.nob, field.radius);
        printf("  %14s %12s %12s %12s\n", "", "points", "checks", "bailed");
        const char* n[] = {"this repo", "CppRobotics", "goktug97", "amslabtech"};
        const BenchWork* ws[] = {&a, &b, &cc, &d};
        for (int i = 0; i < 4; i++)
            printf("  %14s %12.1f %12.1f %11.0f%%\n", n[i], ws[i]->points,
                   ws[i]->checks, 100 * ws[i]->bailed);
    }

    printf("\n  trajectories each one evaluates for the same window:\n");
    printf("  %13s %14s %16s %16s %16s\n", "side", "this repo",
           "CppRobotics", "goktug97 (C)", "amslabtech");
    for (int side : {6, 10, 20, 30, 50})
        printf("  %13d %14d %16d %16d %16d\n", side,
               mine_sweep(side, 5.0, 5.0), count_cpprobotics(side),
               count_goktug(side), count_amslabtech(side));

    printf("\n  %13s %14s %16s %16s %16s\n", "trajectories", "this repo",
           "CppRobotics", "goktug97 (C)", "amslabtech");
    for (int side : {6, 10, 20, 30, 50}) {
        double lax = 5.0, lay = 5.0;
        mine_sweep(side, lax, lay);
        std::vector<double> a;
        int n = 0;
        for (int r = 0; r < reps; r++) {
            auto t0 = std::chrono::steady_clock::now();
            n = mine_sweep(side, lax, lay);
            a.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count());
        }
        double mb = bench_cpprobotics(side, reps, field, nullptr);
        double mc = bench_goktug(side, reps, field, nullptr);
        double md = bench_amslabtech(side, reps, field, nullptr);
        std::sort(a.begin(), a.end());
        double ma = a[a.size()/2];
        printf("  %13d %13.3fms %15.3fms %15.3fms %15.3fms\n", n, ma, mb, mc, md);
    }
    printf("\n");
    return 0;
}
