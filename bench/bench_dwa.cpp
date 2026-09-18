// Times the C++ DWA rollout+score from cpp/src/dwa_controller.cpp.
// motion(), costmap_lookup(), the v/w sweep and the scoring terms are copied
// from the control loop; marker construction is left out so this measures the
// same work as the Python _score_trajectories it is compared against.

#include <unistd.h>
#include <string>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

static constexpr uint8_t LETHAL_COST = 253;
static constexpr uint8_t WARN_COST   = 80;
static constexpr double  WP_TOL      = 0.25;
static constexpr double  WP_TOL2     = WP_TOL * WP_TOL;
static constexpr double  PEN_SCALE   = 1.0 / (double)(LETHAL_COST - WARN_COST);
struct RobotState { double x, y, yaw, v, w; };

static int    NW, NH;
static double RES, OX, OY;
static std::vector<int8_t> MAP;

static double dt_ = 0.1, predict_time_ = 2.5;
static double max_vel_ = 0.50, max_yawrate_ = 2.00;
static double heading_gain_ = 5.0, obstacle_gain_ = 5.0, speed_gain_ = 0.5;

static inline RobotState motion(const RobotState& s, double v, double w)
{
    return { s.x + v * std::cos(s.yaw) * dt_,
             s.y + v * std::sin(s.yaw) * dt_,
             s.yaw + w * dt_, v, w };
}

static inline uint8_t costmap_lookup(double wx, double wy)
{
    int gx = (int)((wx - OX) / RES);
    int gy = (int)((wy - OY) / RES);
    if (gx < 0 || gy < 0 || gx >= NW || gy >= NH) return 0;
    return static_cast<uint8_t>(MAP[(size_t)gy * NW + gx]);
}

static inline double angle_wrap(double a)
{
    while (a >  M_PI) a -= 2 * M_PI;
    while (a < -M_PI) a += 2 * M_PI;
    return a;
}

static std::vector<double> lattice(double lo, double hi, double res)
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

static int sweep(double v_min, double v_max, double w_min, double w_max,
                 double vel_res, double yaw_res, double lax, double lay,
                 double& best_v, double& best_w)
{
    RobotState s{0, 0, 0, 0.25, 0.0};
    int steps = (int)(predict_time_ / dt_);
    double best_score = -std::numeric_limits<double>::infinity();
    int n = 0;
    best_v = best_w = 0.0;

    // Same lattice as the controllers: every multiple of the resolution
    // inside the window, with both bounds present.  Accumulating the
    // resolution from the lower bound admits one sample past the upper bound
    // and usually never evaluates the bound itself, so this was timing a
    // slightly different candidate set from the one it is compared against.
    const std::vector<double> vs = lattice(v_min, v_max, vel_res);
    const std::vector<double> ws = lattice(w_min, w_max, yaw_res);

    // This scored heading_gain * (pi - yaw_err) at the last rollout sample
    // plus obstacle_gain * the raw margin to lethal, which is the form
    // cpp/src/dwa_controller.cpp *replaced* -- so the Python-against-C++ row
    // was comparing the Python controller's current scoring against the C++
    // controller's old scoring. The third copy of this loop in the repo to
    // have drifted from the controller it measures, after mine_sweep in
    // dwa_compare_cpp.cpp and both harnesses' idea of the window.
    //
    // It is now the current scoring, and the current arithmetic with it: cos
    // and sin advanced by one rotation per step rather than recomputed, a
    // squared arrival test, and the penalty scaled by a precomputed
    // reciprocal.
    const double c0 = std::cos(s.yaw), s0 = std::sin(s.yaw);
    for (double w : ws) {
        const double cw = std::cos(w * dt_), sw = std::sin(w * dt_);
        for (double v : vs) {
            n++;
            const double adv = v * dt_;
            double px = s.x, py = s.y, pyaw = s.yaw, c = c0, sn = s0;
            double ax = s.x, ay = s.y, ayaw = s.yaw;
            bool collision = false, arrived = false;
            double pen_sum = 0.0;
            int kept = 0;

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
                kept++;
            }
            if (collision || kept == 0) continue;
            if (!arrived) { ax = px; ay = py; ayaw = pyaw; }

            const double diff = angle_wrap(std::atan2(lay - ay, lax - ax) - ayaw);
            const double total = heading_gain_ * (1.0 - std::abs(diff) / M_PI)
                               + speed_gain_ * (v / max_vel_)
                               - obstacle_gain_ * std::min(10.0,
                                     pen_sum / steps * 10.0);
            if (total > best_score) { best_score = total; best_v = v; best_w = w; }
        }
    }
    return n;
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

int main(int argc, char** argv)
{
  _sig();
    const char* path = argc > 1 ? argv[1] : "bench/local.bin";
    const int reps   = argc > 2 ? atoi(argv[2]) : 25;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    int32_t wh[2]; double meta[3];
    if (fread(wh, 4, 2, f) != 2 || fread(meta, 8, 3, f) != 3) return 1;
    NW = wh[0]; NH = wh[1]; RES = meta[0]; OX = meta[1]; OY = meta[2];
    MAP.resize((size_t)NW * NH);
    if (fread(MAP.data(), 1, MAP.size(), f) != MAP.size()) return 1;
    fclose(f);

    // The accelerations the controller actually carries, out of
    // irobot_create_control/config/control.yaml by way of
    // cpp/src/dwa_controller.cpp and dwa_controller.py.  They were 0.40
    // and 1.00 here, so the row labelled "accel-limited" was timing a
    // window 2.2 and 7.7 times narrower than the one the robot searches
    // -- fewer trajectories, and a quicker sweep, for a window nothing
    // uses.
    const double VEL_RES = 0.02, YAW_RES = 0.04, MAX_ACC = 0.90, MAX_DYAW = 7.725;
    struct Win { const char* name; double vmin, vmax, wmin, wmax; };
    Win wins[2] = {
        {"accel-limited", std::max(0.0, 0.25 - MAX_ACC * dt_), std::min(max_vel_, 0.25 + MAX_ACC * dt_),
                          std::max(-max_yawrate_, -MAX_DYAW * dt_), std::min(max_yawrate_, MAX_DYAW * dt_)},
        {"full velocity space", 0.0, max_vel_, -max_yawrate_, max_yawrate_},
    };

    printf("[\n");
    for (int k = 0; k < 2; k++) {
        double bv, bw;
        int n = sweep(wins[k].vmin, wins[k].vmax, wins[k].wmin, wins[k].wmax,
                      VEL_RES, YAW_RES, 2.5, 0.4, bv, bw);
        std::vector<double> ms;
        for (int r = 0; r < reps; r++) {
            auto t0 = std::chrono::steady_clock::now();
            sweep(wins[k].vmin, wins[k].vmax, wins[k].wmin, wins[k].wmax,
                  VEL_RES, YAW_RES, 2.5, 0.4, bv, bw);
            ms.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(ms.begin(), ms.end());
        printf(" {\"window\": \"%s\", \"N\": %d, \"T\": %d, \"ms\": %.4f, \"min_ms\": %.4f}%s\n",
               wins[k].name, n, (int)(predict_time_ / dt_), ms[ms.size()/2], ms.front(),
               k == 0 ? "," : "");
    }
    printf("]\n");
    return 0;
}
