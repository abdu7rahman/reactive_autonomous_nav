// Times the C++ DWA rollout+score from cpp/src/dwa_controller.cpp.
// motion(), costmap_lookup(), the v/w sweep and the scoring terms are copied
// from the control loop; marker construction is left out so this measures the
// same work as the Python _score_trajectories it is compared against.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

static constexpr uint8_t LETHAL_COST = 253;
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

static int sweep(double v_min, double v_max, double w_min, double w_max,
                 double vel_res, double yaw_res, double lax, double lay,
                 double& best_v, double& best_w)
{
    RobotState s{0, 0, 0, 0.25, 0.0};
    int steps = (int)(predict_time_ / dt_);
    double best_score = -std::numeric_limits<double>::infinity();
    int n = 0;
    best_v = best_w = 0.0;

    for (double v = v_min; v <= v_max + 1e-9; v += vel_res) {
        for (double w = w_min; w <= w_max + 1e-9; w += yaw_res) {
            n++;
            RobotState cur = s;
            bool collision = false;
            double min_clearance = (double)LETHAL_COST;
            int kept = 0;

            for (int i = 0; i < steps; i++) {
                cur = motion(cur, v, w);
                uint8_t cost = costmap_lookup(cur.x, cur.y);
                if (cost >= LETHAL_COST) { collision = true; break; }
                min_clearance = std::min(min_clearance, (double)(LETHAL_COST - cost));
                kept++;
            }
            if (collision || kept == 0) continue;

            double target_yaw = std::atan2(lay - cur.y, lax - cur.x);
            double yaw_err    = std::abs(angle_wrap(target_yaw - cur.yaw));
            double total = heading_gain_  * (M_PI - yaw_err)
                         + obstacle_gain_ * min_clearance
                         + speed_gain_    * (v / max_vel_);
            if (total > best_score) { best_score = total; best_v = v; best_w = w; }
        }
    }
    return n;
}

int main(int argc, char** argv)
{
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

    const double VEL_RES = 0.02, YAW_RES = 0.04, MAX_ACC = 0.40, MAX_DYAW = 1.00;
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
