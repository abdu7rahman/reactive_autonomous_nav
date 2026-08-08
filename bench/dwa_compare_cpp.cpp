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

double bench_cpprobotics(int side, int reps, int n_obstacles);
double bench_goktug(int side, int reps, int n_obstacles);
double bench_amslabtech(int side, int reps, int n_obstacles);

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static constexpr uint8_t LETHAL_COST = 253;
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

// the sweep from cpp/src/dwa_controller.cpp
static int mine_sweep(int side, double lax, double lay) {
    RS s{1.0, 1.0, 0.0, 0.25, 0.0};
    int steps = (int)(predict_time_ / dt_), n = 0;
    double best = -1e18, bv = 0, bw = 0;
    for (int i = 0; i < side; i++) {
        double v = max_vel_ * i / (side - 1);
        for (int j = 0; j < side; j++) {
            double w = -max_yawrate_ + 2 * max_yawrate_ * j / (side - 1);
            n++;
            RS cur = s; bool hit = false; double clear = LETHAL_COST; int kept = 0;
            for (int k = 0; k < steps; k++) {
                cur = mine_motion(cur, v, w);
                uint8_t c = costmap_lookup(cur.x, cur.y);
                if (c >= LETHAL_COST) { hit = true; break; }
                clear = std::min(clear, (double)(LETHAL_COST - c));
                kept++;
            }
            if (hit || !kept) continue;
            double yaw_err = std::abs(wrap(std::atan2(lay - cur.y, lax - cur.x) - cur.yaw));
            double tot = heading_gain_ * (M_PI - yaw_err) + obstacle_gain_ * clear
                       + speed_gain_ * (v / max_vel_);
            if (tot > best) { best = tot; bv = v; bw = w; }
        }
    }
    (void)bv; (void)bw;
    return n;
}

int main(int argc, char** argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 15;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    MAP.assign((size_t)NW * NH, 0);
    for (int k = 0; k < 60; k++) {
        int r = (int)(U(rng) * (NH - 20)), c = (int)(U(rng) * (NW - 20));
        for (int a = 0; a < 10; a++) for (int b = 0; b < 10; b++)
            MAP[(size_t)(r + a) * NW + (c + b)] = (int8_t)254;
    }

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
        double mb = bench_cpprobotics(side, reps, 60);
        double mc = bench_goktug(side, reps, 60);
        double md = bench_amslabtech(side, reps, 60);
        std::sort(a.begin(), a.end());
        double ma = a[a.size()/2];
        printf("  %13d %13.3fms %15.3fms %15.3fms %15.3fms\n", n, ma, mb, mc, md);
    }
    printf("\n");
    return 0;
}
