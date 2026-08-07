// CppRobotics DWA, isolated in its own translation unit because its Point and
// Config types collide with the other baseline's.
#include ".third_party/cpprobotics_core.h"
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>

double bench_cpprobotics(int side, int reps, int n_obstacles) {
    Config cfg;
    cfg.dt = 0.1f; cfg.predict_time = 2.5f;
    cfg.v_reso = (cfg.max_speed - cfg.min_speed) / (side - 1);
    cfg.yawrate_reso = (2 * cfg.max_yawrate) / (side - 1);
    cfg.max_accel = 1e6f; cfg.max_dyawrate = 1e6f;
    State x{{1.0f, 1.0f, 0.0f, 0.0f, 0.0f}};
    Point goal{{5.0f, 5.0f}};
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(0.0, 10.0);
    Obstacle ob;
    for (int i = 0; i < n_obstacles; i++) ob.push_back({{(float)U(rng), (float)U(rng)}});
    Window dw = calc_dynamic_window(x, cfg);
    { Control u{0, 0}; calc_final_input(x, u, dw, cfg, goal, ob); }
    std::vector<double> t;
    for (int r = 0; r < reps; r++) {
        Control u{0, 0};
        auto t0 = std::chrono::steady_clock::now();
        calc_final_input(x, u, dw, cfg, goal, ob);
        t.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}
