// goktug97/DynamicWindowApproach, isolated: its header defines max/min as
// macros and its Config/Point clash with the other baseline's.
extern "C" {
#include ".third_party/dwa.h"
}
#undef max
#undef min
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>

double bench_goktug(int side, int reps, int n_obstacles) {
    Config gc;
    gc.maxSpeed = 0.5f; gc.minSpeed = 0.0f; gc.maxYawrate = 2.0f;
    gc.maxAccel = 1e6f; gc.maxdYawrate = 1e6f;
    gc.velocityResolution = 0.5f / (side - 1);
    gc.yawrateResolution  = 4.0f / (side - 1);
    gc.dt = 0.1f; gc.predictTime = 2.5f;
    gc.heading = 5.0f; gc.clearance = 5.0f; gc.velocity = 0.5f;
    gc.base.xmin = -0.2f; gc.base.ymin = -0.2f;
    gc.base.xmax = 0.2f;  gc.base.ymax = 0.2f;
    Pose gp = {{1.0f, 1.0f}, 0.0f};
    Velocity gv = {0.25f, 0.0f};
    Point ggoal = {5.0f, 5.0f};
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(0.0, 10.0);
    PointCloud* pc = createPointCloud(n_obstacles);
    for (int i = 0; i < n_obstacles; i++) {
        pc->points[i].x = (float)U(rng); pc->points[i].y = (float)U(rng);
    }
    planning(gp, gv, ggoal, pc, gc);
    std::vector<double> t;
    for (int r = 0; r < reps; r++) {
        auto t0 = std::chrono::steady_clock::now();
        planning(gp, gv, ggoal, pc, gc);
        t.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count());
    }
    freePointCloud(pc);
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}
