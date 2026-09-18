// CppRobotics DWA, isolated in its own translation unit because its Point and
// Config types collide with the other baseline's.
#include ".third_party/cpprobotics_core.h"
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>
#include "trace.h"

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

// Its own calc_dynamic_window and calc_final_input, in closed loop on the
// goal.  Its cost is the bearing to that goal, its own demo drives at one, and
// a target inside its rollout reach inverts the bearing -- so it gets the
// goal rather than a carrot on this repo's route.
void trace_cpprobotics(const TraceIn& in, TraceOut& out) {
    Config cfg;
    cfg.dt = (float)in.dt;
    cfg.predict_time = (float)(in.steps * in.dt);
    cfg.max_speed = (float)in.top_speed;
    cfg.min_speed = 0.0f;
    cfg.max_yawrate = (float)in.max_yaw;
    cfg.max_accel = (float)in.acc_v;
    cfg.max_dyawrate = (float)in.acc_w;
    cfg.v_reso = (float)in.vres;
    cfg.yawrate_reso = (float)in.wres;
    cfg.robot_radius = (float)in.radius;

    // The field is translated every tick so the robot sits at the origin,
    // which is the frame its own cost function assumes.  calc_to_goal_cost is
    // the angle between the goal and the trajectory endpoint measured *from
    // the world origin* -- acos(dot(goal, traj.back()) / (|goal| *
    // |traj.back()|)) -- so it is the angle between the direction of travel
    // and the direction to the goal only while the robot is at the origin,
    // which is where its own demo starts.
    //
    // Measured, on this field, three ways.  At the field's own coordinates,
    // (1, 1) to (10.6, 10.6): both lie on one ray from the origin, so that
    // cost is zero everywhere along the diagonal and it drifted 26.49 m to
    // finish 14.48 m away, outside the field.  Translated once, at the start:
    // the bearing is right at first and stales as it drives, and it was still
    // circling at (3.06, 2.71) after 90 s.  Translated every tick, below.
    //
    // This is the same per-tick transform the amslabtech trace makes for the
    // same reason -- that one's rollouts start from a zero state, so its
    // obstacles and goal have to be in the base frame -- and neither touches
    // the scoring.
    TraceState s{in.sx, in.sy, in.syaw, 0.0, 0.0};
    Obstacle ob;
    ob.resize(in.nob);
    Point goal{{0.0f, 0.0f}};
    std::vector<double> ms, rolls;
    out.xy = { s.x, s.y };
    for (int k = 0; k < in.max_steps; k++) {
        for (int i = 0; i < in.nob; i++)
            ob[i] = {{(float)(in.obx[i] - s.x), (float)(in.oby[i] - s.y)}};
        goal = {{(float)(in.gx - s.x), (float)(in.gy - s.y)}};
        State x{{0.0f, 0.0f, (float)s.yaw, (float)s.v, (float)s.w}};
        Window dw = calc_dynamic_window(x, cfg);
        rolls.push_back(std::floor((dw[1] - dw[0]) / cfg.v_reso)
                        * std::floor((dw[3] - dw[2]) / cfg.yawrate_reso));
        Control u{0, 0};
        auto t0 = std::chrono::steady_clock::now();
        calc_final_input(x, u, dw, cfg, goal, ob);
        ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count());
        s = trace_step(s, u[0], u[1], in);
        out.xy.push_back(s.x);
        out.xy.push_back(s.y);
        if (trace_arrived(s, in)) break;
    }
    out.ms = trace_median(ms);
    out.rolls = trace_median(rolls);
}

// How many trajectories its own loop evaluates for a given side, so the
// "same trajectory count" claim in bench/README.md is checkable rather than
// asserted: it walks `for (v = dw[0]; v <= dw[1]; v += v_reso)`, which is a
// different construction from this repo's lattice and need not agree on the
// count even given the same window and the same resolution.
int count_cpprobotics(int side) {
    Config cfg;
    cfg.dt = 0.1f; cfg.predict_time = 2.5f;
    cfg.v_reso = (cfg.max_speed - cfg.min_speed) / (side - 1);
    cfg.yawrate_reso = (2 * cfg.max_yawrate) / (side - 1);
    cfg.max_accel = 1e6f; cfg.max_dyawrate = 1e6f;
    State x{{1.0f, 1.0f, 0.0f, 0.0f, 0.0f}};
    Window dw = calc_dynamic_window(x, cfg);
    int n = 0;
    for (float v = dw[0]; v <= dw[1]; v += cfg.v_reso)
        for (float y = dw[2]; y <= dw[3]; y += cfg.yawrate_reso) n++;
    return n;
}
