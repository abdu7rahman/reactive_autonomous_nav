// CppRobotics DWA, isolated in its own translation unit because its Point and
// Config types collide with the other baseline's.
#include ".third_party/cpprobotics_core.h"
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>
#include "trace.h"
#include <limits>

double bench_cpprobotics(int side, int reps, const BenchField& f, BenchWork* w) {
    Config cfg;
    cfg.dt = (float)f.dt; cfg.predict_time = (float)f.predict_time;
    cfg.max_speed = (float)f.max_speed; cfg.min_speed = 0.0f;
    cfg.max_yawrate = (float)f.max_yaw;
    cfg.v_reso = (cfg.max_speed - cfg.min_speed) / (side - 1);
    cfg.yawrate_reso = (2 * cfg.max_yawrate) / (side - 1);
    cfg.max_accel = 1e6f; cfg.max_dyawrate = 1e6f;
    // The clearance was its own 1.0 m default, which on this field puts an
    // obstacle inside the robot at every start pose: see BenchField.
    cfg.robot_radius = (float)f.radius;
    State x{{(float)f.sx, (float)f.sy, (float)f.syaw, 0.0f, 0.0f}};
    Point goal{{(float)f.gx, (float)f.gy}};
    Obstacle ob;
    for (int i = 0; i < f.nob; i++)
        ob.push_back({{(float)f.obx[i], (float)f.oby[i]}});
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
    if (w) {
        long long pts = 0, chk = 0, bail = 0, traj = 0;
        for (float v = dw[0]; v <= dw[1]; v += cfg.v_reso)
            for (float y = dw[2]; y <= dw[3]; y += cfg.yawrate_reso) {
                Traj tr = calc_trajectory(x, v, y, cfg);
                traj++;
                bool hit = false;
                for (unsigned ii = 0; ii < tr.size() && !hit; ii += 2) {
                    pts++;
                    for (unsigned i = 0; i < ob.size(); i++) {
                        chk++;
                        const float dx = tr[ii][0] - ob[i][0];
                        const float dy = tr[ii][1] - ob[i][1];
                        if (std::sqrt(dx * dx + dy * dy) <= cfg.robot_radius) {
                            hit = true; bail++; break;
                        }
                    }
                }
                (void)0;
            }
        w->points = (double)pts / traj;
        w->checks = (double)chk / traj;
        w->bailed = (double)bail / traj;
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// Its own calc_dynamic_window and calc_final_input, in closed loop on the
// goal.  Its cost is the bearing to that goal, its own demo drives at one, and
// a target inside its rollout reach inverts the bearing -- so it gets the
// goal rather than a carrot on this repo's route.
void step_cpprobotics(const StepIn& in, StepOut& out) {
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
    // Measured, on the figure's field, three ways.  At the field's own
    // coordinates, (1, 1) to (10.6, 10.6): both lie on one ray from the
    // origin, so that cost is zero everywhere along the diagonal and it
    // drifted 26.49 m to finish 14.48 m away, outside the field.  Translated
    // once, at the start: the bearing is right at first and stales as it
    // drives, and it was still circling at (3.06, 2.71) after 90 s.
    // Translated every tick, here.
    //
    // This is the same per-tick transform the amslabtech step makes for the
    // same reason -- that one's rollouts start from a zero state, so its
    // obstacles and goal have to be in the base frame -- and neither touches
    // the scoring.
    Obstacle ob;
    ob.resize(in.nob);
    for (int i = 0; i < in.nob; i++)
        ob[i] = {{(float)(in.obx[i] - in.x), (float)(in.oby[i] - in.y)}};
    Point goal{{(float)(in.gx - in.x), (float)(in.gy - in.y)}};

    State x{{0.0f, 0.0f, (float)in.yaw, (float)in.v, (float)in.w}};
    Window dw = calc_dynamic_window(x, cfg);
    out.rolls = std::floor((dw[1] - dw[0]) / cfg.v_reso)
                * std::floor((dw[3] - dw[2]) / cfg.yawrate_reso);
    Control u{0, 0};
    auto t0 = std::chrono::steady_clock::now();
    calc_final_input(x, u, dw, cfg, goal, ob);
    out.ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    out.v = u[0];
    out.w = u[1];
}

void trace_cpprobotics(const TraceIn& in, TraceOut& out) {
    TraceState s{in.sx, in.sy, in.syaw, 0.0, 0.0};
    std::vector<double> ms, rolls;
    out.xy = { s.x, s.y };
    for (int k = 0; k < in.max_steps; k++) {
        StepOut u;
        step_cpprobotics(trace_to_step(in, s, in.gx, in.gy), u);
        ms.push_back(u.ms);
        rolls.push_back(u.rolls);
        s = trace_step(s, u.v, u.w, in);
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
