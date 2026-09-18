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
#include "trace.h"
#include <cfloat>

double bench_goktug(int side, int reps, const BenchField& f, BenchWork* w) {
    Config gc;
    gc.maxSpeed = (float)f.max_speed; gc.minSpeed = 0.0f;
    gc.maxYawrate = (float)f.max_yaw;
    gc.maxAccel = 1e6f; gc.maxdYawrate = 1e6f;
    gc.velocityResolution = (float)f.max_speed / (side - 1);
    gc.yawrateResolution  = (float)(2 * f.max_yaw) / (side - 1);
    gc.dt = (float)f.dt; gc.predictTime = (float)f.predict_time;
    gc.heading = 5.0f; gc.clearance = 5.0f; gc.velocity = 0.5f;
    // Its footprint is a rectangle: the square with the same inradius as the
    // disc the others are given, as trace_goktug's own sweep also uses.
    gc.base.xmin = -(float)f.radius; gc.base.ymin = -(float)f.radius;
    gc.base.xmax =  (float)f.radius; gc.base.ymax =  (float)f.radius;
    Pose gp = {{(float)f.sx, (float)f.sy}, (float)f.syaw};
    Velocity gv = {0.0f, 0.0f};
    Point ggoal = {(float)f.gx, (float)f.gy};
    PointCloud* pc = createPointCloud(f.nob);
    for (int i = 0; i < f.nob; i++) {
        pc->points[i].x = (float)f.obx[i]; pc->points[i].y = (float)f.oby[i];
    }
    planning(gp, gv, ggoal, pc, gc);
    std::vector<double> t;
    for (int r = 0; r < reps; r++) {
        auto t0 = std::chrono::steady_clock::now();
        planning(gp, gv, ggoal, pc, gc);
        t.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count());
    }
    if (w) {
        DynamicWindow* dw = NULL;
        createDynamicWindow(gv, gc, &dw);
        long long pts = 0, chk = 0, bail = 0, traj = 0;
        const int nsteps = (int)(gc.predictTime / gc.dt);
        for (int i = 0; i < dw->nPossibleV; i++)
            for (int j = 0; j < dw->nPossibleW; j++) {
                Velocity pv = {dw->possibleV[i], dw->possibleW[j]};
                Pose p = gp;
                traj++;
                bool hit = false;
                for (int k = 0; k < nsteps && !hit; k++) {
                    p = motion(p, pv, gc.dt);
                    pts++;
                    for (int o = 0; o < pc->size; o++) {
                        chk++;
                        const float dx = p.point.x - pc->points[o].x;
                        const float dy = p.point.y - pc->points[o].y;
                        const float lx = -dx * cosf(p.yaw) + -dy * sinf(p.yaw);
                        const float ly =  dx * sinf(p.yaw) + -dy * cosf(p.yaw);
                        if (lx <= gc.base.xmax && lx >= gc.base.xmin &&
                            ly <= gc.base.ymax && ly >= gc.base.ymin) {
                            hit = true; bail++; break;
                        }
                    }
                }
            }
        freeDynamicWindow(dw);
        w->points = (double)pts / traj;
        w->checks = (double)chk / traj;
        w->bailed = (double)bail / traj;
    }
    freePointCloud(pc);
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// Its clearance gain is a harness choice, because it ships none: dwa.h's
// Config is a plain C struct with no initialisers and its README documents the
// field without a value.  Taking this repo's obstacle_gain of 5.0 for it, which
// is what the timing harness beside this does, stops it moving at all -- its
// clearance term is 1/minr, unnormalised, against a velocity term worth
// velocity_gain * (max_speed - v), so at 5.0 the cheapest thing it can do is
// stand still and keep its distance.  That is the same pathology this repo's
// own controller was fixed for, and the gain that was chosen against a
// normalised penalty does not carry over to a raw reciprocal.  Swept on this
// field, metres driven and whether it arrived:
//
//   5.0     0.00 m   never moves
//   1.0     0.00 m   never moves
//   0.5     0.00 m   never moves
//   0.2    13.93 m   arrived
//   0.1    13.79 m   arrived
//   0.05   13.71 m   arrived
//
// 0.2 is the largest that drives, so it is the one that keeps the most of the
// clearance behaviour the gain is there for.
static float TRACE_CLEARANCE = 0.2f;
// Its footprint is a rectangle, not a radius, so "the same clearance as the
// others" is a choice too: a square of half-width r has the disc of radius r
// inscribed in it and reaches r*sqrt(2) at the corners, and one of half-width
// r/sqrt(2) is inscribed in the disc instead.  Swept with the gain.
static float TRACE_HALFWIDTH = -1.0f;   // < 0 means in.radius

void trace_goktug_gain(float g) { TRACE_CLEARANCE = g; }
void trace_goktug_footprint(float h) { TRACE_HALFWIDTH = h; }

void step_goktug(const StepIn& in, StepOut& out) {
    Config gc;
    gc.maxSpeed = (float)in.top_speed;
    gc.minSpeed = 0.0f;
    gc.maxYawrate = (float)in.max_yaw;
    gc.maxAccel = (float)in.acc_v;
    gc.maxdYawrate = (float)in.acc_w;
    gc.velocityResolution = (float)in.vres;
    gc.yawrateResolution = (float)in.wres;
    gc.dt = (float)in.dt;
    gc.predictTime = (float)(in.steps * in.dt);
    gc.heading = 5.0f; gc.clearance = TRACE_CLEARANCE; gc.velocity = 0.5f;
    const float half = TRACE_HALFWIDTH > 0 ? TRACE_HALFWIDTH : (float)in.radius;
    gc.base.xmin = -half; gc.base.ymin = -half;
    gc.base.xmax =  half; gc.base.ymax =  half;

    PointCloud* pc = createPointCloud(in.nob);
    for (int i = 0; i < in.nob; i++) {
        pc->points[i].x = (float)in.obx[i];
        pc->points[i].y = (float)in.oby[i];
    }
    Point goal = {(float)in.gx, (float)in.gy};
    Pose p = {{(float)in.x, (float)in.y}, (float)in.yaw};
    Velocity vel = {(float)in.v, (float)in.w};

    DynamicWindow* dw = NULL;
    createDynamicWindow(vel, gc, &dw);
    out.rolls = (double)dw->nPossibleV * (double)dw->nPossibleW;
    freeDynamicWindow(dw);

    auto t0 = std::chrono::steady_clock::now();
    Velocity u = planning(p, vel, goal, pc, gc);
    out.ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    // planning() initialises total_cost to FLT_MAX and keeps a candidate only
    // on `cost < total_cost`, so when every candidate collides -- every
    // clearance cost is FLT_MAX -- bestVelocity is returned uninitialised.
    // Reading it is undefined; in practice it came back zero and the trail
    // stopped dead at (3.27, 3.26) with an obstacle 1.04 m away and no
    // movement for the remaining 86 s.  Turning in place instead is the
    // recovery the amslabtech step gives its own node for the same case, so
    // both get it rather than one.
    out.v = u.linearVelocity;
    out.w = u.angularVelocity;
    bool any = false;
    DynamicWindow* adm = NULL;
    createDynamicWindow(vel, gc, &adm);
    for (int i = 0; i < adm->nPossibleV && !any; i++)
        for (int j = 0; j < adm->nPossibleW && !any; j++) {
            Velocity pv = {adm->possibleV[i], adm->possibleW[j]};
            if (calculateClearanceCost(p, pv, pc, gc) < FLT_MAX) any = true;
        }
    freeDynamicWindow(adm);
    if (!any) { out.v = 0.0; out.w = in.max_yaw; }
    freePointCloud(pc);
}

void trace_goktug(const TraceIn& in, TraceOut& out) {
    TraceState s{in.sx, in.sy, in.syaw, 0.0, 0.0};
    std::vector<double> ms, rolls;
    out.xy = { s.x, s.y };
    for (int k = 0; k < in.max_steps; k++) {
        StepOut u;
        step_goktug(trace_to_step(in, s, in.gx, in.gy), u);
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

// As above: its createDynamicWindow computes nPossibleV as an int division of
// the window by the resolution, which truncates and never includes the upper
// bound.
int count_goktug(int side) {
    Config gc;
    gc.maxSpeed = 0.5f; gc.minSpeed = 0.0f; gc.maxYawrate = 2.0f;
    gc.maxAccel = 1e6f; gc.maxdYawrate = 1e6f;
    gc.velocityResolution = 0.5f / (side - 1);
    gc.yawrateResolution  = 4.0f / (side - 1);
    gc.dt = 0.1f; gc.predictTime = 2.5f;
    Velocity gv = {0.25f, 0.0f};
    DynamicWindow* dw = NULL;
    createDynamicWindow(gv, gc, &dw);
    const int n = dw->nPossibleV * dw->nPossibleW;
    freeDynamicWindow(dw);
    return n;
}
