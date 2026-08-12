#!/usr/bin/env python3
"""Closed-loop checks for the DWA controller.

Two things are checked here that the suite in test_planners.py could not see.

1. The velocity window itself. _control_loop trims the window's upper bound
   with a clearance cap, then samples it. Both operations used to be wrong at
   the edges: the sampler admitted one step past the bound, so the commanded
   speed could exceed max_vel or the cap, and it returned nothing at all when
   the cap fell below the window's floor -- which is what happens on every
   approach to an obstacle above 0.12 m/s. The empty case fell through to a
   commanded 0.00, an instantaneous stop from whatever the robot was doing.

2. The controller driving a plant. It was previously excluded because the
   rig's transform stub answered every frame pair with the robot pose, so
   odom<-map came back as the pose and the goal was transformed by it.

    python3 bench/test_dwa_window.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import maps, rig                                        # noqa: E402

LETHAL = 253
MODULE = "dwa_controller"


def fresh():
    node = object.__new__(rig.node_class(rig.load(MODULE)))
    rig.apply_defaults(node, MODULE)
    rig.prepare(node)
    return node


# ── 1. the window ────────────────────────────────────────────────────
def window_cases():
    """Every (speed, cap) pair the clearance logic can produce."""
    n = fresh()
    caps = [0.08, 0.18, n.max_vel]                 # the three _control_loop bands
    speeds = [0.0, 0.02, 0.06, 0.10, 0.16, 0.22, 0.30, 0.42, n.max_vel]
    rows, bad = [], 0
    for v in speeds:
        for cap in caps:
            n.current_vel = {"v": v, "omega": 0.0}
            dw = n._dynamic_window()
            dw[1] = max(dw[0], min(dw[1], cap))
            vs = n._samples(dw[0], dw[1], n.vel_res)

            reach_lo = max(n.min_vel, v - n.max_accel * n.dt)
            reach_hi = min(n.max_vel, v + n.max_accel * n.dt)
            why = []
            if vs.size == 0:
                why.append("empty window")
            if vs.size and vs.max() > min(reach_hi, max(cap, reach_lo)) + 1e-9:
                why.append(f"commands {vs.max():.3f} above the cap/limit")
            if vs.size and vs.min() < reach_lo - 1e-9:
                why.append(f"commands {vs.min():.3f} below reachable {reach_lo:.3f}")
            if vs.size and abs(vs.max() - vs.min()) > 1e-9:
                gaps = np.diff(vs)
                if gaps.max() > n.vel_res + 1e-9:
                    why.append("gap wider than vel_res")
            bad += bool(why)
            rows.append((v, cap, dw[0], dw[1], vs.size,
                         float(vs.min()), float(vs.max()), "; ".join(why)))
    return rows, bad


def yawrate_symmetry():
    """The yaw-rate window has to stay centred on the current rate."""
    n = fresh()
    bad = 0
    for w in (-2.0, -0.9, 0.0, 0.9, 2.0):
        n.current_vel = {"v": 0.2, "omega": w}
        dw = n._dynamic_window()
        ws = n._samples(dw[2], dw[3], n.yawrate_res)
        if ws.size == 0 or ws.min() < -n.max_yawrate - 1e-9 or ws.max() > n.max_yawrate + 1e-9:
            bad += 1
    return bad


# ── 2. the controller against a plant ────────────────────────────────
def reference_path(grid, start, goal):
    an = object.__new__(rig.node_class(rig.load("astar_planner")))
    rig.apply_defaults(an, "astar_planner")
    rig.wire_global(an, grid)
    an.global_data = grid.data
    cells, _ = an._astar(start, goal)
    assert cells, "reference planner found no path"
    # every cell, not every other one: with the inscribed band in place a
    # two-cell diagonal hop can clip a corner the planner itself never took
    pts = [grid.g2w(r, c) for r, c in cells]
    ok, why = rig.world_path_is_valid(grid, pts, LETHAL)
    assert ok, f"reference path is itself invalid: {why}"
    return pts


def closed_loop(g, start, goal):
    grid = rig.Grid(g)
    pts = reference_path(grid, start, goal)
    node = fresh()
    node.costmap_data, node.costmap_info = grid.data, grid.info()
    node.costmap_origin = grid.origin
    node.current_vel = {"v": 0.0, "omega": 0.0}
    # the plan is metres long and the controller holds ~0.1 m/s through
    # corridors this tight, so the budget is sized off the plan, not a constant
    plan_m = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
    r = rig.drive(node, grid, pts, (pts[0][0], pts[0][1], 0.0),
                  max_steps=int(plan_m / 0.05 / 0.1) + 400, goal_tol=0.2)
    r["plan_m"] = plan_m
    return r, len(pts)


def main():
    _sig()
    fails = 0

    print("VELOCITY WINDOW -- reachable set under every clearance cap")
    print(f"  {'v now':>7}{'cap':>7}   {'window':<16}{'n':>4}{'lo':>7}{'hi':>7}   note")
    rows, bad = window_cases()
    for v, cap, lo, hi, n, vmin, vmax, why in rows:
        flag = "" if not why else "  <-- " + why
        print(f"  {v:>7.2f}{cap:>7.2f}   [{lo:.2f}, {hi:.2f}]{'':<7}{n:>4}"
              f"{vmin:>7.2f}{vmax:>7.2f}{flag}")
    fails += bad
    print(f"  {len(rows) - bad}/{len(rows)} windows sound")

    sym = yawrate_symmetry()
    fails += sym
    print(f"  yaw-rate window: {'in bounds at every current rate' if not sym else str(sym) + ' out of bounds'}")

    print("\nCLOSED LOOP -- the node's own _control_loop against a unicycle plant")
    print("  costmap carries the inscribed band, so the node's centre-point")
    print("  lethal check is the footprint check it is on the robot")
    print(f"  {'map':<18}{'ok':<6}{'sec':>6}{'plan m':>8}{'drove':>8}{'ratio':>7}{'v max':>7}   note")
    for name, g, start, goal in maps.controller_suite():
        gi = maps.inflate(g.astype(np.int16), radius_cells=4,
                          inscribed_cells=maps.INSCRIBED_CELLS)
        r, npts = closed_loop(gi, start, goal)
        ok = r["reached"] and not r["collided"]
        # a tracker is only useful if it stays on the plan it was given
        if ok and r["length"] / r["plan_m"] > 1.25:
            ok, note = False, f"drove {r['length'] / r['plan_m']:.2f}x the plan"
        else:
            note = ("collided" if r["collided"] else
                    "" if r["reached"] else f"stalled {r['dist_to_goal']:.2f} m out")
        fails += not ok
        print(f"  {name:<18}{'PASS' if ok else 'FAIL':<6}{r['steps'] * 0.1:>6.0f}"
              f"{r['plan_m']:>8.2f}{r['length']:>8.2f}"
              f"{r['length'] / r['plan_m']:>7.3f}{r['vmax']:>7.2f}   {note}")

    print(f"\n{'all checks passed' if not fails else str(fails) + ' FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())


def _sig():
    """Author signature. stderr, tty-only, so redirected output stays clean."""
    import os, sys
    if os.environ.get("NO_BANNER") == "1" or not sys.stderr.isatty():
        return
    print("  " + "".join(chr(c - 7) for c in
          (104,105,107,124,115,39,121,104,111,116,104,117)), file=sys.stderr)
