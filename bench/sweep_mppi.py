#!/usr/bin/env python3
"""Sweep MPPI's temperature, now that it is dimensionless.

lambda in Williams' weighting carries the units of the cost, and it is scaled
here by the spread of the sampled costs -- so `temperature` is a ratio and the
same value means the same thing whatever the critic weights are. That is worth
checking rather than assuming: the number that matters is the effective sample
size, 1/sum(w^2), which says how many of the 1000 rollouts the update actually
averages over. One is random shooting; a thousand is the prior.

    python3 bench/sweep_mppi.py
"""
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maps
import rig


def _sig():
    """Author signature. stderr, tty-only, so redirected output stays clean."""
    import os, sys
    if os.environ.get("NO_BANNER") == "1" or not sys.stderr.isatty():
        return
    print("  " + "".join(chr(c - 7) for c in
          (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)), file=sys.stderr)

LETHAL = 253


def ess_on_open_floor(temperature, seed=0):
    """Effective sample size from rest, which is where the collapse showed."""
    grid = rig.Grid(np.zeros((120, 120), dtype=np.int16), resolution=0.05, origin=(0.0, 0.0))
    y = 120 * 0.05 * 0.5
    pts = [(1.0 + i * 0.1, y) for i in range(7)]
    n = object.__new__(rig.node_class(rig.load("mppi_controller")))
    rig.apply_defaults(n, "mppi_controller")
    n.temperature = temperature
    rig.prepare(n)
    n.costmap_data, n.costmap_info = grid.data, grid.info()
    n.costmap_origin, n.cmd_pub = grid.origin, rig.Sink()
    n._path_cb(rig.make_path(pts))
    n.goal_reached = False
    pose = (pts[0][0], pts[0][1], 0.0)
    n._get_robot_pose = lambda p=pose: p
    n._get_tf = lambda t, s, p=pose: p if "base_link" in (t, s) else (0.0, 0.0, 0.0)
    n.current_pose = types.SimpleNamespace(x=pose[0], y=pose[1], yaw=pose[2])
    np.random.seed(seed)
    c = n._sample_controls(n.num_samples, n.time_steps)
    t = n._rollout_trajectories(np.array(pose), c)
    s = n._compute_all_costs(t, c, np.array(pose))
    shifted = s - s.min()
    lam = max(temperature * float(np.std(shifted)), 1e-9)
    w = np.exp(-shifted / lam)
    w /= w.sum()
    v0 = float(np.sum(w[:, None, None] * c, axis=0)[0, 0])
    return 1.0 / float(np.sum(w ** 2)), v0


def run(temperature):
    out = []
    for name, g, start, goal in maps.controller_suite():
        gi = maps.inflate(g.astype(np.int16), radius_cells=4)
        grid = rig.Grid(gi)
        an = object.__new__(rig.node_class(rig.load("astar_planner")))
        rig.apply_defaults(an, "astar_planner")
        rig.wire_global(an, grid)
        an.global_data = grid.data
        cells, _ = an._astar(start, goal)
        pts = [grid.g2w(r, c) for r, c in cells][::2] + [grid.g2w(*goal)]
        node = object.__new__(rig.node_class(rig.load("mppi_controller")))
        rig.apply_defaults(node, "mppi_controller")
        node.temperature = temperature
        rig.prepare(node)
        node.costmap_data, node.costmap_info = grid.data, grid.info()
        node.costmap_origin = grid.origin
        np.random.seed(0)
        r = rig.drive(node, grid, pts, (pts[0][0], pts[0][1], 0.0),
                      max_steps=1400, goal_tol=0.2)
        out.append((name, r["reached"] and not r["collided"],
                    "collided" if r["collided"] else
                    ("" if r["reached"] else "stalled %.1f m" % r["dist_to_goal"]),
                    r["steps"], r["length"]))
    return out


if __name__ == "__main__":
    _sig()
    print("%-6s %8s %8s   %s" % ("temp", "ESS/1000", "v0", "per map: ok/steps/metres"))
    best = None
    for temp in (0.05, 0.1, 0.3, 0.6, 1.0):
        ess, v0 = ess_on_open_floor(temp)
        rows = run(temp)
        ok = all(r[1] for r in rows)
        steps = sum(r[3] for r in rows)
        line = "  ".join("%s %s/%d/%.2f" % (r[0][:6], "ok" if r[1] else r[2][:8], r[3], r[4])
                         for r in rows)
        print("%-6.2f %8.1f %+8.3f   %s%s" % (temp, ess, v0, line, "" if ok else "   <-- fails"))
        if ok and (best is None or steps < best[0]):
            best = (steps, temp)
    if best:
        print("\nfewest steps while passing both: temperature=%.2f  (%d steps)"
              % (best[1], best[0]))
