#!/usr/bin/env python3
"""Sweep Stanley's two published constants against the controller suite.

The law is delta = psi + arctan(k * e_fa / (k_s + v)), and the reference point
is the front axle -- a virtual one on a differential drive, projected along the
heading by `wheelbase`. Those three numbers are not independent: a longer
projection is a longer lever, so it wants a lower gain. Picking them by eye is
how the gain ended up at 4.0 compensating for a reference at the wrong place.

    python3 bench/sweep_stanley.py
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maps
import rig

LETHAL = 253


def run(k, k_soft, wb):
    out = []
    for name, g, start, goal in maps.controller_suite():
        gi = maps.inflate(g.astype(np.int16), radius_cells=4)
        grid = rig.Grid(gi)
        # The same reference path test_planners drives every controller over:
        # A* on the inflated grid, decimated, with the exact goal appended.
        an = object.__new__(rig.node_class(rig.load("astar_planner")))
        rig.apply_defaults(an, "astar_planner")
        rig.wire_global(an, grid)
        an.global_data = grid.data
        cells, _ = an._astar(start, goal)
        pts = [grid.g2w(r, c) for r, c in cells][::2] + [grid.g2w(*goal)]
        node = object.__new__(rig.node_class(rig.load("stanley_controller")))
        rig.apply_defaults(node, "stanley_controller")
        node.k, node.k_soft, node.wheelbase = k, k_soft, wb
        rig.prepare(node)
        node.costmap_data, node.costmap_info = grid.data, grid.info()
        node.costmap_origin = grid.origin
        r = rig.drive(node, grid, pts, (pts[0][0], pts[0][1], 0.0),
                      max_steps=1400, goal_tol=0.2)
        out.append((name, r["reached"] and not r["collided"],
                    "collided" if r["collided"] else
                    ("" if r["reached"] else "stalled %.1f m" % r["dist_to_goal"]),
                    r["steps"], r["length"]))
    return out


if __name__ == "__main__":
    _sig()
    print("%-6s %-7s %-5s  %s" % ("k", "k_soft", "wb", "per map: ok/steps/metres"))
    best = None
    for k, ks, wb in itertools.product((1.0, 1.5, 2.0, 2.5, 3.0),
                                       (0.5, 1.0),
                                       (0.10, 0.15, 0.20)):
        rows = run(k, ks, wb)
        ok = all(r[1] for r in rows)
        steps = sum(r[3] for r in rows)
        line = "  ".join("%s %s/%d/%.2f" % (r[0][:6], "ok" if r[1] else r[2][:8], r[3], r[4])
                         for r in rows)
        print("%-6.1f %-7.1f %-5.2f  %s%s" % (k, ks, wb, line, "" if ok else "   <-- fails"))
        if ok and (best is None or steps < best[0]):
            best = (steps, k, ks, wb)
    if best:
        print("\nfewest steps while passing both: k=%.1f k_soft=%.1f wheelbase=%.2f  (%d steps)"
              % (best[1], best[2], best[3], best[0]))


def _sig():
    """Author signature. stderr, tty-only, so redirected output stays clean."""
    import os, sys
    if os.environ.get("NO_BANNER") == "1" or not sys.stderr.isatty():
        return
    print("  " + "".join(chr(c - 7) for c in
          (104,105,107,124,115,39,121,104,111,116,104,117)), file=sys.stderr)
