#!/usr/bin/env python3
"""Sweep the TEB band's three load-bearing constants against the suite.

The band is timed now, so the numbers that decide how fast it goes are the
acceleration edges and the vertex spacing -- not a tracking gain. Those are not
independent: a coarser band is a longer interval, which is a lower speed for the
same limits, and a wider spacing also means the heading of each interval is
averaged over more ground.

`desired_sep` in particular stopped being cosmetic when the command started
being read off the first interval. It used to only feed a weak separation force.

    python3 bench/sweep_teb.py
"""
import itertools
import os
import sys

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


def run(accel, yaw_accel, sep):
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
        node = object.__new__(rig.node_class(rig.load("teb_controller")))
        rig.apply_defaults(node, "teb_controller")
        node.max_accel, node.max_yaw_accel, node.desired_sep = accel, yaw_accel, sep
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
    print("%-7s %-9s %-5s  %s" % ("accel", "yaw_accel", "sep", "per map: ok/steps/metres"))
    best = None
    for a, ya, sep in itertools.product((1.5, 3.0, 6.0), (1.6, 3.2), (0.10, 0.15, 0.25)):
        rows = run(a, ya, sep)
        ok = all(r[1] for r in rows)
        steps = sum(r[3] for r in rows)
        line = "  ".join("%s %s/%d/%.2f" % (r[0][:6], "ok" if r[1] else r[2][:8], r[3], r[4])
                         for r in rows)
        print("%-7.1f %-9.1f %-5.2f  %s%s" % (a, ya, sep, line, "" if ok else "   <-- fails"))
        if ok and (best is None or steps < best[0]):
            best = (steps, a, ya, sep)
    if best:
        print("\nfewest steps while passing both: max_accel=%.1f max_yaw_accel=%.1f "
              "desired_sep=%.2f  (%d steps)" % (best[1], best[2], best[3], best[0]))
