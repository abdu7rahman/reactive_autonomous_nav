#!/usr/bin/env python3
"""Reproduce the site's cursor-chase geometry, and measure MPPI's reversal.

The handoff carried an open item: *MPPI reverses and stalls in the chase demo,
v = -0.09 m/s at start, stops 0.33 m out against a 0.15 m goal tolerance.* It
never reproduced on the controller suite, whose maps are 153 and 200 cells wide
and whose reference path is decimated. The chase is neither: an 86x44 plate at
0.05 m a cell, and the raw A* cell path handed straight to the controller, one
waypoint every five centimetres.

Rebuilding those conditions splits the item in two, and only one half is the
controller's:

  * The reversal is real, and it was the softmax. Before `fix(mppi): scale the
    softmax temperature to the units of the cost`, lambda was a bare 0.3
    against a cost spanning ~300, the effective sample size was 1.00 of 1000,
    and the command was one noise draw's first element. `--legacy` runs that
    version from git and the two print side by side.

  * The 0.33 m is the plate's own goal snap, not the controller. The chase
    moves a cursor inside the inflation band to the nearest cell at or below
    FREE_COST before A* sees it. With nav2's exponential falloff and a four
    cell radius, the first such cell is five cells from a wall, so a cursor in
    a corner moves the goal 0.35 m -- and the robot then arrives at the goal it
    was given while the readout measures to the cursor. `--snap` prints that
    table. Any controller shows it.

    python3 bench/chase_mppi.py            # both weightings, five goals
    python3 bench/chase_mppi.py --snap     # the goal-snap table
"""
import argparse
import contextlib
import importlib.util
import io
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rig


def _sig():
    """Author signature. stderr, tty-only, so redirected output stays clean."""
    import os, sys
    if os.environ.get("NO_BANNER") == "1" or not sys.stderr.isatty():
        return
    print("  " + "".join(chr(c - 7) for c in
          (104,105,107,124,115,39,121,104,111,116,104,117)), file=sys.stderr)

#: The plate, as demo.js sizes it on a desktop canvas: 1204x616 CSS pixels at
#: 14 px a cell, 0.05 m a cell, a one-cell lethal border and 4 cells of
#: inflation. 4.30 m by 2.20 m.
RES, CW, CH, INFL = 0.05, 86, 44, 4

#: The commit that scaled the temperature; its parent is the version measured
#: in the handoff.
FIX = "2bd01a1"

#: Where the chase parks the robot, and the goals swept across the plate.
START = (CW * RES * 0.15, CH * RES * 0.5, 0.0)
GOALS = ((2.50, 1.10), (1.50, 1.10), (1.20, 1.60), (3.40, 0.60), (1.00, 1.10))


def inflate(g, radius=INFL, res=RES, lethal=253):
    """demo.js `_inflate`: nav2_costmap_2d's exponential falloff.

    Not `maps.inflate`. That one decays linearly and puts a cell three from a
    wall under FREE_COST, which moves the goal snap below the threshold the
    whole second half of this measurement is about.
    """
    big = 1e9
    d = np.where(g >= lethal, 0.0, big)
    for _ in range(int(radius) + 2):
        p = np.pad(d, 1, constant_values=big)
        d = np.minimum.reduce([d,
            p[:-2, 1:-1] + 1.0,  p[2:, 1:-1] + 1.0,
            p[1:-1, :-2] + 1.0,  p[1:-1, 2:] + 1.0,
            p[:-2, :-2] + 1.414, p[:-2, 2:] + 1.414,
            p[2:, :-2] + 1.414,  p[2:, 2:] + 1.414])
    band = (g < lethal) & (d <= radius)
    scaled = np.minimum(252.0, 252.0 * np.exp(-6.0 * np.minimum(d, 1e3) * res))
    out = g.copy()
    out[band] = np.maximum(out[band], scaled.astype(np.int16)[band])
    return out


def plate():
    g = np.zeros((CH, CW), dtype=np.int16)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 254
    return rig.Grid(inflate(g), resolution=RES)


def planner(grid):
    node = object.__new__(rig.node_class(rig.load("astar_planner")))
    rig.apply_defaults(node, "astar_planner")
    rig.wire_global(node, grid)
    node.global_data = grid.data
    return node


def nearest_free(node, grid, r, c):
    """demo.js `_nearest_free`: out of the inflation band, then off lethal."""
    m = rig.load("astar_planner")

    def ok(rr, cc, clear):
        if not (0 <= rr < grid.h and 0 <= cc < grid.w):
            return False
        v = node._merged_cell_cost(rr, cc)
        return v <= m.FREE_COST if clear else v < m.LETHAL_COST

    for clear in (True, False):
        if ok(r, c, clear):
            return (r, c)
        for rad in range(1, 14):
            best, bd = None, 1e9
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if max(abs(dr), abs(dc)) != rad or not ok(r + dr, c + dc, clear):
                        continue
                    if dr * dr + dc * dc < bd:
                        best, bd = (r + dr, c + dc), dr * dr + dc * dc
            if best:
                return best
    return None


def chase_path(node, grid, start, goal):
    """A* between snapped cells, one world point per cell -- no decimation.

    The controller suite decimates by two before handing the path over. The
    chase does not, and a path sampled every five centimetres is what any
    critic that reads `path_yaw` at the nearest point sees as a staircase.
    """
    s = nearest_free(node, grid, *grid.w2g(start[0], start[1]))
    g = nearest_free(node, grid, *grid.w2g(goal[0], goal[1]))
    if s is None or g is None or s == g:
        return []
    cells, _ = node._astar(s, g)
    return [grid.g2w(r, c) for r, c in cells]


def _legacy_module(tmp):
    """`mppi_controller.py` as it was before the temperature fix."""
    src = subprocess.run(["git", "show", f"{FIX}^:reactive_autonomous_nav/mppi_controller.py"],
                         capture_output=True, text=True, check=True).stdout
    path = os.path.join(tmp, "mppi_controller.py")
    with open(path, "w") as fh:
        fh.write(src)
    # Loaded under the module's real name on purpose: `rig.prepare` finds the
    # publishers by reading the package source for `type(node).__module__`, and
    # a node whose module is called something else silently loses `marker_pub`.
    spec = importlib.util.spec_from_file_location("mppi_controller", path)
    mod = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(mod)
    return mod


def drive(module, grid, pts, start, steps=140, dt=0.1):
    node = object.__new__(rig.node_class(module))
    # The current file's defaults, on both versions. The fix changed only the
    # weighting, so sharing them is what isolates it.
    rig.apply_defaults(node, "mppi_controller")
    rig.prepare(node)
    node.costmap_data, node.costmap_info = grid.data, grid.info()
    node.costmap_origin = grid.origin
    cmd = rig.Sink()
    node.cmd_pub = cmd
    node._path_cb(rig.make_path(pts))
    node.goal_reached = False
    x, y, yaw = start
    gx, gy = pts[-1]
    vs = []
    for _ in range(steps):
        pose = (x, y, yaw)
        node._get_robot_pose = lambda p=pose: p
        node._get_tf = lambda t, s, p=pose: p if "base_link" in (t, s) else (0.0, 0.0, 0.0)
        before = len(cmd.msgs)
        node._control_loop()
        if node.goal_reached:
            break
        v = w = 0.0
        if len(cmd.msgs) > before:
            m = cmd.msgs[-1]
            v, w = float(m.linear.x), float(m.angular.z)
        if isinstance(getattr(node, "current_vel", None), dict):
            node.current_vel = {"v": v, "omega": w}
        vs.append(v)
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw = (yaw + w * dt + math.pi) % (2 * math.pi) - math.pi
    return dict(reached=bool(node.goal_reached), vs=vs,
                final=math.hypot(gx - x, gy - y))


def run_weightings():
    grid = plate()
    p = planner(grid)
    paths = [(g, chase_path(p, grid, START, g)) for g in GOALS]
    with tempfile.TemporaryDirectory() as tmp:
        versions = (("pre-" + FIX, _legacy_module(tmp)),
                    ("current", rig.load("mppi_controller")))
        for label, module in versions:
            print(f"\n  {label} weighting")
            print("    %-14s %4s  %-5s %8s  %8s  %s"
                  % ("goal", "wps", "ok", "final m", "min v", "first six v"))
            for goal, pts in paths:
                if not pts:
                    print("    %-14s no path" % (goal,))
                    continue
                np.random.seed(7)
                r = drive(module, grid, pts, START)
                print("    %-14s %4d  %-5s %8.3f  %8.3f  %s"
                      % (str(goal), len(pts), r["reached"], r["final"],
                         min(r["vs"]), [round(v, 3) for v in r["vs"][:6]]))


def run_snap():
    grid = plate()
    p = planner(grid)
    print("\n  cursor            snapped goal      moved")
    worst = 0.0
    for gx, gy in ((0.03, 0.03), (0.05, 0.05), (0.10, 0.10), (0.15, 0.15),
                   (0.05, 1.10), (0.20, 1.10), (4.25, 2.15), (2.15, 1.10)):
        cell = nearest_free(p, grid, *grid.w2g(gx, gy))
        if cell is None:
            print("    (%.2f, %.2f)      unreachable" % (gx, gy))
            continue
        wx, wy = grid.g2w(*cell)
        d = math.hypot(wx - gx, wy - gy)
        worst = max(worst, d)
        print("    (%.2f, %.2f)      (%.2f, %.2f)      %.3f m" % (gx, gy, wx, wy, d))
    print("\n  worst %.3f m, against the node's 0.15 m goal tolerance" % worst)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap", action="store_true", help="only the goal-snap table")
    args = ap.parse_args()
    _sig()
    if args.snap:
        run_snap()
    else:
        run_weightings()
        run_snap()
