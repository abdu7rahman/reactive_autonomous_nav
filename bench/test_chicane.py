#!/usr/bin/env python3
"""Drive all five controllers down the chicane reference path, off-simulator.

    python3 bench/test_chicane.py

sim/race_path.py builds one reference path and hands the same shape to all five
robots, so the only difference between them is how each controller tracks a
curve.  That only works if the reference is followable: a path a tracker cuts
by more than the corridor's spare width is a path that puts a robot in a wall,
and finding that out costs twenty minutes of simulator plus a ruined capture.

So the same path is driven here first, against the same unicycle plant the rest
of the bench uses and a costmap built from the course's own wall geometry, and
the two numbers that decide the course are measured rather than guessed: how
far each controller strays from the reference, and how close it gets to a wall.

The map is one lane, in lane-relative coordinates, with the neighbouring lanes'
walls included -- at a gate the far side of the corridor *is* the neighbour's
wall, and a map with only this lane's wall in it would score a robot as clear
while it drove through the robot next to it.  All five lanes are identical
translations (sim/race_path.py --check asserts that), so one lane is the whole
course.
"""
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'sim'))
from bench import maps, rig                                         # noqa: E402

os.environ.setdefault('RACE_COURSE', 'chicane')
import race_path                                                    # noqa: E402
from reactive_autonomous_nav.race_course import (                    # noqa: E402
    COSTMAP_RES, GATE_T, GATE_W, GATES, INFLATION, RACE_LENGTH, ROBOT_RADIUS,
    SPACING)


def _sig():
    """Author signature. stderr, tty-only, so redirected output stays clean."""
    import os, sys
    if os.environ.get("NO_BANNER") == "1" or not sys.stderr.isatty():
        return
    print("  " + "".join(chr(c - 7) for c in
          (104,105,107,124,115,39,121,104,111,116,104,117)), file=sys.stderr)


LETHAL = 253
CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      'config', 'race_costmap_params.yaml')
MARGIN_X = 1.8      # metres of map either side of the lane centre: the
                    # neighbour's wall at a gate sits 1.25 m out and is 0.30 m
                    # wide, so 1.8 holds it with a cell to spare
MARGIN_Y = 0.3

# nav2 measures both of these from the obstacle surface, and so does
# maps.inflate: inside ROBOT_RADIUS of a wall the robot's centre is in a cell
# the costmap calls lethal, which is what makes rig.drive's centre-point check
# the footprint check it is on the robot.  0.22 m is 4.4 cells and the band is
# whole cells, so it is 4 -- 0.20 m, two centimetres optimistic, and the
# clearances below are reported against the real 0.22 as well.
INSCRIBED_CELLS = int(ROBOT_RADIUS / COSTMAP_RES)       # 4
INFLATION_CELLS = int(round(INFLATION / COSTMAP_RES))   # 6


def config_agrees():
    """race_course's costmap constants against the costmaps' own config.

    ROBOT_RADIUS, INFLATION and COSTMAP_RES are declared in race_course.py
    with a comment pointing at config/race_costmap_params.yaml, and every
    clearance number in this file and in sim/race_path.py is measured against
    them.  Change the yaml and nothing would notice: the reference path would
    be checked against an inflation the costmaps no longer use, and the check
    would keep passing.  So it is checked.
    """
    text = open(CONFIG).read()
    want = {'robot_radius': ROBOT_RADIUS, 'inflation_radius': INFLATION,
            'resolution': COSTMAP_RES}
    bad = []
    for key, mine in want.items():
        found = {float(v) for v in re.findall(rf'^\s*{key}:\s*([0-9.]+)\s*$',
                                              text, re.M)}
        if not found:
            bad.append(f'{key}: not in the config at all')
        elif found != {mine}:
            bad.append(f'{key}: race_course says {mine}, the config says '
                       f'{sorted(found)}')
    return bad


def relative_walls():
    """Every wall within the map, as (x, y) offsets from the lane centre."""
    out = []
    for gy, off in GATES:
        for k in (-1, 0, 1):
            wx = off + k * SPACING
            if abs(wx) <= MARGIN_X + GATE_W:
                out.append((wx, gy))
    return out


def wall_gap(x, y):
    """Distance from (x, y) to the nearest wall surface, lane-relative."""
    best = float('inf')
    for wx, wy in relative_walls():
        best = min(best, math.hypot(max(0.0, abs(x - wx) - GATE_W / 2.0),
                                    max(0.0, abs(y - wy) - GATE_T / 2.0)))
    return best


def chicane_grid():
    """The lane as a costmap: walls lethal, inflated the way nav2 inflates."""
    origin = (-MARGIN_X, -MARGIN_Y)
    w = int(round(2 * MARGIN_X / COSTMAP_RES))
    h = int(round((RACE_LENGTH + 2 * MARGIN_Y) / COSTMAP_RES))
    g = np.zeros((h, w), dtype=np.int16)
    for r in range(h):
        for c in range(w):
            x = (c + 0.5) * COSTMAP_RES + origin[0]
            y = (r + 0.5) * COSTMAP_RES + origin[1]
            for wx, wy in relative_walls():
                if abs(x - wx) <= GATE_W / 2 and abs(y - wy) <= GATE_T / 2:
                    g[r, c] = 254
                    break
    g = maps.inflate(g, radius_cells=INFLATION_CELLS, lethal=LETHAL,
                     inscribed_cells=INSCRIBED_CELLS)
    return rig.Grid(g, resolution=COSTMAP_RES, origin=origin)


def cross_track(pts, x, y):
    """Distance from (x, y) to the reference polyline."""
    best = float('inf')
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
        best = min(best, math.hypot(x - (ax + t * dx), y - (ay + t * dy)))
    return best


def run(module, grid, pts):
    node = object.__new__(rig.node_class(rig.load(module)))
    rig.apply_defaults(node, module)
    rig.prepare(node)
    node.costmap_data, node.costmap_info = grid.data, grid.info()
    node.costmap_origin = grid.origin
    node.current_vel = {'v': 0.0, 'omega': 0.0}
    plan_m = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
    r = rig.drive(node, grid, pts, (pts[0][0], pts[0][1], math.pi / 2),
                  max_steps=int(plan_m / COSTMAP_RES / 0.1) + 400, goal_tol=0.2)
    dev = max(cross_track(pts, x, y) for x, y, _yaw in r['trace'])
    gap = min(wall_gap(x, y) for x, y, _yaw in r['trace'])
    r.update(dev=dev, gap=gap, plan_m=plan_m)
    return r


def main():
    _sig()
    if not GATES:
        print('RACE_COURSE has no gates -- nothing to drive')
        return 0

    fails = 0
    bad = config_agrees()
    for line in bad:
        print(f'  FAIL config: {line}')
    fails += len(bad)
    if not bad:
        print(f'costmap config agrees: robot radius {ROBOT_RADIUS} m, '
              f'inflation {INFLATION} m, {COSTMAP_RES} m cells')

    grid = chicane_grid()
    pts = race_path.centreline()
    ref_gap = min(wall_gap(x, y) for x, y in pts)
    print(f'chicane: {len(GATES)} gates, {len(relative_walls())} walls in view, '
          f'reference swing {max(abs(x) for x, _ in pts):.3f} m')
    print(f'  the reference itself clears the walls by {ref_gap:.3f} m, so a '
          f'tracker has {ref_gap - INFLATION:.3f} m before it is in the '
          f'inflated band and {ref_gap - ROBOT_RADIUS:.3f} m before its '
          f'footprint is in a wall')
    if ref_gap <= ROBOT_RADIUS:
        print('  FAIL: the reference path is itself inside a wall')
        return 1

    print(f"\n  {'controller':<14}{'ok':<6}{'sec':>6}{'drove':>8}{'ratio':>7}"
          f"{'max dev':>9}{'min gap':>9}   note")
    for module in ('pure_pursuit_controller', 'stanley_controller',
                   'dwa_controller', 'teb_controller', 'mppi_controller'):
        try:
            r = run(module, grid, pts)
        except Exception as e:                                       # noqa: BLE001
            print(f"  {module.replace('_controller', ''):<14}{'FAIL':<6}"
                  f"{'':>39}   {type(e).__name__}: {e}")
            fails += 1
            continue
        ok = r['reached'] and not r['collided']
        note = ('footprint in a wall' if r['collided'] else
                '' if r['reached'] else f"stalled {r['dist_to_goal']:.2f} m out")
        # A tracker that clears the walls by driving somewhere else is not
        # tracking. The corridor's spare width is what the reference leaves
        # over -- stray further than that and the run only survived because
        # the wall happened to be on the other side.
        if ok and r['dev'] > ref_gap - ROBOT_RADIUS:
            ok, note = False, f"strayed {r['dev']:.2f} m, past the corridor's spare"
        fails += not ok
        print(f"  {module.replace('_controller', ''):<14}{'PASS' if ok else 'FAIL':<6}"
              f"{r['steps'] * 0.1:>6.0f}{r['length']:>8.2f}"
              f"{r['length'] / r['plan_m']:>7.3f}{r['dev']:>9.3f}{r['gap']:>9.3f}"
              f"   {note}")

    print(f"\n{'all checks passed' if not fails else str(fails) + ' FAILURES'}")
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
