"""One reference path through the chicane, handed to all five controllers.

    race_path.py --check     measure it against the walls; the gate
    race_path.py --draw      draw the lane, the bands and the path, in text

On the straight each robot runs its own A* and the race is a nav stack against
a nav stack.  On the chicane that is the wrong experiment: five A* runs on five
rolling costmaps do not produce the same path twice, because each robot sees a
different slice of the walls through its own lidar and replans on its own
schedule, so a controller that arrived second may simply have been given two
extra metres to drive.  The user asked for identical paths, and identical is
what this is -- one centreline, translated to each lane, published once to all
five at the same instant by sim/race_timer.py.  What differs between the five
is then only the tracker.

Shape.  The path is a slalom with a straight through each gate, not a smooth
weave.  The corridor at a gate *is* straight -- 1.10 m between one wall's edge
and the next lane's wall edge -- so holding the centre of it for DWELL metres
either side of the wall lets the path reach the middle of the corridor instead
of only passing through it on a curve, and moves whatever a tracker cuts onto
the approach, where there is 1.7 m of clear lane.

Where the swing comes from.  Not chosen: race_course.gate_corridor() computes
the free span at a gate from the wall geometry, and the path takes the middle
of it.  A hand-typed 0.40 would be a number nobody could re-derive after the
wall width changed.

All of the constants below were picked by driving the five controllers down
the result -- bench/test_chicane.py, off-simulator, same plant as the rest of
the bench -- and reading the clearance off the rollout.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import math
import signal
import sys

from reactive_autonomous_nav.race_course import (
    CONTROLLER, COSTMAP_RES, COURSE, GATES, GATE_T, GATE_W, INFLATION, LANES,
    RACE_LENGTH, ROBOT_RADIUS, START_Y, gate_corridor, wall_distance,
    wall_poses)

# Metres of straight either side of a gate's wall, before the path turns for
# the next one.  Measured at 0.00, 0.15, 0.35 and 0.60 by driving all five
# controllers down each: the dwell buys swing, and swing is clearance.  With no
# straight at all the smoothed path never reaches the corridor centre -- it
# peaks at 0.375 m of a possible 0.400 -- and clears the walls by 0.465 m,
# where DWA's rollout came within 0.361 m of one.  At 0.35 the path reaches
# the full 0.400, clears by 0.549, and DWA's worst approach is 0.420.  Past
# that it is free: 0.60 measured 0.550 m of reference clearance, 1 mm more,
# while DWA's deviation from the path grew from 0.171 m to 0.233 m and both
# pure_pursuit and MPPI took two seconds longer, because the straights from
# consecutive gates leave only 0.5 m of the 1.7 m run to make an 0.8 m
# lateral step in.  0.35 also covers the wall's own 0.15 m thickness four
# times over, so the path is straight across the whole of it.
DWELL = 0.35

# Smoothing.  The same data-plus-difference relaxation nav2's simple smoother
# runs, on the lateral coordinate only: the path is a function of distance up
# the course, so smoothing x(y) cannot fold it back on itself the way smoothing
# (x, y) jointly can.  The knot polyline's corners are what it is for --
# unsmoothed, the yaw rate commanded at a corner is a step, and a step is
# something all five of these controllers handle differently for reasons that
# have nothing to do with tracking a curve.
#
# nav2's own weights, kept.  The relaxation settles where the data and
# difference terms balance, which is a screened Poisson problem with length
# scale sqrt(W_SMOOTH/W_DATA) * 0.05 m: 0.076 m here, a cell and a half, and
# the reason a straight dwell is needed rather than more smoothing.  The
# reasonable-looking alternative was to soften the corner by dropping W_DATA,
# and it measured worse on both counts at once.  W_DATA 0.008 rounds the
# corner to a 0.92 m radius, but the rounding eats the swing -- 0.400 m to
# 0.344 -- so the reference clears the walls by 0.457 m instead of 0.549, and
# the spread between the best and worst tracker narrows from 0.045..0.171 m to
# 0.039..0.128, which is less of the difference the clip exists to show.  A
# softer curve is an easier curve; the point here was a hard one that is still
# safe.
W_DATA, W_SMOOTH, ITERS = 0.15, 0.35, 400


def knots():
    """The path's corners, lane-relative, as (dx, y) with y from the start."""
    out = [(0.0, 0.0)]
    for (gy, _off), swing in zip(GATES, _gate_offsets()):
        out.append((swing, gy - DWELL))
        out.append((swing, gy + DWELL))
    out.append((0.0, RACE_LENGTH))
    return out


def _gate_offsets():
    """The path's lateral offset at each gate: the middle of its corridor."""
    return [sum(gate_corridor(gy, off)) / 2.0 for gy, off in GATES]


def centreline():
    """The reference, lane-relative, resampled at the costmap resolution.

    One sample per costmap cell along the course.  Denser buys nothing -- the
    costmap cannot tell two samples in one cell apart -- and sparser is what
    made the first A* reference paths look like they cut corners they had not:
    the controllers steer at a waypoint, so the waypoint spacing is the
    resolution of the thing being compared.
    """
    k = knots()
    n = int(round(RACE_LENGTH / COSTMAP_RES)) + 1
    ys = [i * COSTMAP_RES for i in range(n)]

    # linear interpolation through the knots, then relax
    xs, j = [], 0
    for y in ys:
        while j + 2 < len(k) and k[j + 1][1] < y:
            j += 1
        (x0, y0), (x1, y1) = k[j], k[j + 1]
        t = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
        xs.append(x0 + t * (x1 - x0))

    raw = list(xs)
    for _ in range(ITERS):
        for i in range(1, len(xs) - 1):
            xs[i] += (W_DATA * (raw[i] - xs[i])
                      + W_SMOOTH * (xs[i - 1] + xs[i + 1] - 2.0 * xs[i]))
    return list(zip(xs, ys))


def path_for(ns):
    """The reference in `map`, for one lane."""
    lx = LANES[ns]
    return [(lx + dx, START_Y + y) for dx, y in centreline()]


def free_span(x0, y):
    """How far the path can be pushed each way at (x0, y) and stay free.

    Free means at least INFLATION from a wall surface -- the boundary of the
    band a costmap charges nothing for.  Returned as (left, right) in metres,
    infinite where there is no wall on that side.

    This is the measurement the edge walls in race_course.wall_poses() exist
    for, and the one the distance to the *nearest* wall cannot make: a missing
    wall 1.7 m away on the far side of a corridor is never the nearest thing
    to anything, so the nearest-wall profile of an open gate and a closed one
    are identical.  What differs is how far you can go before you are charged,
    which is what this returns.

    Solved rather than stepped.  A wall forbids the x interval within INFLATION
    of its box, which at a height y is its own half-width plus
    sqrt(INFLATION^2 - dy^2), so the span is exact arithmetic.  Walking
    outwards in 0.01 m steps was the first spelling and it reported 0.25 m for
    three lanes and 0.26 for two, off nothing but the float error in lane
    centres 1.7 m apart -- a gate failing on its own discretisation.
    """
    lo, hi = -math.inf, math.inf
    for wx, wy in wall_poses():
        dy = max(0.0, abs(y - wy) - GATE_T / 2.0)
        if dy >= INFLATION:
            continue
        e = GATE_W / 2.0 + math.sqrt(INFLATION ** 2 - dy ** 2)
        if wx + e <= x0:
            lo = max(lo, wx + e)
        elif wx - e >= x0:
            hi = min(hi, wx - e)
        else:
            return 0.0, 0.0          # x0 is itself inside the charged band
    return x0 - lo, hi - x0


def curvature(pts):
    """Max |curvature| along a polyline, 1/m, by circumscribed circle."""
    worst = 0.0
    for (ax, ay), (bx, by), (cx, cy) in zip(pts, pts[1:], pts[2:]):
        a = math.hypot(bx - ax, by - ay)
        b = math.hypot(cx - bx, cy - by)
        c = math.hypot(cx - ax, cy - ay)
        area2 = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
        if a * b * c > 1e-9:
            worst = max(worst, 2.0 * area2 / (a * b * c))
    return worst


def check():
    """Measure the path against the walls.  Non-zero exit if it is not clear.

    Three things have to hold, and the run is worthless if any of them does
    not.  The path must stay out of the inflated band, or the controllers are
    being handed a reference their own costmaps score as unsafe and will refuse
    to follow -- which reads in a clip as five robots stopping for no visible
    reason.  It must be the same shape in every lane, or the lane is a
    variable.  And its deviation has to be big enough to see: the clip is
    560 px across a 6.8 m grid, so a metre is 80 px and a swing of 0.40 m is
    32 px of lateral travel each way.
    """
    if not GATES:
        print(f'course {COURSE} has no gates; the reference path is the straight '
              f'line each robot\'s own planner already finds')
        return 0

    pts = centreline()
    swing = max(abs(dx) for dx, _y in pts)
    knot_swing = max(abs(dx) for dx, _y in knots())
    print(f'course {COURSE}: {len(pts)} samples at {COSTMAP_RES:.2f} m, '
          f'{len(GATES)} gates, dwell {DWELL:.2f} m')
    print(f'  swing {swing:.3f} m off the lane centre '
          f'({knot_swing:.3f} m at the knots, so smoothing pulled in '
          f'{knot_swing - swing:.3f} m)')
    print(f'  max curvature {curvature(pts):.2f} 1/m '
          f'(radius {1.0 / max(curvature(pts), 1e-9):.2f} m)')

    # The band a costmap charges nothing for starts INFLATION from a wall
    # surface; inside ROBOT_RADIUS of one the robot's centre is in a lethal
    # cell.  Both are measured from the surface, which is why wall_distance is
    # point-to-box.
    print(f'  {"lane":<5}{"controller":<14}{"wall gap":>10}{"to lethal":>11}'
          f'{"free either side":>18}')
    bad, seen = 0, {}
    for ns in LANES:
        p = path_for(ns)
        gap = min(wall_distance(x, y) for x, y in p)
        # measured at the gates, where the corridor is; between them the lane
        # is 1.7 m of open floor and the span is the search cap
        spans = [free_span(LANES[ns] + dx, START_Y + gy)
                 for (gy, _off), dx in zip(GATES, _gate_offsets())]
        # every side of every gate, not the tightest of them: the tightest is
        # the robot's own wall and is there whatever the far side is doing, so
        # a min over the six hides exactly the open gate this is looking for
        flat = [d for pair in spans for d in pair]
        seen[ns] = tuple(round(d, 3) for d in flat)
        flag = ''
        if gap < INFLATION:
            flag, bad = '  <-- inside the inflated band', bad + 1
        if not all(math.isfinite(d) for d in flat):
            flag, bad = '  <-- a gate with nothing on the far side', bad + 1
        span = (f'{min(flat):.2f}' if not math.isfinite(max(flat))
                else f'{min(flat):.2f}..{max(flat):.2f}')
        print(f'  {ns:<5}{CONTROLLER[ns]:<14}{gap:>10.3f}'
              f'{gap - ROBOT_RADIUS:>11.3f}{span:>18}{flag}')
    # the majority profile, not the first lane's: if lane one is the broken
    # one, comparing against it names the other four as the anomaly
    ref = max(set(seen.values()), key=list(seen.values()).count)
    odd = [ns for ns, v in seen.items() if v != ref]
    if odd:
        print(f'  FAIL: {", ".join(odd)} do not race between the same walls '
              f'as the rest: {seen[odd[0]]} against {ref}')
        bad += 1
    else:
        print(f'  all {len(LANES)} lanes are the same path between the same '
              f'walls, translated')

    px = 560.0 / (max(LANES.values()) - min(LANES.values()) + 2 * 0.7)
    print(f'  lateral travel in the clip: {2 * swing:.2f} m '
          f'= {2 * swing * px:.0f} px at 560 px wide')
    if 2 * swing * px < 20:
        print('  FAIL: under 20 px of weave -- the clip would read as a straight line')
        bad += 1

    print('' if bad else '  path is clear' )
    return 1 if bad else 0


def draw(ns='r3', step=2, half=24):
    """The lane as text: walls, the band they charge for, and the path.

    Reading a table of 121 offsets does not tell you whether a slalom weaves;
    looking at it does.  This is how the shape was checked before any of it
    reached the simulator, and it is what showed the path threading the middle
    of each corridor with the inflated band clear on both sides.
    """
    if not GATES:
        print(f'course {COURSE} is a straight line')
        return 0
    at = {round(y, 2): dx for dx, y in centreline()}
    lx = LANES[ns]
    n = int(round(RACE_LENGTH / COSTMAP_RES))
    print(f'  {ns} ({CONTROLLER[ns]}) lane: x offset across, course up the page')
    print(f'  #  wall     :  inflated band     o  the reference path')
    for i in range(n, -1, -step):
        y = i * COSTMAP_RES
        row = ''
        for c in range(-half, half + 1):
            dx = c * COSTMAP_RES
            d = wall_distance(lx + dx, START_Y + y)
            ch = '#' if d == 0.0 else (':' if d < INFLATION else ' ')
            if abs(at[round(y, 2)] - dx) < COSTMAP_RES / 2:
                ch = 'o' if ch == ' ' else 'X'
            row += ch
        mark = f'  y={y:4.1f}' if i % 20 == 0 else ''
        print(f'  |{row}|{mark}')
    return 0


def main():
    # `race_path.py --draw | head` is how the drawing gets read, and python's
    # default SIGPIPE handling turns that into a traceback on the last line.
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    if '--check' in sys.argv[1:]:
        return check()
    if '--draw' in sys.argv[1:]:
        return draw()
    print(__doc__.strip().splitlines()[0])
    print('  --check | --draw')
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
