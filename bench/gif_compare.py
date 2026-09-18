#!/usr/bin/env python3
"""Drive every DWA implementation across one field and draw what each one did.

    python3 bench/gif_compare.py            # -> bench/gif/dwa-python.gif
    python3 bench/gif_compare.py --cpp      # -> bench/gif/dwa-cpp.gif
    python3 bench/gif_compare.py --seeds    # the three-field table below
    python3 bench/gif_compare.py --frames   # keep the PNGs

Run it on python3.12: the rendering needs matplotlib, and the matplotlib in
dist-packages is built for 3.12 while `python3` here is 3.11.  The timing
tables next door are quoted on 3.11 with numpy 2.4 and the ms/tick here is
3.12 with numpy 1.26.4, so the two are not the same measurement and this one
is not a substitute for them -- see the note on what each measures below.

The tables say how long each implementation takes to score one window.  They
say nothing about where any of them goes, and a reader has no way to tell a
fast implementation from a fast one that drives badly.  This is the same
implementations, closed loop across one obstacle field, each one's own scoring
function choosing its own commands, integrated by one unicycle plant.

What each is given is what it is written for, and they are not the same thing:

  - this repo's controller tracks a path.  It gets one, from this repo's own
    A* across the same field, and picks its own lookahead along it -- eight
    waypoints, about 0.4 m at costmap resolution, which is its own default.
  - PythonRobotics and kmilo7204 are goal seekers.  Their cost is the bearing
    to a goal, their own demos drive at one, and they get the final goal.

That asymmetry is the comparison, not a flaw in it: half of this repo is a
global planner, and the line on the chart is a planner and a controller
together against a controller alone.  The distance column is where it shows.
Four other arrangements were measured and rejected, each for a reason worth
keeping:

  - Everyone on the goal, ours included: a 13 m goal with no path leaves this
    repo's lookahead nothing to walk along.
  - Everyone on a straight line to the goal: the field left that line clear by
    0.654 m against a 0.55 m inflated radius, so this repo drove it untouched
    while the references searched around obstacles, and reported 26.5 s
    against 33.5 s.  That number measured the field.
  - Everyone on a straight line with obstacles parked on it: this repo stalls,
    correctly.  It follows a path with a 0.4 m lookahead; routing around a
    blockage two metres ahead is the global planner's job.
  - Everyone on a serpentine A* route through three rows of shelving, with
    obstacles parked on it that the route was planned without: nobody
    finished.  A parked obstacle freezes a stale route's waypoint index -- the
    point inside it is never reached and never retired -- so all three orbited
    it at 0.5 m for the rest of the run.  Skipping the points in collision got
    this repo from 6.46 m short to 6.15 m short and no further.  A blocked
    route is what the replanner this repo is named for is for, and there is no
    replanner in this harness.

Each gets the obstacle representation it is written for: an explicit list to
the baselines, and the same obstacles rasterised into a costmap with a 0.30 m
inflation for this repo, which is what it reads.  That difference is the whole
structural argument in the timing tables and it would be dishonest to hide it
here -- so the field is drawn once, as circles, and the costmap it becomes is
drawn underneath.

Two of the four C and C++ implementations cross about 3 m of this field and
then crawl, and the reason is worth more than the outcome.  CppRobotics' cost
is to_goal_cost + speed_cost + ob_cost with ob_cost = 1/min_r and no gain on
it, against a speed cost of max_speed - v: instrumented at tick 300 it sits
1.14 m from an obstacle with ob_cost 0.876 against a speed cost of 0.480, so
the whole speed term is worth less than the clearance it would give up by
moving, and standing still is optimal.  goktug97's clearance term is the same
reciprocal.  That is the pathology this repo's own C++ controller was fixed
for -- its obstacle term was the raw margin to lethal, up to 253, added to a
heading term worth at most pi, and it maximised room instead of making
progress -- and the fix, a normalised penalty capped and subtracted, is in the
comment beside the scoring in cpp/src/dwa_controller.cpp.  PythonRobotics has
the same 1/min_r but ships to_goal_cost_gain 0.15 against speed_cost_gain 1.0,
and it arrives.

Held equal: the plant and its accelerations, the sampling resolutions, and the
clearance.  PythonRobotics defaults to a 1.0 m robot radius and kmilo7204 to a
1.0 m chassis radius, against the 0.22 m this repo inflates for, and on a
field with obstacles 1.5 m apart that difference is the whole result -- both
found every trajectory in collision and crawled at 0.02 m/s.  All three ask
for RADIUS + OB_R here, which is what their centre-to-centre test has to be
given to match a costmap that has already inflated the obstacle.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import dwa_compare, maps, rig                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'gif')

DT, HORIZON = 0.1, 25                  # 2.5 s, the tables' horizon
FIELD = 12.0                           # metres square
START = (1.0, 1.0, math.pi / 4)
GOAL = (10.6, 10.6)
GOAL_TOL = 0.4
MAX_STEPS = 900                        # 90 s; the route is 13.8 m at 0.50 m/s
N_OBSTACLES = 18
SPACING = 1.5                          # m between obstacle centres
OB_R = 0.25                            # what an obstacle is, to the baselines
RES = 0.05                             # costmap resolution for this repo
INFLATION = 0.30
RADIUS = 0.22                          # the clearance all of them are held to

# Seed 7 of the three that were measured, because all three implementations
# arrive on it and a field where one simply stops compares less than a field
# where all of them finish.  The other two are not hidden -- `--seeds` prints
# them, and they are the table in bench/README.md.  Picking a seed the
# references fail on would have flattered this repo by a factor of three.
SEED = 7

# One plant for all of them, and the window that goes with it.  These are the
# robot this repo actually drives -- irobot_create_control/config/control.yaml,
# the same four numbers sim/race_robot.py puts into Gazebo's DiffDrive and
# dwa_controller.py's own defaults.  A dynamic window is the velocities
# reachable in the next interval given the robot's accelerations (Fox, Burgard
# and Thrun), so leaving the accelerations out is not a wider window, it is a
# different method.  step_plant clamps to these too.
TOP_SPEED = 0.50                       # m/s
MAX_YAW = 2.0                          # rad/s
ACC_V = 0.9                            # m/s^2
ACC_W = 7.725                          # rad/s^2

# dwa_controller.py's own sampling resolutions, given to all three.  Deriving
# them from a fixed trajectory count instead -- 20 x 20 over the full range,
# to match the tables' middle row -- quantises the yaw axis at 0.21 rad/s, and
# since the window is acceleration-limited to 1.545 rad/s wide that is nine
# samples rather than thirty-nine: this repo stalled after 3.5 m on a route its
# own planner had just produced.  The tables hold the count equal because they
# are timing one window; this holds the resolution equal because it is driving.
VEL_RES = 0.02                         # m/s
YAW_RES = 0.04                         # rad/s

CARROT = 0.4                           # lookahead_wps=8 at costmap resolution
WP_TOL = 0.25                          # dwa_controller.py's own wp_tol

COLOUR = {'this repo': '#ff6a1f',
          'PythonRobotics': '#50e6a0',
          'kmilo7204': '#5aaaff',
          'CppRobotics': '#50e6a0',
          'goktug97': '#5aaaff',
          'amslabtech': '#e57ad0'}
GIVEN = {'this repo': 'route', 'CppRobotics': 'goal', 'goktug97': 'goal',
         'amslabtech': 'goal', 'PythonRobotics': 'goal', 'kmilo7204': 'goal'}


def obstacles(seed=SEED):
    """A scatter, clear of the start and the goal, with no trap in it.

    A scatter rather than rows of shelving, which was tried: a goal seeker has
    no global plan, so a wall is not an obstacle to it but a trap, and scoring
    two implementations that cannot see round a corner on a maze measures
    whether they have a planner rather than how they drive.  SPACING keeps
    every pair far enough apart that a gap between them is passable.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(60000):
        if len(out) == N_OBSTACLES:
            break
        p = rng.uniform(1.2, FIELD - 1.2, 2)
        if math.dist(p, START[:2]) < 1.5 or math.dist(p, GOAL) < 1.5:
            continue
        if any(math.dist(p, q) < SPACING for q in out):
            continue
        out.append(tuple(p))
    return np.array(out)


def costmap(ob, radius=OB_R):
    """The same obstacles, rasterised and inflated the way nav2 inflates."""
    n = int(round(FIELD / RES))
    g = np.zeros((n, n), dtype=np.int16)
    ys, xs = np.mgrid[0:n, 0:n]
    wx = (xs + 0.5) * RES
    wy = (ys + 0.5) * RES
    for cx, cy in ob:
        g[np.hypot(wx - cx, wy - cy) <= radius] = 254
    g = maps.inflate(g, radius_cells=int(round(INFLATION / RES)),
                     inscribed_cells=int(round(RADIUS / RES)))
    return rig.Grid(g, resolution=RES, origin=(0.0, 0.0))


def clearance(ob, grid):
    """The clearance this repo's costmap actually enforces, measured off it.

    Not OB_R + RADIUS, which is what it was asked for.  The obstacle is
    rasterised by testing cell centres against OB_R and the inscribed band is
    int(RADIUS / RES) = 4 cells rather than 4.4, so the nearest free cell to an
    obstacle centre sits at 0.425 m against the 0.470 m asked for -- and
    handing the baselines 0.470 for their own centre-to-centre test let this
    repo drive 45 mm nearer every obstacle than they were allowed to.  Small,
    and exactly the sort of quiet edge that makes a comparison worthless, so
    the number they are held to is this one.
    """
    n = grid.data.shape[0]
    ys, xs = np.mgrid[0:n, 0:n]
    d = np.min(np.hypot((xs + 0.5) * RES - ob[:, 0].reshape(-1, 1, 1),
                        (ys + 0.5) * RES - ob[:, 1].reshape(-1, 1, 1)), axis=0)
    return float(d[grid.data < 253].min())


def route(ob):
    """This repo's own A* across the field, which is what its controller
    tracks: a global planner that routes and a local controller that follows
    what it routed is the arrangement this half of the repo is written for."""
    grid = costmap(ob)
    node = object.__new__(rig.node_class(rig.load('astar_planner')))
    node.global_info = grid.info()
    node.global_origin = grid.origin
    node.global_data = grid.data
    node.local_data = None             # -> _local_cost_at_map returns -1
    node.odom_to_map = None
    cells, _explored = node._astar(grid.w2g(*START[:2]), grid.w2g(*GOAL))
    if not cells:
        raise RuntimeError('no route across the field')
    return np.array([grid.g2w(r, c) for r, c in cells])


class Carrot:
    """Where on the route this repo's controller is steering, out of
    dwa_controller.py's own control_loop rather than reinvented here.

      - the index only moves forward, retired when its successor is nearer,
        which makes progress monotonic in distance along the route rather than
        in how close the robot happened to pass.  Proximity alone strands a
        tracker behind a waypoint it flew past.
      - the target is then walked back to the furthest point whose straight
        segment from the robot is clear of lethal cells, which is what pure
        pursuit does and what stops the target sitting on the far side of an
        obstacle the route wraps.  Measured on the three fields: it fires on 25
        to 29 of about 277 ticks, so it is carrying real weight here.

    The distance is arc length, not a count of waypoints: A*'s own steps are
    0.05 m straight and 0.0707 m diagonal, so a fixed count reaches 41 percent
    further out on the diagonal stretches and the robot would speed up and slow
    down with the shape of the route rather than with its own scoring.
    """

    def __init__(self, pts, grid, d=CARROT):
        self.pts, self.grid, self.d = pts, grid, d
        self.i = 0
        self.run = np.concatenate([[0.0], np.cumsum(
            np.hypot(*np.diff(pts, axis=0).T))])
        self.walked = 0

    def clear(self, x0, y0, x1, y1):
        """Lethal-free straight segment, sampled at half a cell as
        _segment_clear does -- Bresenham skips cells the segment crosses."""
        step = self.grid.resolution * 0.5
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step) + 1)
        t = np.linspace(0.0, 1.0, n + 1)
        gx = ((x0 + (x1 - x0) * t - self.grid.origin[0])
              / self.grid.resolution).astype(int)
        gy = ((y0 + (y1 - y0) * t - self.grid.origin[1])
              / self.grid.resolution).astype(int)
        ok = (gx >= 0) & (gy >= 0) & (gx < self.grid.w) & (gy < self.grid.h)
        return not bool(np.any(self.grid.data[gy[ok], gx[ok]] >= 253))

    def next(self, x):
        pts, n = self.pts, len(self.pts)
        while self.i < n - 1:
            d_cur = math.dist(pts[self.i], x[:2])
            if d_cur < WP_TOL or math.dist(pts[self.i + 1], x[:2]) < d_cur:
                self.i += 1
            else:
                break
        j = min(int(np.searchsorted(self.run, self.run[self.i] + self.d)), n - 1)
        j0 = j
        while j > self.i and not self.clear(x[0], x[1], pts[j][0], pts[j][1]):
            j -= 1
        self.walked += j != j0
        return pts[j]


def step_plant(x, v, w):
    """One unicycle, clamped to the plant, for every implementation.

    The clamp is not cosmetic.  PythonRobotics reuses max_delta_yaw_rate as the
    magnitude of its own stuck recovery -- best_u[1] = -max_delta_yaw_rate when
    the best sample is v=0 -- so setting that to 1e6 to stop the acceleration
    limit narrowing the window, which is what the first version of this file
    did, had it commanding 1e6 rad/s the moment it stalled, and none of the
    three moved more than 2.3 m.  With the window acceleration-limited and the
    plant clamped, the same recovery is a spin in place at MAX_YAW, which is
    what it is written to be.
    """
    v = min(max(v, -TOP_SPEED), TOP_SPEED)
    w = min(max(w, -MAX_YAW), MAX_YAW)
    return np.array([x[0] + v * math.cos(x[2]) * DT,
                     x[1] + v * math.sin(x[2]) * DT,
                     x[2] + w * DT, v, w])


def drive_mine(ob, pts, d=CARROT, _radius=None):
    grid = costmap(ob)
    node = object.__new__(rig.node_class(rig.load('dwa_controller')))
    rig.apply_defaults(node, 'dwa_controller')
    rig.prepare(node)
    node.costmap_data, node.costmap_info = grid.data, grid.info()
    node.costmap_origin = grid.origin
    node.predict_time = HORIZON * DT
    node.dt = DT
    node.max_vel = TOP_SPEED
    node.max_yawrate = MAX_YAW
    node.max_accel = ACC_V
    node.max_dyawrate = ACC_W
    node.vel_res, node.yawrate_res = VEL_RES, YAW_RES
    cart = Carrot(pts, grid, d)
    x = np.array([*START, 0.0, 0.0])
    trail, ts, rolls = [x[:2].copy()], [], []
    for _ in range(MAX_STEPS):
        g = cart.next(x)
        node.current_vel = {'v': float(x[3]), 'omega': float(x[4])}
        dw = node._dynamic_window()
        vs = node._samples(dw[0], dw[1], node.vel_res)
        ws = node._samples(dw[2], dw[3], node.yawrate_res)
        rolls.append(len(vs) * len(ws))
        t0 = time.perf_counter()
        r = node._score_trajectories(x[0], x[1], x[2], vs, ws, g[0], g[1])
        ts.append((time.perf_counter() - t0) * 1000.0)
        x = step_plant(x, float(r[0]), float(r[1]))
        trail.append(x[:2].copy())
        if math.dist(x[:2], GOAL) < GOAL_TOL:
            break
    return np.array(trail), ts, rolls


def drive_reference(ob, _pts=None, _d=None, radius=None):
    pr = dwa_compare.load_reference()
    cfg = pr.Config()
    cfg.dt = DT
    cfg.predict_time = HORIZON * DT
    cfg.max_speed = TOP_SPEED
    cfg.min_speed = 0.0                # this repo's min_vel; no reverse
    cfg.max_yaw_rate = MAX_YAW
    cfg.max_accel = ACC_V
    cfg.max_delta_yaw_rate = ACC_W
    cfg.v_resolution, cfg.yaw_rate_resolution = VEL_RES, YAW_RES
    # Their test is centre-to-centre against the obstacle list, so the radius
    # carries the obstacle too, or they may drive OB_R into every one of them
    # while this repo's costmap calls the same cell lethal.  The value is what
    # that costmap measures, not what it was asked for -- see clearance().
    cfg.robot_radius = radius if radius else RADIUS + OB_R
    goal = np.array(GOAL)
    x = np.array([*START, 0.0, 0.0])
    trail, ts, rolls = [x[:2].copy()], [], []
    for _ in range(MAX_STEPS):
        dw = pr.calc_dynamic_window(x, cfg)
        rolls.append(len(np.arange(dw[0], dw[1], cfg.v_resolution))
                     * len(np.arange(dw[2], dw[3], cfg.yaw_rate_resolution)))
        t0 = time.perf_counter()
        u, _traj = pr.calc_control_and_trajectory(x, dw, cfg, goal, ob)
        ts.append((time.perf_counter() - t0) * 1000.0)
        x = step_plant(x, float(u[0]), float(u[1]))
        trail.append(x[:2].copy())
        if math.dist(x[:2], GOAL) < GOAL_TOL:
            break
    return np.array(trail), ts, rolls


def drive_kmilo(ob, _pts=None, _d=None, radius=None):
    km = dwa_compare.load_kmilo()
    k = km.DWA()
    cfg = k.config_params
    cfg.dt = DT
    cfg.dw_time = HORIZON * DT
    cfg.max_v = TOP_SPEED
    cfg.min_v = 0.0
    cfg.max_w, cfg.min_w = MAX_YAW, -MAX_YAW
    cfg.max_a = ACC_V
    cfg.max_d_w = ACC_W
    cfg.v_res, cfg.w_res = VEL_RES, YAW_RES
    cfg.chassis_radius = radius if radius else RADIUS + OB_R   # as above
    cfg.obstacles = ob
    goal = np.array(GOAL)
    x = np.array([*START, 0.0, 0.0])
    trail, ts, rolls = [x[:2].copy()], [], []
    for _ in range(MAX_STEPS):
        dw = k.calculate_dw(x)
        rolls.append(len(np.arange(dw[0], dw[1], cfg.v_res))
                     * len(np.arange(dw[2], dw[3], cfg.w_res)))
        t0 = time.perf_counter()
        out = k.calculate_ctrl_traj(x, goal)
        ts.append((time.perf_counter() - t0) * 1000.0)
        u = out[0] if isinstance(out, (tuple, list)) else out
        x = step_plant(x, float(u[0]), float(u[1]))
        trail.append(x[:2].copy())
        if math.dist(x[:2], GOAL) < GOAL_TOL:
            break
    return np.array(trail), ts, rolls


DRIVERS = (('this repo', drive_mine),
           ('PythonRobotics', drive_reference),
           ('kmilo7204', drive_kmilo))


def check(ob, pts):
    """What the field is, and what makes it worth driving across.

    Asserted rather than described: a route no longer than the straight line
    means nothing had to be avoided, and that is exactly how the first version
    of this file came to report a win that was really its field.
    """
    straight = math.dist(START[:2], GOAL)
    run = float(np.hypot(*np.diff(pts, axis=0).T).sum())
    head = np.unwrap(np.arctan2(*np.diff(pts, axis=0).T[::-1]))
    turn = math.degrees(float(np.abs(np.diff(head)).sum()))
    d = np.min(np.hypot(pts[:, None, 0] - ob[None, :, 0],
                        pts[:, None, 1] - ob[None, :, 1]), axis=1)
    grid = costmap(ob)
    gxy = ((pts - np.array(grid.origin)) / RES).astype(int)
    lethal = int(np.sum(grid.data[gxy[:, 1], gxy[:, 0]] >= 253))
    print(f'  route {run:.2f} m across {len(ob)} obstacles, '
          f'{100 * (run / straight - 1):.1f} % longer than the straight line, '
          f'turning {turn:.0f} deg, closest approach {d.min():.3f} m')
    assert lethal == 0, f'{lethal} route points are in collision'
    assert run > straight, 'the route is the straight line: nothing to avoid'

    # How much the goal seekers have to get round, which is the same question
    # asked of their input: the obstacles on the straight line to the goal.
    t = np.linspace(0, 1, int(straight / RES))[:, None]
    line = np.array(START[:2]) * (1 - t) + np.array(GOAL) * t
    near = np.min(np.hypot(line[:, None, 0] - ob[None, :, 0],
                           line[:, None, 1] - ob[None, :, 1]), axis=0)
    blocking = int(np.sum(near <= OB_R + RADIUS))
    print(f'  {blocking} of them block the straight line to the goal '
          f'(nearest {near.min():.3f} m, lethal at {OB_R + RADIUS:.2f} m)')
    assert blocking > 0, "nothing is in the goal seekers' way either"

    sep = min(math.dist(p, q) for i, p in enumerate(ob) for q in ob[i + 1:])
    print(f'  clearance every implementation is held to '
          f'{clearance(ob, grid):.3f} m, measured off the costmap')
    print(f'  closest pair {sep:.3f} m apart, '
          f'{sep - 2 * (OB_R + RADIUS):.3f} m of free floor between them')
    assert sep > 2 * (OB_R + RADIUS), 'two of them seal the gap between'
    return True


def run_all(ob, pts):
    r = clearance(ob, costmap(ob))
    runs = {}
    for label, fn in DRIVERS:
        trail, ms, rolls = fn(ob, pts, CARROT, r)
        runs[label] = (trail, ms, rolls)
    return runs


def report(label, trail, ms, rolls):
    arrived = math.dist(trail[-1], GOAL) < GOAL_TOL
    drove = float(np.hypot(*np.diff(trail, axis=0).T).sum())
    return (f'  {label:<16}{GIVEN[label]:>6}  {len(trail) * DT:6.1f} s  '
            f'{drove:6.2f} m  {statistics.median(ms):7.2f} ms/tick  '
            f'{statistics.median(rolls):5.0f} rollouts  '
            f'{"arrived" if arrived else f"{math.dist(trail[-1], GOAL):.2f} m short"}')


def seeds():
    """Every field that was measured, not only the one that is drawn."""
    for s in (3, SEED, 11):
        ob = obstacles(s)
        pts = route(ob)
        print(f'seed {s}: route {np.hypot(*np.diff(pts, axis=0).T).sum():.2f} m'
              + ('   <- the one drawn' if s == SEED else ''))
        for label, (trail, ms, rolls) in run_all(ob, pts).items():
            print(report(label, trail, ms, rolls))


def density(cpp=False):
    """Arrivals against how much clutter there is.

    "did not arrive" on its own says nothing about why, and clutter is the
    first thing to rule out: same start, goal and plant, obstacle count swept.

    The fields are nested -- obstacles() takes the first n of one sequence --
    so this adds obstacles to a field rather than drawing a new one each time,
    which is the sweep worth having and also why the two that crawl report the
    same distance to the centimetre at every count: the obstacle that decides
    it is in all four fields.  It is not the clutter.
    """
    global N_OBSTACLES
    keep = N_OBSTACLES
    try:
        for n in (6, 10, 14, 18):
            N_OBSTACLES = n
            ob = obstacles()
            pts = route(ob)
            sep = min(math.dist(p, q) for i, p in enumerate(ob)
                      for q in ob[i + 1:])
            print(f'{n} obstacles, closest pair {sep:.2f} m, '
                  f'route {np.hypot(*np.diff(pts, axis=0).T).sum():.2f} m')
            runs = run_cpp(ob, pts, costmap(ob)) if cpp else run_all(ob, pts)
            for label, (trail, ms, rolls) in (runs or {}).items():
                print(report(label, trail, ms, rolls))
    finally:
        N_OBSTACLES = keep


def render(ob, pts, runs, grid, name, keep_frames=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    os.makedirs(OUT, exist_ok=True)
    frames = os.path.join(OUT, '.frames')
    os.makedirs(frames, exist_ok=True)
    for f in os.listdir(frames):
        os.remove(os.path.join(frames, f))

    # The legend gets a band above the field rather than a corner inside it:
    # its top line used to land across an obstacle, which is unreadable in the
    # one frame anybody looks at.
    band = 0.62 * len(runs) + 0.62
    longest = max(len(t) for t, _ms, _r in runs.values())
    step = max(1, longest // 90)            # about 90 frames whatever happens
    paths = []
    for k, n in enumerate(range(1, longest + step, step)):
        fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=88)
        ax.set_facecolor('#3e424a')
        fig.patch.set_facecolor('#3e424a')
        # Only the lethal cells, the rest transparent.  Drawing zeros too
        # washed the whole 12 m square a shade lighter than the margin around
        # it, so the field read as a panel inset in the frame rather than as
        # the floor the obstacles sit on.
        lethal = np.where(grid.data >= 253, 1.0, np.nan)
        ax.imshow(lethal, origin='lower', extent=(0, FIELD, 0, FIELD),
                  cmap='Greys', vmin=0, vmax=1.6, alpha=0.5,
                  interpolation='nearest')
        for cx, cy in ob:
            ax.add_patch(Circle((cx, cy), OB_R, color='#d98a3a', alpha=0.9))
        # Over the trails, not under them: this repo's own trail sits on the
        # route almost exactly -- 13.29 m driven against 13.83 m planned --
        # so underneath it the route is invisible in every frame after the
        # first, and the one line in the caption pointing at it reads as a
        # caption about nothing.
        ax.plot(pts[:, 0], pts[:, 1], lw=1.1, ls=(0, (5, 4)),
                color='#e8eaee', alpha=0.85, zorder=7)
        ax.plot(*GOAL, marker='*', ms=18, color='#ffd65c', zorder=5)
        ax.plot(START[0], START[1], marker='o', ms=7, color='#cfd3da', zorder=5)
        for label, (trail, ms, _r) in runs.items():
            m = min(n, len(trail))
            ax.plot(trail[:m, 0], trail[:m, 1], lw=2.6, color=COLOUR[label],
                    solid_capstyle='round', zorder=4)
            ax.plot(trail[m - 1, 0], trail[m - 1, 1], marker='o', ms=7,
                    color=COLOUR[label], zorder=6)
        ax.text(0.28, FIELD + band - 0.40, f'{len(ob)} obstacles, '
                f'{HORIZON}-step horizon, {TOP_SPEED} m/s, dashed is this '
                f'repo\'s A* route', color='#c8ccd4', fontsize=8.5,
                family='monospace', va='center')
        for i, (label, (trail, ms, _r)) in enumerate(runs.items()):
            reached = math.dist(trail[-1], GOAL) < GOAL_TOL
            drove = float(np.hypot(*np.diff(trail[:min(n, len(trail))],
                                            axis=0).T).sum())
            ax.text(0.28, FIELD + band - 1.02 - i * 0.62,
                    f'{label:<15}{GIVEN[label]:>5}  '
                    f'{statistics.median(ms):6.2f} ms/tick  '
                    f'{min(n, len(trail)) * DT:5.1f} s  {drove:5.2f} m'
                    f'{"" if reached or n < longest else "   did not arrive"}',
                    color=COLOUR[label], fontsize=9.5, family='monospace',
                    va='center')
        ax.set_xlim(0, FIELD)
        ax.set_ylim(0, FIELD + band)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.tight_layout(pad=0.2)
        p = os.path.join(frames, f'f{k:04d}.png')
        fig.savefig(p, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths.append(p)

    out = os.path.join(OUT, name)
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-framerate', '12',
                    '-i', os.path.join(frames, 'f%04d.png'),
                    '-vf', 'scale=560:-2:flags=lanczos,split[s0][s1];'
                           '[s0]palettegen=max_colors=64[p];'
                           '[s1][p]paletteuse=dither=none',
                    '-loop', '0', out], check=True)
    if not keep_frames:
        for p in paths:
            os.remove(p)
        os.rmdir(frames)
    return out, len(paths)


def field_file(ob, pts, grid, dest):
    """The field, the route, the costmap and every shared constant, for
    --trace.  Written rather than recomputed on the other side: two random
    number generators agreeing is a thing to verify, and one file both sides
    read is not.

    Plain keys and counted arrays rather than JSON, and the costmap as raw
    int8 beside it: the C++ side has no JSON parser and adding a dependency to
    a bench harness to carry twenty numbers is worse than the format.
    """
    binary = dest + '.map'
    grid.data.astype(np.int8).tofile(binary)
    with open(dest, 'w') as f:
        for k, v in (('dt', DT), ('horizon', HORIZON), ('res', RES),
                     ('radius', clearance(ob, grid)), ('top_speed', TOP_SPEED),
                     ('max_yaw', MAX_YAW), ('acc_v', ACC_V), ('acc_w', ACC_W),
                     ('vel_res', VEL_RES), ('yaw_res', YAW_RES),
                     ('carrot', CARROT), ('wp_tol', WP_TOL),
                     ('goal_tol', GOAL_TOL), ('max_steps', MAX_STEPS),
                     ('start_x', START[0]), ('start_y', START[1]),
                     ('start_yaw', START[2]), ('goal_x', GOAL[0]),
                     ('goal_y', GOAL[1])):
            f.write(f'{k} {v}\n')
        f.write(f'obstacles {len(ob)}\n')
        for a, b in ob:
            f.write(f'{a:.6f} {b:.6f}\n')
        f.write(f'route {len(pts)}\n')
        for a, b in pts:
            f.write(f'{a:.6f} {b:.6f}\n')
        f.write(f'map {grid.w} {grid.h} {binary}\n')
    return dest, binary


def run_cpp(ob, pts, grid):
    """Each C++ implementation's own closed-loop trace, from --trace."""
    exe = os.path.join(HERE, 'dwa_compare_cpp')
    if not os.path.exists(exe):
        print(f'  {exe} missing -- run bench/run.sh first', file=sys.stderr)
        return None
    os.makedirs(OUT, exist_ok=True)
    field, binary = field_file(ob, pts, grid, os.path.join(OUT, '.field'))
    raw = subprocess.run([exe, '--trace', field], check=True,
                         capture_output=True, text=True).stdout
    runs = {}
    for line in raw.splitlines():
        if not line.startswith('trace '):
            continue
        _, label, ms, rolls, poses = line.split(' ', 4)
        xy = np.array([float(v) for v in poses.split(',')]).reshape(-1, 2)
        runs[label.replace('_', ' ')] = (xy, [float(ms)], [float(rolls)])
    os.remove(field)
    os.remove(binary)
    if not runs:
        print('  no trace lines from ' + exe, file=sys.stderr)
        return None
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', action='store_true', help='keep the PNGs')
    ap.add_argument('--cpp', action='store_true', help='the C++ four')
    ap.add_argument('--seeds', action='store_true', help='every field measured')
    ap.add_argument('--density', action='store_true', help='arrivals vs clutter')
    ap.add_argument('--field', metavar='PATH', help='write the field and stop')
    args = ap.parse_args()

    if args.seeds:
        seeds()
        return 0
    if args.density:
        density(args.cpp)
        return 0

    ob = obstacles()
    pts = route(ob)
    print(f'seed {SEED}, {len(ob)} obstacles, {FIELD:.0f} m square, '
          f'{START[:2]} to {GOAL}, {HORIZON}-step horizon, '
          f'{VEL_RES}/{YAW_RES} resolution')
    check(ob, pts)

    grid = costmap(ob)
    if args.field:
        # Kept rather than deleted, which --cpp does with its own copy: it is
        # the input `dwa_compare_cpp --sweep-goktug` takes, and that sweep is
        # quoted in bench/README.md.  A documented command whose input only
        # exists inside another command's temporary file is not reproducible.
        f, b = field_file(ob, pts, grid, args.field)
        print(f'  {f}\n  {b}')
        return 0
    if args.cpp:
        runs = run_cpp(ob, pts, grid)
        if runs is None:
            return 1
        name = 'dwa-cpp.gif'
    else:
        runs = run_all(ob, pts)
        name = 'dwa-python.gif'
    for label, (trail, ms, rolls) in runs.items():
        print(report(label, trail, ms, rolls))

    out, n = render(ob, pts, runs, grid, name, args.frames)
    print(f'{out}  {n} frames  {os.path.getsize(out) / 1048576.0:.2f} MB')
    return 0


if __name__ == '__main__':
    sys.exit(main())
