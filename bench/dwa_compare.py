#!/usr/bin/env python3
"""Times this repo's DWA against the reference Python implementation.

Baseline is PythonRobotics' dynamic_window_approach.py (AtsushiSakai, MIT),
the most widely referenced Python DWA. It is fetched at run time rather than
vendored.

Both are given the same dynamic window, the same number of sampled
trajectories and the same horizon, so the comparison is per-trajectory cost
rather than per-configuration. The interesting difference is structural:
this repo scores every rollout in one numpy batch and looks obstacles up in an
O(1) costmap, while the reference loops in Python over (v, w) and measures
distance to every obstacle point, which is O(n_obstacles) per sample.

    python3 bench/dwa_compare.py
"""
import os, statistics, sys, time, types, urllib.request
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import rig, maps                                        # noqa: E402

PR_URL = ("https://raw.githubusercontent.com/AtsushiSakai/PythonRobotics/"
          "master/PathPlanning/DynamicWindowApproach/dynamic_window_approach.py")
PR_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pr_dwa.py")


def load_reference():
    if not os.path.exists(PR_CACHE):
        src = urllib.request.urlopen(PR_URL, timeout=30).read().decode()
        src = src.replace("show_animation = True", "show_animation = False")
        # the reference imports matplotlib at module scope purely for its demo;
        # the planner functions themselves do not need it
        src = src.replace("import matplotlib.pyplot as plt",
                          "plt = None  # plotting stripped: only the planner is timed")
        open(PR_CACHE, "w").write(src)
    import importlib.util
    spec = importlib.util.spec_from_file_location("pr_dwa", PR_CACHE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.show_animation = False
    return m


def mine(n_traj, horizon, grid):
    node = object.__new__(rig.node_class(rig.load("dwa_controller")))
    rig.apply_defaults(node, "dwa_controller")
    rig.prepare(node)
    node.costmap_data, node.costmap_info = grid.data, grid.info()
    node.costmap_origin = grid.origin
    node.predict_time = horizon * node.dt
    side = int(np.sqrt(n_traj))
    vs = np.linspace(0.0, node.max_vel, side)
    ws = np.linspace(-node.max_yawrate, node.max_yawrate, side)
    node._score_trajectories(1.0, 1.0, 0.0, vs, ws, 5.0, 5.0)          # warm
    ts = []
    for _ in range(15):
        t0 = time.perf_counter()
        r = node._score_trajectories(1.0, 1.0, 0.0, vs, ws, 5.0, 5.0)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), int(r[7]), int(r[8])


def reference(pr, n_traj, horizon, n_obstacles):
    cfg = pr.Config()
    cfg.dt = 0.1
    cfg.predict_time = horizon * cfg.dt
    side = int(np.sqrt(n_traj))
    cfg.v_resolution = (cfg.max_speed - cfg.min_speed) / (side - 1)
    cfg.yaw_rate_resolution = (2 * cfg.max_yaw_rate) / (side - 1)
    cfg.max_accel = 1e6                       # so the window is the full range
    cfg.max_delta_yaw_rate = 1e6
    rng = np.random.default_rng(0)
    ob = rng.uniform(0.0, 10.0, (n_obstacles, 2))
    x = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    goal = np.array([5.0, 5.0])
    dw = pr.calc_dynamic_window(x, cfg)
    pr.calc_control_and_trajectory(x, dw, cfg, goal, ob)               # warm
    ts = []
    for _ in range(15):
        t0 = time.perf_counter()
        pr.calc_control_and_trajectory(x, dw, cfg, goal, ob)
        ts.append(time.perf_counter() - t0)
    n = len(np.arange(dw[0], dw[1], cfg.v_resolution)) * \
        len(np.arange(dw[2], dw[3], cfg.yaw_rate_resolution))
    return statistics.median(ts), n


def main():
    pr = load_reference()
    _, g, _, _ = maps.hard_suite()[2]
    grid = rig.Grid(maps.inflate(g.astype(np.int16), 4))
    horizon = 25

    print(f"\nDWA rollout + score, {horizon}-step horizon, median of 15 runs\n")
    print(f"  {'trajectories':>13}{'this repo':>12}{'PythonRobotics':>17}{'per-traj':>11}{'speedup':>9}")
    for n_traj in (36, 100, 400, 900, 2500):
        t_mine, n_mine, _ = mine(n_traj, horizon, grid)
        t_ref, n_ref = reference(pr, n_traj, horizon, 60)
        um, ur = t_mine * 1e6 / n_mine, t_ref * 1e6 / n_ref
        print(f"  {n_mine:>13}{t_mine*1000:>11.2f}ms{t_ref*1000:>16.2f}ms"
              f"{um:>8.1f}us{ur/um:>8.1f}x")

    print(f"\n  scaling with obstacle count (400 trajectories):")
    print(f"  {'obstacles':>13}{'this repo':>12}{'PythonRobotics':>17}")
    t_mine, n_mine, _ = mine(400, horizon, grid)
    for n_obs in (20, 100, 500, 2000):
        t_ref, _ = reference(pr, 400, horizon, n_obs)
        print(f"  {n_obs:>13}{t_mine*1000:>11.2f}ms{t_ref*1000:>16.2f}ms")
    print("""
  This repo reads obstacles from a costmap, so its cost is flat in obstacle
  count. The reference keeps an explicit obstacle list and measures every
  rollout point against every obstacle, so it grows linearly. Neither is
  strictly better -- a costmap has to be built and maintained -- but it is the
  reason the gap widens in cluttered scenes.""")


if __name__ == "__main__":
    main()
