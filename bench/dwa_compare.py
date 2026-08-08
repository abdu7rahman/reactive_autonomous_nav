#!/usr/bin/env python3
"""Times this repo's DWA against the reference Python implementation.

Baselines are PythonRobotics' dynamic_window_approach.py (AtsushiSakai, MIT),
the most widely referenced Python DWA, and kmilo7204/dwa_planner, a standalone
one. Both are fetched at run time rather than vendored, and only their plotting
is stripped -- the planner functions are theirs.

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

HERE = os.path.dirname(os.path.abspath(__file__))
PR_URL = ("https://raw.githubusercontent.com/AtsushiSakai/PythonRobotics/"
          "master/PathPlanning/DynamicWindowApproach/dynamic_window_approach.py")
PR_CACHE = os.path.join(HERE, ".pr_dwa.py")

# kmilo7204/dwa_planner: a standalone Python DWA that keeps its tuning in a
# sibling module, so both files are fetched.
KM_URL = "https://raw.githubusercontent.com/kmilo7204/dwa_planner/master/dwa.py"
KM_CFG_URL = "https://raw.githubusercontent.com/kmilo7204/dwa_planner/master/robot_config.py"
KM_CACHE = os.path.join(HERE, ".km_dwa.py")
KM_CFG_CACHE = os.path.join(HERE, ".km_robot_config.py")


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


def load_kmilo():
    """kmilo7204/dwa_planner, fetched with only its plotting stripped."""
    if not os.path.exists(KM_CFG_CACHE):
        open(KM_CFG_CACHE, "w").write(
            urllib.request.urlopen(KM_CFG_URL, timeout=30).read().decode())
    if not os.path.exists(KM_CACHE):
        src = urllib.request.urlopen(KM_URL, timeout=30).read().decode()
        src = src.replace("import matplotlib.pyplot as plt",
                          "plt = None  # plotting stripped: only the planner is timed")
        src = src.replace("import robot_config", "import robot_config_km as robot_config")
        # an unused import that numpy removed the target of; the planner never
        # calls it, so dropping it changes nothing that is timed
        src = src.replace("from numpy.lib.npyio import load", "")
        open(KM_CACHE, "w").write(src)
    import importlib.util
    for name, path in (("robot_config_km", KM_CFG_CACHE), ("km_dwa", KM_CACHE)):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules["km_dwa"]


def kmilo(km, n_traj, horizon, n_obstacles):
    d = km.DWA()
    cfg = d.config_params
    cfg.dt = 0.1
    cfg.dw_time = horizon * cfg.dt
    side = int(np.sqrt(n_traj))
    cfg.v_res = (cfg.max_v - cfg.min_v) / side
    cfg.w_res = (cfg.max_w - cfg.min_w) / side
    cfg.max_a = 1e6                           # so the window is the full range
    cfg.max_d_w = 1e6
    rng = np.random.default_rng(0)
    cfg.obstacles = rng.uniform(0.0, 10.0, (n_obstacles, 2))
    x = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    goal = np.array([5.0, 5.0])
    d.calculate_ctrl_traj(x, goal)                                     # warm
    ts = []
    for _ in range(15):
        t0 = time.perf_counter()
        d.calculate_ctrl_traj(x, goal)
        ts.append(time.perf_counter() - t0)
    dw = d.calculate_dw(x)
    n = len(np.arange(dw[0], dw[1], cfg.v_res)) * \
        len(np.arange(dw[2], dw[3], cfg.w_res))
    return statistics.median(ts), n


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
    km = load_kmilo()
    _, g, _, _ = maps.hard_suite()[2]
    grid = rig.Grid(maps.inflate(g.astype(np.int16), 4))
    horizon = 25

    print(f"\nDWA rollout + score, {horizon}-step horizon, median of 15 runs\n")
    print(f"  {'trajectories':>13}{'this repo':>12}{'PythonRobotics':>17}{'kmilo7204':>13}"
          f"{'vs PR':>8}{'vs km':>8}")
    for n_traj in (36, 100, 400, 900, 2500):
        t_mine, n_mine, _ = mine(n_traj, horizon, grid)
        t_ref, n_ref = reference(pr, n_traj, horizon, 60)
        t_km, n_km = kmilo(km, n_traj, horizon, 60)
        um = t_mine * 1e6 / n_mine
        ur = t_ref * 1e6 / n_ref
        uk = t_km * 1e6 / max(1, n_km)
        print(f"  {n_mine:>13}{t_mine*1000:>11.2f}ms{t_ref*1000:>16.2f}ms{t_km*1000:>12.2f}ms"
              f"{ur/um:>7.1f}x{uk/um:>7.1f}x")

    print(f"\n  scaling with obstacle count (400 trajectories):")
    print(f"  {'obstacles':>13}{'this repo':>12}{'PythonRobotics':>17}{'kmilo7204':>13}")
    t_mine, n_mine, _ = mine(400, horizon, grid)
    for n_obs in (20, 100, 500, 2000):
        t_ref, _ = reference(pr, 400, horizon, n_obs)
        t_km, _ = kmilo(km, 400, horizon, n_obs)
        print(f"  {n_obs:>13}{t_mine*1000:>11.2f}ms{t_ref*1000:>16.2f}ms{t_km*1000:>12.2f}ms")
    print("""
  This repo reads obstacles from a costmap, so its cost is flat in obstacle
  count. Both references keep an explicit obstacle list and measure every
  rollout point against every obstacle, so they grow linearly -- PythonRobotics
  vectorises that inner comparison over numpy, kmilo7204 does it per point in
  Python. Neither approach is strictly better: a costmap has to be built and
  maintained, and it is only cheap because something else already paid for it.
  It is the reason the gap widens in cluttered scenes.""")


if __name__ == "__main__":
    main()
