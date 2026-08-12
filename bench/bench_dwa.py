"""Times reactive_autonomous_nav's numpy DWA rollout+score on a fixed costmap.

Calls _score_trajectories from dwa_controller.py directly -- the same rollout,
batch costmap lookup and scoring the control loop runs at 20 Hz, minus the
marker publishing.
"""
import sys, os, time, json, statistics, types
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.maps import dump_local
from bench.bench_astar import stub_ros

DT, PREDICT, VEL_RES, YAW_RES = 0.1, 2.5, 0.02, 0.04
MAX_VEL, MAX_YAW, MAX_ACC, MAX_DYAW = 0.50, 2.00, 0.40, 1.00


def windows():
    """(label, vs, ws) for the per-cycle accel-limited window and the full space."""
    v, w = 0.25, 0.0
    tight = (np.arange(max(0.0, v - MAX_ACC * DT), min(MAX_VEL, v + MAX_ACC * DT) + VEL_RES, VEL_RES),
             np.arange(max(-MAX_YAW, w - MAX_DYAW * DT), min(MAX_YAW, w + MAX_DYAW * DT) + YAW_RES, YAW_RES))
    wide = (np.arange(0.0, MAX_VEL + VEL_RES, VEL_RES),
            np.arange(-MAX_YAW, MAX_YAW + YAW_RES, YAW_RES))
    return [("accel-limited", *tight), ("full velocity space", *wide)]


def main(reps=25):
    _sig()
    stub_ros()
    import importlib.util
    src = os.path.join(os.path.dirname(__file__), "..",
                       "reactive_autonomous_nav", "dwa_controller.py")
    spec = importlib.util.spec_from_file_location("dwa_controller", src)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

    grid, res, origin = dump_local()
    node = object.__new__(m.DWAControllerNode)
    node.costmap_info = types.SimpleNamespace(resolution=res, width=grid.shape[1], height=grid.shape[0])
    node.costmap_origin = origin
    node.costmap_data = grid
    node.predict_time, node.dt, node.max_vel = PREDICT, DT, MAX_VEL
    node.heading_cost_gain, node.speed_cost_gain, node.obstacle_cost_gain = 5.0, 0.5, 5.0

    out = []
    for label, vs, ws in windows():
        node._score_trajectories(0.0, 0.0, 0.0, vs, ws, 2.5, 0.4)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            r = node._score_trajectories(0.0, 0.0, 0.0, vs, ws, 2.5, 0.4)
            ts.append((time.perf_counter() - t0) * 1000.0)
        N, T = r[7], r[8]
        out.append({"window": label, "N": int(N), "T": int(T),
                    "ms": round(statistics.median(ts), 4), "min_ms": round(min(ts), 4)})
        print(f"  py  {label:>20}  N={N:5d} T={T}  {out[-1]['ms']:8.4f} ms")
    return out


if __name__ == "__main__":
    print("Python DWA rollout+score (dwa_controller.py::_score_trajectories, numpy)")
    json.dump(main(), open(os.path.join(os.path.dirname(__file__), "py_dwa.json"), "w"), indent=1)


def _sig():
    """Author signature. stderr, tty-only, so redirected output stays clean."""
    import os, sys
    if os.environ.get("NO_BANNER") == "1" or not sys.stderr.isatty():
        return
    print("  " + "".join(chr(c - 7) for c in
          (104,105,107,124,115,39,121,104,111,116,104,117)), file=sys.stderr)
