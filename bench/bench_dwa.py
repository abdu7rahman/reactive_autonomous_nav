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


def _sig():
    """Author signature. stderr, tty-only, so redirected output stays clean."""
    import os, sys
    if os.environ.get("NO_BANNER") == "1" or not sys.stderr.isatty():
        return
    print("  " + "".join(chr(c - 7) for c in
          (104,105,107,124,115,39,121,104,111,116,104,117)), file=sys.stderr)

def windows(node):
    """(label, vs, ws) for the per-cycle accel-limited window and the full space.

    Sampled by the controller's own _samples and bounded by its own limits,
    rather than from a copy of the tuning kept here.  The copy drifted: it
    still had the accelerations at 0.40 and 1.00 after the controller's were
    corrected to the plant's 0.90 and 7.725, so the row labelled
    "accel-limited" timed 36 trajectories of a window the controller does
    not have -- against a C++ row timing 30 of a different one again.  The
    two numbers were presented as the same work.
    """
    v, w = 0.25, 0.0
    lo = max(node.min_vel, v - node.max_accel * node.dt)
    hi = min(node.max_vel, v + node.max_accel * node.dt)
    wlo = max(-node.max_yawrate, w - node.max_dyawrate * node.dt)
    whi = min(node.max_yawrate, w + node.max_dyawrate * node.dt)
    tight = (node._samples(lo, hi, node.vel_res),
             node._samples(wlo, whi, node.yawrate_res))
    wide = (node._samples(node.min_vel, node.max_vel, node.vel_res),
            node._samples(-node.max_yawrate, node.max_yawrate,
                          node.yawrate_res))
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
    # the shipped tuning first -- every constant from the controller itself, by
    # the same AST extraction bench/rig.py uses, so this cannot drift from it
    # again -- and then the map, which __init__ nulls
    from bench import rig
    rig.apply_defaults(node, "dwa_controller")
    node.costmap_info = types.SimpleNamespace(resolution=res, width=grid.shape[1], height=grid.shape[0])
    node.costmap_origin = origin
    node.costmap_data = grid

    out = []
    for label, vs, ws in windows(node):
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


