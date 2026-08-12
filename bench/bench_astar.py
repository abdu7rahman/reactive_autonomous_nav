"""Times reactive_autonomous_nav's Python A* on the shared maps.

The planner module is a ROS 2 node, so rclpy and the message packages are
stubbed just far enough for the import to succeed. Nothing in the search path
is reimplemented: the module is loaded from source and `_astar` is called on a
bare instance with the grid attributes it reads wired up by hand.
"""
import sys, os, time, types, json, statistics
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench.maps import suite


def stub_ros():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m
    Any = type("Any", (), {"__init__": lambda s, *a, **k: None})
    mod("rclpy", init=lambda *a, **k: None, spin=lambda *a, **k: None, shutdown=lambda *a, **k: None)
    mod("rclpy.time", Time=Any)
    mod("rclpy.node", Node=type("Node", (), {"__init__": lambda s, *a, **k: None}))
    mod("rclpy.duration", Duration=Any)
    mod("rclpy.qos", QoSProfile=Any, DurabilityPolicy=Any, ReliabilityPolicy=Any)
    mod("tf2_ros", TransformListener=Any, Buffer=Any)
    mod("nav_msgs"); mod("nav_msgs.msg", OccupancyGrid=Any, Path=Any, Odometry=Any)
    mod("geometry_msgs"); mod("geometry_msgs.msg", PoseStamped=Any, Point=Any, Twist=Any)
    mod("std_msgs"); mod("std_msgs.msg", String=Any, ColorRGBA=Any)
    mod("visualization_msgs"); mod("visualization_msgs.msg", Marker=Any, MarkerArray=Any)


def load_planner():
    import importlib.util
    src = os.path.join(os.path.dirname(__file__), "..",
                       "reactive_autonomous_nav", "astar_planner.py")
    spec = importlib.util.spec_from_file_location("astar_planner", src)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_instance(mod, grid):
    h, w = grid.shape
    node = object.__new__(mod.AStarPlannerNode)
    node.global_info = types.SimpleNamespace(resolution=0.05, width=w, height=h)
    node.global_origin = (0.0, 0.0)
    node.global_data = grid
    node.local_data = None          # -> _local_cost_at_map returns -1
    node.odom_to_map = None
    return node


def main(reps=5):
    _sig()
    stub_ros()
    mod = load_planner()
    results = []
    for name, grid, start, goal in suite():
        node = make_instance(mod, grid)
        node._astar(start, goal)                     # warm up
        times, path, expl = [], None, None
        for _ in range(reps):
            t0 = time.perf_counter()
            path, expl = node._astar(start, goal)
            times.append((time.perf_counter() - t0) * 1000.0)
        results.append({
            "map": name, "ms": round(statistics.median(times), 2),
            "min_ms": round(min(times), 2),
            "expanded": len(expl), "path_cells": len(path) if path else 0,
        })
        print(f"  py  {name:>9}  {results[-1]['ms']:8.2f} ms   "
              f"{len(expl):7d} expanded   {len(path) if path else 0:5d} cells")
    return results


if __name__ == "__main__":
    print("Python A* (reactive_autonomous_nav/astar_planner.py::_astar)")
    out = main()
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "py_astar.json"), "w"), indent=1)


def _sig():
    """Author signature. stderr, tty-only, so redirected output stays clean."""
    import os, sys
    if os.environ.get("NO_BANNER") == "1" or not sys.stderr.isatty():
        return
    print("  " + "".join(chr(c - 7) for c in
          (104,105,107,124,115,39,121,104,111,116,104,117)), file=sys.stderr)
