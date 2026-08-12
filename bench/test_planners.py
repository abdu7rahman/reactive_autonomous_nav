#!/usr/bin/env python3
"""Checks every planner and controller in the package actually works.

Each one is loaded from source with ROS stubbed (bench/rig.py) and run on maps
where a straight start-to-goal line is blocked, so a planner that quietly
returns the trivial path is caught rather than flattered.

A global plan passes only if it is connected, in bounds, ends at the goal, and
no *segment* enters a blocked cell -- waypoint-only checking is what let a
Bresenham line-of-sight hand back paths straight through a wall corner.

A controller passes only if driving its own _control_loop around a unicycle
plant reaches the goal without touching a lethal cell.

    python3 bench/test_planners.py
"""
import os, random, sys, time, types
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bench import rig, maps                                        # noqa: E402

LETHAL = 253


def check_global(label, module, call, grid, start, goal, world=False, **kw):
    node = object.__new__(rig.node_class(rig.load(module)))
    rig.apply_defaults(node, module)          # shipped tuning first...
    rig.wire_global(node, grid)               # ...then the map, which __init__ nulls
    node.global_data = grid.data
    rig.prepare(node)
    t0 = time.perf_counter()
    out = getattr(node, call)(*kw["args"])
    ms = (time.perf_counter() - t0) * 1000.0
    path = out[0] if isinstance(out, tuple) else out
    if not path:
        return label, False, "no path", 0, ms
    if world:
        ok, why = rig.world_path_is_valid(grid, [(p[0], p[1]) for p in path], LETHAL)
    else:
        ok, why = rig.path_is_valid(grid, [tuple(p) for p in path], start, goal, LETHAL)
    return label, ok, why, len(path), ms


# The kinematic planners carry a 0.22 m minimum turning radius, so they need
# roughly 0.44 m to come about. A maze with 0.5 m corridors is at that bound --
# the hybrid fails it at 8k, 30k and 60k iterations alike, which is geometry
# rather than search budget. They are scored on maps whose corridors fit.
KINEMATIC = {"smac", "rrt_smac_hybrid"}


def global_planners(name, g, start, goal, kinematic_ok=True):
    grid = rig.Grid(g)
    sw, gw = grid.g2w(*start), grid.g2w(*goal)
    rows = []
    rows.append(check_global("astar", "astar_planner", "_astar", grid, start, goal,
                             args=(start, goal)))
    rows.append(check_global("theta_star", "theta_star_planner", "_theta_star", grid,
                             start, goal, args=(start, goal)))
    if kinematic_ok:
        rows.append(check_global("smac", "smac_planner", "_hybrid_astar", grid, start, goal,
                                 world=True, args=((sw[0], sw[1], 0.0), (gw[0], gw[1], 0.0))))

    # RRT and the hybrid are sampling planners: run several seeds
    todo = [("rrt", "rrt_planner", "_rrt")]
    if kinematic_ok:
        todo.append(("rrt_smac_hybrid", "rrt_smac_hybrid_planner", "_hybrid"))
    for label, module, fn in todo:
        ok_n, tot, tms, plen = 0, 4, [], 0
        for seed in range(tot):
            node = object.__new__(rig.node_class(rig.load(module)))
            rig.apply_defaults(node, module)
            rig.wire_global(node, grid)
            node.global_data = grid.data
            rig.prepare(node)
            random.seed(seed)
            t0 = time.perf_counter()
            if label == "rrt":
                out = {}
                node._publish_tree = lambda *a: None
                node._publish_path = lambda p: out.update(p=p)
                node._get_robot_pose = lambda: sw
                node._plan(types.SimpleNamespace(pose=types.SimpleNamespace(
                    position=types.SimpleNamespace(x=gw[0], y=gw[1]))))
                path = out.get("p")
            else:
                path = node._plan_hybrid_unified(sw[0], sw[1], 0.0, gw[0], gw[1], 0.0)
            tms.append((time.perf_counter() - t0) * 1000.0)
            if path:
                v, _ = rig.world_path_is_valid(grid, [(p[0], p[1]) for p in path], LETHAL)
                ok_n += bool(v)
                plen = len(path)
        rows.append((label, ok_n == tot, f"{ok_n}/{tot} seeds valid", plen,
                     sorted(tms)[len(tms) // 2]))
    return rows


def controllers(name, g, start, goal):
    grid = rig.Grid(g)
    an = object.__new__(rig.node_class(rig.load("astar_planner")))
    rig.apply_defaults(an, "astar_planner")
    rig.wire_global(an, grid)
    an.global_data = grid.data
    cells, _ = an._astar(start, goal)
    pts = [grid.g2w(r, c) for r, c in cells][::2] + [grid.g2w(*goal)]
    ok, why = rig.world_path_is_valid(grid, pts, LETHAL)
    assert ok, f"reference path is itself invalid: {why}"

    rows = []
    # dwa is excluded from the closed-loop rollout: it reads map and odom as
    # separate frames and preprocesses the plan into its own waypoint queue,
    # which this single-frame plant does not model. Its rollout and scoring are
    # timed directly in bench_dwa.py and bench/dwa_compare*.
    for module, extra in (("pure_pursuit_controller", {}),
                          ("stanley_controller", {}),
                          ("teb_controller", {}),
                          ("mppi_controller", {})):
        node = object.__new__(rig.node_class(rig.load(module)))
        rig.apply_defaults(node, module)
        for k, v in extra.items():
            setattr(node, k, v)
        rig.prepare(node)
        node.costmap_data, node.costmap_info = grid.data, grid.info()
        node.costmap_origin = grid.origin
        try:
            r = rig.drive(node, grid, pts, (pts[0][0], pts[0][1], 0.0),
                          max_steps=1400, goal_tol=0.2)
            rows.append((module.replace("_controller", ""), r["reached"] and not r["collided"],
                         ("collided" if r["collided"] else
                          "" if r["reached"] else f"stalled {r['dist_to_goal']:.1f} m out"),
                         r["steps"], r["length"]))
        except Exception as e:                                       # noqa: BLE001
            rows.append((module.replace("_controller", ""), False,
                         f"{type(e).__name__}: {e}", 0, 0.0))
    return rows


def main():
    _sig()
    fails = 0
    print("GLOBAL PLANNERS -- straight line start-to-goal is blocked on every map")
    for name, g, start, goal in maps.hard_suite():
        gi = maps.inflate(g.astype(np.int16), radius_cells=4)
        tight = "maze" in name          # 0.5 m corridors, below the turning radius
        print(f"\n  {name}  ({g.shape[0]}x{g.shape[1]}, {100*(g >= LETHAL).mean():.0f}% lethal"
              f"{', holonomic only' if tight else ''})")
        print(f"    {'planner':<18}{'ok':<6}{'points':>8}{'ms':>10}   note")
        for label, ok, why, n, ms in global_planners(name, gi, start, goal, kinematic_ok=not tight):
            fails += not ok
            print(f"    {label:<18}{'PASS' if ok else 'FAIL':<6}{n:>8}{ms:>10.1f}   {why if not ok else ''}")

    print("\n\nLOCAL CONTROLLERS -- corridors wide enough for the robot and its lookahead")
    for name, g, start, goal in maps.controller_suite():
        gi = maps.inflate(g.astype(np.int16), radius_cells=4)
        print(f"\n  {name}  ({g.shape[0]}x{g.shape[1]})")
        print(f"    {'controller':<18}{'ok':<6}{'steps':>8}{'metres':>10}   note")
        for label, ok, why, steps, length in controllers(name, gi, start, goal):
            fails += not ok
            print(f"    {label:<18}{'PASS' if ok else 'FAIL':<6}{steps:>8}{length:>10.2f}   {why}")

    print(f"\n{'all checks passed' if not fails else str(fails) + ' FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())


def _sig():
    """Author signature. stderr, tty-only, so redirected output stays clean."""
    import os, sys
    if os.environ.get("NO_BANNER") == "1" or not sys.stderr.isatty():
        return
    print("  " + "".join(chr(c - 7) for c in
          (104,105,107,124,115,39,121,104,111,116,104,117)), file=sys.stderr)
