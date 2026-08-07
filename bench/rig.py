"""Loads any planner/controller node from the package and runs it without ROS.

Same idea as bench_astar.py: rclpy and the message packages are stubbed only
far enough for the import to succeed, then the node is built with
object.__new__ and the grid/state attributes it reads are wired up by hand.
Everything below that is the module's own code.
"""
import io, os, sys, types, contextlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "reactive_autonomous_nav")


def stub_ros():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        if "." in name:
            parent, child = name.rsplit(".", 1)
            if parent in sys.modules:
                setattr(sys.modules[parent], child, m)
        return m

    class _AnyMeta(type):
        # message types carry class-level constants (Marker.SPHERE, ...)
        def __getattr__(cls, name):
            return 0

    class Any(metaclass=_AnyMeta):
        def __init__(self, *a, **k): pass
        def __getattr__(self, n): return Any()
        def __call__(self, *a, **k): return Any()
        def __setattr__(self, n, v): object.__setattr__(self, n, v)

    mod("rclpy", init=lambda *a, **k: None, spin=lambda *a, **k: None,
        shutdown=lambda *a, **k: None, ok=lambda *a, **k: True)
    mod("rclpy.time", Time=Any)
    mod("rclpy.duration", Duration=Any)
    mod("rclpy.node", Node=type("Node", (), {"__init__": lambda s, *a, **k: None}))
    mod("rclpy.qos", QoSProfile=Any, DurabilityPolicy=Any, ReliabilityPolicy=Any)
    mod("rclpy.callback_groups", ReentrantCallbackGroup=Any)
    mod("rclpy.executors", MultiThreadedExecutor=Any)
    mod("tf2_ros", TransformListener=Any, Buffer=Any)
    class Vec3:
        def __init__(self, *a, **k): self.x = self.y = self.z = 0.0

    class Twist:
        """Real fields, because controllers read back cmd.angular.z to clamp it."""
        def __init__(self, *a, **k): self.linear, self.angular = Vec3(), Vec3()

    for n, syms in (
        ("nav_msgs", ()), ("nav_msgs.msg", ("OccupancyGrid", "Path", "Odometry")),
        ("geometry_msgs", ()), ("geometry_msgs.msg", ("PoseStamped", "Point", "Pose", "Quaternion")),
        ("std_msgs", ()), ("std_msgs.msg", ("String", "ColorRGBA", "Header")),
        ("visualization_msgs", ()), ("visualization_msgs.msg", ("Marker", "MarkerArray")),
        ("builtin_interfaces", ()), ("builtin_interfaces.msg", ("Time",)),
    ):
        m = mod(n)
        for s_ in syms:
            setattr(m, s_, Any)
    gm = sys.modules["geometry_msgs.msg"]
    gm.Twist = Twist          # real fields: controllers read back cmd.angular.z
    gm.Vector3 = Vec3
    return Any


ANY = stub_ros()


def load(module_name):
    """Import a module from the package with ROS stubbed out."""
    import importlib.util
    src = os.path.join(PKG, module_name + ".py")
    spec = importlib.util.spec_from_file_location(module_name, src)
    m = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(m)
    return m


def node_class(module):
    """The rclpy node class, not the plain data classes beside it."""
    best = None
    for name in dir(module):
        obj = getattr(module, name)
        if not (isinstance(obj, type) and obj.__module__ == module.__name__):
            continue
        if name.endswith(("PlannerNode", "ControllerNode")):
            return obj
        if name.endswith("Node") and best is None:
            best = obj
    if best is None:
        raise LookupError("no node class in " + module.__name__)
    return best


class Grid:
    """A costmap plus the world<->grid conversion the planners assume."""

    def __init__(self, data, resolution=0.05, origin=(0.0, 0.0)):
        self.data = data
        self.h, self.w = data.shape
        self.resolution = resolution
        self.origin = origin

    def info(self):
        return types.SimpleNamespace(resolution=self.resolution, width=self.w, height=self.h)

    def w2g(self, wx, wy):
        return (int((wy - self.origin[1]) / self.resolution),
                int((wx - self.origin[0]) / self.resolution))

    def g2w(self, r, c):
        return ((c + 0.5) * self.resolution + self.origin[0],
                (r + 0.5) * self.resolution + self.origin[1])

    def free(self, r, c, lethal=253):
        if not (0 <= r < self.h and 0 <= c < self.w):
            return False
        v = int(self.data[r, c])
        return 0 <= v < lethal


def wire_global(node, grid):
    """Attributes every global planner in this package reads."""
    node.global_data = grid.data
    node.global_info = grid.info()
    node.global_origin = grid.origin
    node.local_data = None
    node.local_info = None
    node.local_origin = (0.0, 0.0)
    node.odom_to_map = None
    node.current_path = None
    return node


def segment_clear(grid, a, b, lethal=253, samples_per_cell=8):
    """Does the straight segment a->b stay out of every blocked cell's interior?

    Cell (r, c) owns the square [r-0.5, r+0.5] x [c-0.5, c+0.5], so the cell
    containing a continuous point is round(). Samples are offset off the exact
    midpoint because a pure diagonal step grazes a cell *corner* at t = 0.5 --
    that touches the corner without entering the interior, and rounding there
    would flag a legal move.
    """
    (r0, c0), (r1, c1) = a, b
    span = max(abs(r1 - r0), abs(c1 - c0))
    n = max(4, int(span * samples_per_cell))
    for i in range(n):
        t = (i + 0.37) / n          # 0.37 keeps samples off exact cell corners,
        r = int(round(r0 + (r1 - r0) * t))   # where a diagonal only grazes and
        c = int(round(c0 + (c1 - c0) * t))   # does not enter the interior
        if not grid.free(r, c, lethal):
            return False, (r, c)
    return True, None


def diagonal_corner_cut(grid, a, b, lethal=253):
    """True when a one-cell diagonal step slips between two blocked cells.

    Only meaningful for a single step. On a longer segment the two cells this
    would test are nowhere near the line, so the check does not apply.
    """
    (r0, c0), (r1, c1) = a, b
    if abs(r0 - r1) != 1 or abs(c0 - c1) != 1:
        return False
    return not grid.free(r0, c1, lethal) and not grid.free(r1, c0, lethal)


def path_is_valid(grid, cells, start, goal, lethal=253, max_step=None):
    """Connected, in bounds, collision-free, and actually ends at the goal."""
    if not cells:
        return False, "empty path"
    if cells[0] != tuple(start):
        return False, f"starts at {cells[0]}, not {tuple(start)}"
    if cells[-1] != tuple(goal):
        return False, f"ends at {cells[-1]}, not {tuple(goal)}"
    for r, c in cells:
        if not grid.free(r, c, lethal):
            return False, f"cell {(r, c)} is lethal or out of bounds"
    if max_step:
        for a, b in zip(cells, cells[1:]):
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) > max_step:
                return False, f"jump from {a} to {b}"
    for a, b in zip(cells, cells[1:]):
        if diagonal_corner_cut(grid, a, b, lethal):
            return False, f"corner cut {a} -> {b}"
        ok, hit = segment_clear(grid, a, b, lethal)
        if not ok:
            return False, f"segment {a} -> {b} enters blocked cell {hit}"
    return True, "ok"


def world_path_is_valid(grid, pts, lethal=253):
    """Same, for planners that return world-frame waypoints."""
    if not pts:
        return False, "empty path"
    for wx, wy in pts:
        r, c = grid.w2g(wx, wy)
        if not grid.free(r, c, lethal):
            return False, f"waypoint {(round(wx,2), round(wy,2))} -> cell {(r,c)} blocked"
    # densely resample each segment and re-check
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        n = max(2, int(np.hypot(x2 - x1, y2 - y1) / (grid.resolution * 0.5)))
        for t in np.linspace(0, 1, n):
            r, c = grid.w2g(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
            if not grid.free(r, c, lethal):
                return False, f"segment {(round(x1,2),round(y1,2))}->{(round(x2,2),round(y2,2))} clips {(r,c)}"
    return True, "ok"


# ── controllers ──────────────────────────────────────────────────────

def apply_defaults(node, module_name):
    """Set every `self.x = <literal>` default the node's __init__ assigns.

    Reading them out of the source keeps the harness honest: the controller is
    exercised on its shipped tuning, and the test cannot silently drift from it.
    """
    import ast
    src = open(os.path.join(PKG, module_name + ".py")).read()
    tree = ast.parse(src)
    for cls_node in ast.walk(tree):
        if not isinstance(cls_node, ast.ClassDef):
            continue
        for fn in cls_node.body:
            if not (isinstance(fn, ast.FunctionDef) and fn.name == "__init__"):
                continue
            for stmt in ast.walk(fn):
                if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                    continue
                tgt = stmt.targets[0]
                if not (isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"):
                    continue
                try:
                    setattr(node, tgt.attr, ast.literal_eval(stmt.value))
                    continue
                except (ValueError, TypeError):
                    pass          # publishers, timers, tf -- not literals
                # deque(maxlen=N) and friends are state the loop reads back
                if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name):
                    fname = stmt.value.func.id
                    if fname == "deque":
                        from collections import deque
                        ml = next((ast.literal_eval(k.value) for k in stmt.value.keywords
                                   if k.arg == "maxlen"), None)
                        setattr(node, tgt.attr, deque(maxlen=ml))
                    elif fname in ("list", "dict", "set"):
                        setattr(node, tgt.attr, {"list": [], "dict": {}, "set": set()}[fname])
                    else:
                        # message constructors (Path(), Twist(), ...) -- the loop
                        # only ever appends to or overwrites these
                        setattr(node, tgt.attr, types.SimpleNamespace(
                            poses=[], header=types.SimpleNamespace(frame_id="map"),
                            data=[], points=[]))
    return node

def make_path(pts):
    """A nav_msgs/Path stand-in with the attribute chain controllers read."""
    poses = []
    for x, y in pts:
        p = types.SimpleNamespace()
        p.pose = types.SimpleNamespace(
            position=types.SimpleNamespace(x=float(x), y=float(y), z=0.0),
            orientation=types.SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0))
        p.header = types.SimpleNamespace(frame_id="map")
        poses.append(p)
    return types.SimpleNamespace(poses=poses,
                                 header=types.SimpleNamespace(frame_id="map"))


class Sink:
    """Captures whatever a node publishes."""
    def __init__(self): self.msgs = []
    def publish(self, m): self.msgs.append(m)


def prepare(node):
    """Give a bare node the plumbing every _control_loop touches.

    Publishers become sinks and the clock is a stub, so marker publishing runs
    without ROS instead of being patched out -- the control path stays intact.
    """
    node.get_logger = lambda: types.SimpleNamespace(
        info=lambda *a, **k: None, warn=lambda *a, **k: None,
        error=lambda *a, **k: None, debug=lambda *a, **k: None)
    stamp = types.SimpleNamespace(to_msg=lambda: types.SimpleNamespace(sec=0, nanosec=0))
    node.get_clock = lambda: types.SimpleNamespace(now=lambda: stamp)
    # every `self.*_pub = self.create_publisher(...)` in __init__ becomes a sink,
    # read out of the source so a new publisher cannot break the harness
    import ast, re as _re
    src = ""
    for mod_name in (getattr(type(node), "__module__", ""),):
        cand = os.path.join(PKG, mod_name + ".py")
        if os.path.exists(cand):
            src = open(cand).read()
    names = set(_re.findall(r"self\.(\w+)\s*=\s*self\.create_publisher", src))
    names |= set(_re.findall(r"self\.(\w*pub\w*)\s*=", src)) | {"cmd_pub", "status_pub"}
    for name in names:
        if not isinstance(getattr(node, name, None), Sink):
            setattr(node, name, Sink())
    return node


def drive(node, grid, path_pts, start_pose, dt=0.1, max_steps=1500,
          goal_tol=0.25, cmd_attr="cmd_pub"):
    """Closed-loop rollout: the controller's own _control_loop, unicycle plant.

    Returns a dict with reached / collided / steps / path length / min clearance.
    """
    x, y, yaw = start_pose
    cmd = Sink()
    setattr(node, cmd_attr, cmd)
    msg = make_path(path_pts)
    # go through the node's own path callback where it has one, so derived
    # state (numpy caches, elastic band, control sequence) is built the way it
    # is at runtime instead of being half-initialised by the test
    if hasattr(node, "_path_cb"):
        node._path_cb(msg)
    if getattr(node, "current_path", None) is None:
        node.current_path = msg
    node.goal_reached = False

    def tf(target, source, p=(x, y, yaw)):
        """Frame-aware stub.

        Returning the robot pose for every frame pair is wrong for any node
        that asks for odom<-map: it would transform the goal by the robot's own
        pose. Here base_link resolves to the pose and odom/map are aligned,
        which is what they are before localisation drifts.
        """
        return p if "base_link" in (target, source) else (0.0, 0.0, 0.0)

    node._get_robot_pose = lambda: (x, y, yaw)
    node._get_tf = tf

    gx, gy = path_pts[-1]
    travelled = 0.0
    collided = False
    steps = 0
    vmax = 0.0
    for steps in range(1, max_steps + 1):
        pose_now = (x, y, yaw)
        node._get_robot_pose = lambda p=pose_now: p
        node._get_tf = lambda target, source, p=pose_now: tf(target, source, p)
        node.current_pose = types.SimpleNamespace(x=x, y=y, yaw=yaw)
        before = len(cmd.msgs)
        node._control_loop()
        if node.goal_reached:
            break
        if len(cmd.msgs) == before:
            v = w = 0.0
        else:
            m = cmd.msgs[-1]
            v = float(getattr(m.linear, "x", 0.0))
            w = float(getattr(m.angular, "z", 0.0))
        # close the odometry loop: a node that sizes its window off the
        # measured velocity has to be told what the plant actually did
        if isinstance(getattr(node, "current_vel", None), dict):
            node.current_vel = {"v": v, "omega": w}
        vmax = max(vmax, abs(v))
        nx = x + v * math.cos(yaw) * dt
        ny = y + v * math.sin(yaw) * dt
        yaw = (yaw + w * dt + math.pi) % (2 * math.pi) - math.pi
        travelled += math.hypot(nx - x, ny - y)
        x, y = nx, ny
        r, c = grid.w2g(x, y)
        if not grid.free(r, c):
            collided = True
            break
        if math.hypot(x - gx, y - gy) < goal_tol:
            node.goal_reached = True
            break

    return {"reached": bool(node.goal_reached), "collided": collided,
            "steps": steps, "length": travelled, "vmax": vmax,
            "final": (x, y, yaw),
            "dist_to_goal": math.hypot(x - gx, y - gy)}


import math  # noqa: E402  (used by drive)
