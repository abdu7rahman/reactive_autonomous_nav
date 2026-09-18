"""The race course: where the lanes are, what is on them, and who is in them.

Data only -- no ROS, no node, nothing to launch.  It lives in the package
rather than beside the harness because both sides need it and neither owns it:
launch/race_launch.py reads which controller belongs in which lane, and
sim/grid_tf.py, sim/race_path.py, sim/race_timer.py and sim/race_rviz.py read
the lanes, the walls, the colours and the length.  Two declarations of the same
numbers is two chances for a robot to be somewhere the rest of the stack
believes it is not, and the first five-robot race lost a full recorded run that
way -- a lane value in a shell script that no longer matched the odom pin.

The course is chosen by the RACE_COURSE environment variable, which race_up.sh
and race.sh export so every process started under them agrees without an
argument being threaded through six files.

  straight   five clear lanes 1.3 m apart.  Each robot runs its own A* and its
             own costmaps, so the race is planner-plus-controller end to end,
             which is what a nav stack actually is.
  chicane    five lanes 1.7 m apart with three wall segments each, and one
             reference path (sim/race_path.py) handed to all five controllers
             instead of five separate plans.  On a curve the thing worth
             comparing is how a controller tracks, and five A* runs on five
             rolling costmaps do not produce the same path twice -- different
             lidar views, different replans, different corners cut.  Same path,
             five trackers, and the difference is the tracker.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import math
import os

# Who is in which lane.  RACE_FIELD picks the field; a lane is either one of
# this package's own controller nodes ('pkg') or a nav2 controller plugin
# hosted in its own controller_server ('nav2').
#
#   ours   the five controllers in this package, one per lane.  What differs
#          is the method.
#   versus this package's DWA in both languages against nav2's own: the C++
#          controller in cpp/src/dwa_controller.cpp, the Python one in
#          reactive_autonomous_nav/dwa_controller.py, and three nav2 plugins.
#          Five implementations of local control, one robot design, one
#          costmap configuration, one path.
#   nav2   this package's DWA -- the quickest of the five on both courses --
#          against nav2's own local controllers, same costmap settings, same
#          reference path, same robot.  bench/README.md says of
#          nav2_dwb_controller that it "needs a live ROS 2 graph and costmap
#          plugins to run at all, so any number taken outside that would be
#          measuring the harness", and uses nav2's published figures instead.
#          This is that live graph.
FIELDS = {
    'ours': {
        'r1': ('pkg', 'dwa'),
        'r2': ('pkg', 'pure_pursuit'),
        'r3': ('pkg', 'stanley'),
        'r4': ('pkg', 'teb'),
        'r5': ('pkg', 'mppi'),
    },
    'versus': {
        'r1': ('cpp', 'dwa'),
        'r2': ('pkg', 'dwa'),
        'r3': ('nav2', 'dwb_core::DWBLocalPlanner'),
        'r4': ('nav2', 'nav2_mppi_controller::MPPIController'),
        'r5': ('nav2', 'nav2_regulated_pure_pursuit_controller::'
                       'RegulatedPurePursuitController'),
    },
    'nav2': {
        'r1': ('pkg', 'dwa'),
        'r2': ('nav2', 'dwb_core::DWBLocalPlanner'),
        'r3': ('nav2', 'nav2_mppi_controller::MPPIController'),
        'r4': ('nav2', 'nav2_regulated_pure_pursuit_controller::'
                       'RegulatedPurePursuitController'),
        'r5': ('nav2', 'nav2_graceful_controller::GracefulController'),
    },
    # The two bench fields put bench/README.md's seven-way comparison on the
    # robot. They are two clips rather than one because the chicane is five
    # lanes wide and the comparison is seven implementations: the free
    # rectangle the course was measured into is 10.0 m across at its longest
    # (see COURSES below), and five 1.7 m lanes already use 8.5 m of it, so a
    # seventh lane is 11.9 m of course in a 10.0 m hall. Splitting on language
    # rather than arbitrarily keeps each clip to one comparison a reader can
    # hold, and puts this package's own DWA in both as the common lane.
    #
    # bench-py   this package's DWA in both languages against the two
    #            reference Python implementations, which is the
    #            dwa_compare.py table driven rather than timed.
    # bench-cpp  the C++ controller against the three C and C++ baselines,
    #            which is dwa_compare_cpp.
    'bench-py': {
        'r1': ('cpp', 'dwa'),
        'r2': ('pkg', 'dwa'),
        'r3': ('bench', 'pythonrobotics'),
        'r4': ('bench', 'kmilo7204'),
    },
    'bench-cpp': {
        'r1': ('cpp', 'dwa'),
        'r2': ('bench', 'cpprobotics'),
        'r3': ('bench', 'goktug97'),
        'r4': ('bench', 'amslabtech'),
    },
}

FIELD = os.environ.get('RACE_FIELD', 'ours')
if FIELD not in FIELDS:
    raise SystemExit(f'RACE_FIELD={FIELD!r} is not one of {sorted(FIELDS)}')
ENTRANTS = FIELDS[FIELD]

# What each lane is called in the scene, the table and the log.  Short, because
# it is drawn over a robot in a 560 px clip: the plugin's own class name is the
# thing that identifies it, so the label is its last component with the
# repeated package prefix taken off.
# One token each, no spaces.  RViz's TEXT_VIEW_FACING breaks a label on its
# spaces, so "dwa c++" and "nav2 pursuit" came out stacked on two lines and
# crowded the lanes either side of them at 560 px.
_SHORT = {'DWBLocalPlanner': 'nav2-dwb',
          'MPPIController': 'nav2-mppi',
          'RegulatedPurePursuitController': 'nav2-pursuit',
          'GracefulController': 'nav2-graceful'}

# A baseline lane is labelled with whose implementation it is, because that is
# the only thing distinguishing it: same robot, same costmap, same path, same
# plant limits, same sampling resolution. One token each, for the same reason
# as above -- RViz breaks a TEXT_VIEW_FACING label on its spaces.
_BENCH = {'pythonrobotics': 'PythonRobotics', 'kmilo7204': 'kmilo7204',
          'cpprobotics': 'CppRobotics', 'goktug97': 'goktug97',
          'amslabtech': 'amslabtech'}


_MIXED = any(k == 'cpp' for k, _s in ENTRANTS.values())


def _label(kind, spec):
    """What the lane is called in the scene, the table and the log.

    Short, because it is drawn over a robot in a 560 px clip: a nav2 plugin's
    class name is what identifies it, so the label is its last component with
    the repeated package prefix taken off.  In a field that races both of this
    package's DWAs the Python one says so -- "dwa" beside "dwa c++" names the
    language of one and not the other, which is the one thing a reader of that
    clip needs to tell them apart.
    """
    if kind == 'cpp':
        return f'{spec}-c++'
    if kind == 'pkg':
        return f'{spec}-py' if _MIXED else spec
    if kind == 'bench':
        return _BENCH[spec]
    return _SHORT[spec.rsplit('::', 1)[1]]


CONTROLLER = {ns: _label(*e) for ns, e in ENTRANTS.items()}

# The colour that stands for a lane in every view and every label.  One
# mapping, so a legend cannot disagree with a trail.
COLOUR = {'r1': (255, 106, 31), 'r2': (80, 230, 160), 'r3': (90, 170, 255),
          'r4': (235, 110, 210), 'r5': (255, 214, 92)}


def check_field():
    """A nav2 lane needs a course that hands out a path.

    nav2's controller_server is a controller and nothing else: the only way to
    give it a path is a FollowPath action goal, and on the straight course each
    lane is expected to plan its own way to a goal with its own A*.  Racing the
    two arrangements against each other would be racing a planner against no
    planner, so it is refused here rather than measured.
    """
    if nav2_lanes() and not GATES:
        raise SystemExit(
            f'RACE_FIELD={FIELD} races nav2 controller plugins, which take a '
            f'path and cannot plan one, but RACE_COURSE={COURSE} expects each '
            f'lane to plan its own. Use a course with gates.')


def nav2_lanes():
    """The lanes whose controller is a nav2 plugin, in lane order."""
    return [ns for ns, (kind, _spec) in ENTRANTS.items() if kind == 'nav2']


def own_lanes():
    """The lanes running a controller node of this package's own launching.

    A 'bench' lane is somebody else's planner inside this package's
    baseline_controller node, so it comes up and is wired exactly like a 'pkg'
    lane -- same costmap, same remaps, same /plan -- and only the scoring
    function inside it is theirs.
    """
    return [ns for ns, (kind, _spec) in ENTRANTS.items()
            if kind in ('pkg', 'cpp', 'bench')]

# The two courses.
#
# straight: the original, kept exactly as it was recorded.  Lanes 1.3 m apart
# on the corridor measured clear from y = -1.6 to 7.8; the robot is 0.34 m
# across and carries a 0.30 m inflation, so anything tighter starts the race
# with every robot inside its neighbour's forbidden zone.
#
# chicane: wider lanes, because a lane has to hold the robot plus 0.30 m of
# inflation on each side -- 0.94 m of corridor that stays clear -- so a wall
# that blocks the middle of a lane can be at most 0.76 m wide at 1.7 m spacing.
# Searched the warehouse's occupancy map for the tallest all-free rectangle at
# each width: 4.5 m gives 29.4 m, 7.0 m gives 14.9 m, 10.0 m gives 8.8 m at
# x -7.06..3.02, y -0.73..8.03.  Five 1.7 m lanes centred on x = -2.0 fit that
# with 1.6 m to spare.  Start line at y = 0: at y = -0.4 the outermost lane's
# swung extreme came within 0.54 m of the standing person at (1.0, -1.0),
# inside the inflation, and at y = 0 the tightest point of the whole swept
# corridor is 0.93 m from the nearest occupied or unknown cell, with 2.0 m of
# run-off past the finish -- more than MPPI's 1.3 m horizon.
COURSES = {
    'straight': {
        'lanes': {'r1': -5.6, 'r2': -4.3, 'r3': -3.0, 'r4': -1.7, 'r5': -0.4},
        'start_y': -1.0,
        'length': 6.0,
        'gates': [],
    },
    'chicane': {
        'lanes': {'r1': -5.4, 'r2': -3.7, 'r3': -2.0, 'r4': -0.3, 'r5': 1.4},
        'start_y': 0.0,
        'length': 6.0,
        # (y of the wall, offset of its centre from the lane centre).
        # Alternating signs make the free corridor swap sides at each gate, so
        # the route is a slalom rather than one long bend, and every lane's
        # three walls are identical translations of each other.
        'gates': [(0.9, -0.45), (2.6, +0.45), (4.3, -0.45)],
    },
}

COURSE = os.environ.get('RACE_COURSE', 'chicane')
if COURSE not in COURSES:
    raise SystemExit(f'RACE_COURSE={COURSE!r} is not one of {sorted(COURSES)}')

# The lanes the field actually fills. A course declares five, and the two
# bench fields enter four, so the grid, the gates, the RViz framing, the odom
# pins and the timer all follow the entrants rather than the course -- five
# robots spawned for four controllers is a robot parked on the start line for
# the whole race, and the fifth lane's gates standing in an empty lane.
#
# Contiguous from r1, which is what wall_poses() needs: it closes the outside
# of the grid by putting the wall a missing neighbour would have, and a gap in
# the middle would leave two lanes with an open gate on one side. Four lanes
# is 6.8 m of the 10.0 m free width the course was measured into, so it fits
# wherever five did.
LANES = {ns: x for ns, x in COURSES[COURSE]['lanes'].items()
         if ns in ENTRANTS}
if list(LANES) != list(COURSES[COURSE]['lanes'])[:len(LANES)]:
    raise SystemExit(f'RACE_FIELD={FIELD} fills {sorted(LANES)}, which is not '
                     f'the first {len(LANES)} lanes of RACE_COURSE={COURSE}')
START_Y = COURSES[COURSE]['start_y']
RACE_LENGTH = COURSES[COURSE]['length']
GATES = COURSES[COURSE]['gates']

YAW = math.pi / 2          # up the course, +y in the warehouse
SPACING = (sorted(LANES.values())[1] - sorted(LANES.values())[0]) if len(LANES) > 1 else 0.0

GATE_W = 0.60              # along x, the blocking span
GATE_T = 0.15              # along y, thin enough not to be a corridor itself
GATE_H = 0.60              # tall enough for a lidar 0.25 m off the floor

ROBOT_RADIUS = 0.22        # config/race_costmap_params.yaml
INFLATION = 0.30           # config/race_costmap_params.yaml
COSTMAP_RES = 0.05         # config/race_costmap_params.yaml


def wall_poses():
    """Every chicane wall, as (x of its centre, y of its centre).

    One per lane per gate, plus one edge wall per gate.  Without the edge walls
    the middle three robots pass through a 1.10 m gap between their own wall
    and their neighbour's while the outer two find one gate each with nothing
    on the far side -- measured, twelve corridors at 1.10 m and three open.  An
    open gate is a wider gate, and the point of translating one chicane across
    five lanes is that the lane must not be the variable.  The outermost lane
    gets the wall its missing neighbour would have put there.
    """
    if not GATES:
        return []
    lanes = sorted(LANES.values())
    out = []
    for gy, off in GATES:
        for lx in lanes:
            out.append((lx + off, gy))
        edge = (lanes[-1] + SPACING) if off < 0 else (lanes[0] - SPACING)
        out.append((edge + off, gy))
    return out


def wall_distance(x, y):
    """Distance from (x, y) in `map` to the nearest wall surface, in metres.

    Zero inside a wall.  The walls are axis-aligned boxes, so this is the exact
    point-to-box distance rather than a centre-to-centre approximation: at a
    gate the difference is the 0.30 m half-width of the wall the robot is
    squeezing past, which is most of the clearance being measured.
    """
    best = float('inf')
    for wx, wy in wall_poses():
        dx = max(0.0, abs(x - wx) - GATE_W / 2.0)
        dy = max(0.0, abs(y - wy) - GATE_T / 2.0)
        best = min(best, math.hypot(dx, dy))
    return best


def gate_corridor(gy, off):
    """The free span at one gate, as (lo, hi) offsets from a lane's centre.

    A gate's wall blocks `off` +/- GATE_W/2 of its own lane; the corridor runs
    from that edge to the edge of the neighbouring lane's wall, which sits one
    lane spacing away on the side the offset points at.  The neighbour is
    always there -- that is what the edge walls in wall_poses() are for -- so
    this is the same span for every lane.
    """
    if off < 0:
        return off + GATE_W / 2.0, off + SPACING - GATE_W / 2.0
    return off - SPACING + GATE_W / 2.0, off - GATE_W / 2.0


check_field()
