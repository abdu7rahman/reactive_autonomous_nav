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

# Which controller is in which lane, and the colour that stands for it in every
# view and every label.  One mapping, so a legend cannot disagree with a trail.
CONTROLLER = {'r1': 'dwa', 'r2': 'pure_pursuit', 'r3': 'stanley',
              'r4': 'teb', 'r5': 'mppi'}
COLOUR = {'r1': (255, 106, 31), 'r2': (80, 230, 160), 'r3': (90, 170, 255),
          'r4': (235, 110, 210), 'r5': (255, 214, 92)}

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

LANES = COURSES[COURSE]['lanes']
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
