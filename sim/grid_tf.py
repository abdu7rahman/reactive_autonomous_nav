"""Pin each robot's odom origin into the shared `map` frame.

    grid_tf.py              publish them
    grid_tf.py --lanes      print the grid as shell assignments
    grid_tf.py --gates      print one `gz service` request per chicane wall

One node publishing all five map -> <ns>/odom transforms on /tf at 10 Hz,
which is where a localiser publishes map -> odom and at what sort of rate.
The first version of this used five `static_transform_publisher` processes on
/tf_static and it did not work: with fifteen transient-local publishers on that
topic (five robots' robot_state_publisher trees, five odom pins, five lidar
identities) a freshly started listener received some of the latched samples and
not others.  Measured with nothing but tf2 involved -- `tf2_echo map
r1/base_link` resolved, and r2 through r5 all reported "Could not find a
connection ... Tf has two or more unconnected trees" -- so it is delivery, not
the transforms.  Downstream that read as nav2 costmaps failing to activate:
"Invalid frame ID map passed to canTransform target_frame - frame does not
exist", one to four robots per attempt, different ones each time.

The values are exact rather than estimated.  Each robot's odom frame is created
where the robot was spawned, and race_up.sh chose those poses, so this is the
whole of the localisation the race needs and there is no filter in it to drift.
It is the five-robot version of what localize.py does for a single robot.

This file is where the course is defined, once: the lanes, the start line, the
race length and the chicane.  race_up.sh spawns the robots and the walls from
it and race_timer.py places each goal from it, rather than any of them keeping
a copy -- three declarations of the same numbers is three chances for a robot
to be spawned somewhere the rest of the stack believes it is not.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

# Where the grid sits, measured against the warehouse's own occupancy map
# rather than chosen by eye.
#
# The first straight ran from y = 1 to y = 7 on lanes centred at x = 0, and
# there is a wall across all five of those lanes at y = 7.9: 0.9 m past the
# finish.  MPPI rolls 56 steps at 0.05 s, so 2.8 s, which at its ceiling is
# about 1.3 m of lookahead -- the wall was inside its horizon for the whole
# last second of the race, every rollout that held the lane ended in it, and
# r5 was pushed 0.79 m sideways out of its lane, replanned from (3.39, 4.86)
# and stopped 0.8 m short while the other four finished.  It is the one lane
# that also had shelf_big_3 at (3.5, 9.5) beside it, so the straight was not
# the same race for all five.
#
# Scanning the map for the longest corridor with no occupied and no unknown
# cell, across the full 6.4 m width the five lanes and a robot radius need:
# lanes centred on x = 0 give 8.6 m (y -0.8 to 7.8), lanes centred on x = -3
# give 9.4 m (y -1.6 to 7.8).  Starting at y = -1 and finishing at y = 5 uses
# the second, and leaves 2.8 m of run-off past the finish -- twice MPPI's
# horizon -- with 0.6 m behind the grid.
LANES = {'r1': -5.4, 'r2': -3.7, 'r3': -2.0, 'r4': -0.3, 'r5': 1.4}
START_Y = 0.0
YAW = math.pi / 2          # up the course, +y in the warehouse
RATE = 10.0

# The chicane.
#
# A curved path is not something a planner can be asked for: on clear floor A*
# returns the straight line, because the straight line is the cheapest. The
# curve has to come from geometry the costmap can see, so each lane gets three
# wall segments that block its centre and one side, alternately, and the robot
# has to go round them.
#
# (y of the wall, offset of its centre from the lane centre). Alternating signs
# make the free corridor swap sides at each gate, so the path is a slalom
# rather than one long bend, and every lane's three walls are identical
# translations of each other -- the five paths are congruent, so a lane cannot
# be an easier lane.
#
# The sizes are what fits. A 1.7 m lane has to hold the robot (0.34 m across)
# plus the costmap's 0.30 m inflation on each side, which is 0.94 m of corridor
# that has to stay clear, so the blocking wall can be at most 0.76 m wide.
# 0.60 m leaves 1.10 m of corridor, a 0.30 m deviation from the lane centre,
# and 0.60 m from a robot's passing line to its neighbour's nearest wall end --
# twice the inflation radius, so no robot pays for its neighbour's chicane.
#
# Measured against the warehouse's own occupancy map, the tightest point of the
# swept corridor (every lane, the full +/-0.45 m of swing, the whole course) is
# 0.93 m from the nearest occupied or unknown cell. Starting the grid at
# y = -0.4 instead of 0.0 put the outermost lane 0.54 m from the standing
# person at (1.0, -1.0), which is inside the inflation.
GATES = [(0.9, -0.45), (2.6, +0.45), (4.3, -0.45)]

# Three more walls, where a sixth lane's wall would have been.
#
# With one wall per lane per gate, the middle three robots drive through a
# 1.10 m gap between their own wall and their neighbour's, and the outer two
# find one gate each with nothing on the far side -- r5 at the gates that push
# right, r1 at the gate that pushes left. Measured: twelve of the fifteen
# corridors are 1.10 m wide with a 0.50 m band of zero cost down the middle,
# and three are open. An open gate is a wider gate, and the whole point of
# translating one chicane across five lanes is that the lane cannot be the
# variable.
#
# So the outermost lane gets the wall its missing neighbour would have put
# there: lane_max + spacing + offset when the corridor runs right, lane_min -
# spacing + offset when it runs left. That is -6.65 at y = 2.6 and +2.65 at
# y = 0.9 and 4.3, both on floor the occupancy map says is free -- the left one
# spans -6.95..-6.35 against a clear region starting at -7.06, and the right
# one 2.35..2.95 against one ending at +3.02.
GATE_W = 0.60              # along x, the blocking span
GATE_T = 0.15              # along y, thin enough not to be a corridor
GATE_H = 0.60              # tall enough for a lidar 0.25 m off the floor
RACE_LENGTH = 6.0          # start line to finish line, along the course


def wall_poses():
    """Every chicane wall, as (x of its centre, y, the gate's offset).

    One per lane per gate, plus the three edge walls the outermost lanes need
    so that all five corridors are the same 1.10 m gap -- see the note beside
    GATES.
    """
    lanes = sorted(LANES.values())
    spacing = lanes[1] - lanes[0]
    out = []
    for gy, off in GATES:
        for lx in lanes:
            out.append((lx + off, gy, off))
        edge = (lanes[-1] + spacing) if off < 0 else (lanes[0] - spacing)
        out.append((edge + off, gy, off))
    return out


class GridTf(Node):

    def __init__(self) -> None:
        super().__init__('grid_tf')
        self.pub = self.create_publisher(TFMessage, '/tf', 10)
        self.msg = TFMessage()
        for ns, x in LANES.items():
            t = TransformStamped()
            t.header.frame_id = 'map'
            t.child_frame_id = f'{ns}/odom'
            t.transform.translation.x = x
            t.transform.translation.y = START_Y
            t.transform.rotation.z = math.sin(YAW / 2)
            t.transform.rotation.w = math.cos(YAW / 2)
            self.msg.transforms.append(t)
        # The chicane, drawn where it was placed.  The walls are Gazebo models
        # and this race serves no map, so the only other way they reach RViz is
        # as lidar returns off whichever robot is near one -- which means the
        # opening frame of a clip shows five robots and no course. Latched, so
        # RViz gets it whenever it starts.
        wall_qos = QoSProfile(depth=1)
        wall_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.wall_pub = self.create_publisher(MarkerArray, '/chicane', wall_qos)
        self.wall_pub.publish(self._walls())

        self.create_timer(1.0 / RATE, self._tick)
        self.get_logger().info(
            f'pinning {len(LANES)} odom origins into map at {RATE:.0f} Hz, '
            f'{len(wall_poses())} chicane walls drawn')

    def _walls(self) -> MarkerArray:
        ma = MarkerArray()
        i = 0
        for lx, gy, off in wall_poses():
            if True:
                m = Marker()
                m.header.frame_id = 'map'
                m.ns = 'chicane'
                m.id = i
                i += 1
                m.type = Marker.CUBE
                m.action = Marker.ADD
                m.pose.position.x = lx
                m.pose.position.y = gy
                m.pose.position.z = GATE_H / 2
                m.pose.orientation.w = 1.0
                m.scale.x, m.scale.y, m.scale.z = GATE_W, GATE_T, GATE_H
                m.color.r, m.color.g, m.color.b, m.color.a = 0.85, 0.45, 0.1, 0.9
                ma.markers.append(m)
        return ma

    def _tick(self) -> None:
        now = self.get_clock().now().to_msg()
        for t in self.msg.transforms:
            t.header.stamp = now
        self.pub.publish(self.msg)


def main() -> None:
    if '--lanes' in sys.argv[1:]:
        # Shell assignments, for `eval "$(grid_tf.py --lanes)"`.
        lanes = ' '.join(f'{ns}:{x}' for ns, x in LANES.items())
        print(f'LANES="{lanes}" START_Y={START_Y} YAW={YAW!r} '
              f'RACE_LENGTH={RACE_LENGTH}')
        return

    if '--gates' in sys.argv[1:]:
        # One line per wall: a name and an SDF model, for race_up.sh to spawn.
        # Boxes rather than cylinders: a flat face is what makes the corridor
        # beside it a corridor, and the lidar returns off a curve at a shallow
        # angle are the sparsest returns there are.
        for k, (wx, gy, off) in enumerate(wall_poses()):
            if True:
                name = f'gate_{k}'
                sdf = (
                    f'<?xml version="1.0"?><sdf version="1.7">'
                    f'<model name="{name}"><static>true</static>'
                    f'<pose>{wx:.3f} {gy:.3f} {GATE_H / 2:.3f} 0 0 0</pose>'
                    f'<link name="link">'
                    f'<collision name="c"><geometry><box><size>'
                    f'{GATE_W} {GATE_T} {GATE_H}</size></box></geometry></collision>'
                    f'<visual name="v"><geometry><box><size>'
                    f'{GATE_W} {GATE_T} {GATE_H}</size></box></geometry>'
                    f'<material><ambient>0.85 0.45 0.1 1</ambient>'
                    f'<diffuse>0.85 0.45 0.1 1</diffuse></material></visual>'
                    f'</link></model></sdf>')
                print(f'{name}\t{sdf}')
        return
    rclpy.init()
    n = GridTf()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
