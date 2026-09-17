"""Pin each robot's odom origin into the shared `map` frame.

    grid_tf.py              publish them
    grid_tf.py --lanes      print them as shell assignments

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

This file is where the grid geometry is defined, once.  race_up.sh spawns the
robots at these poses and race_timer.py places each goal directly ahead of its
robot, and both read it from here rather than keeping a copy -- three
declarations of the same five numbers is three chances for a robot to be
spawned somewhere its odom pin says it is not.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import math
import sys

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

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
LANES = {'r1': -5.6, 'r2': -4.3, 'r3': -3.0, 'r4': -1.7, 'r5': -0.4}
START_Y = -1.0
YAW = math.pi / 2          # up the straight, +y in the warehouse
RATE = 10.0


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
        self.create_timer(1.0 / RATE, self._tick)
        self.get_logger().info(
            f'pinning {len(LANES)} odom origins into map at {RATE:.0f} Hz')

    def _tick(self) -> None:
        now = self.get_clock().now().to_msg()
        for t in self.msg.transforms:
            t.header.stamp = now
        self.pub.publish(self.msg)


def main() -> None:
    if '--lanes' in sys.argv[1:]:
        # Shell assignments, for `eval "$(grid_tf.py --lanes)"`.
        lanes = ' '.join(f'{ns}:{x}' for ns, x in LANES.items())
        print(f'LANES="{lanes}" START_Y={START_Y} YAW={YAW!r}')
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
