"""Print the simulator's clock, in whole seconds, or nothing.

    clock_now.py [seconds_to_wait]

`ros2 topic echo /clock --once` is what this replaces, and it is not reliable
on this machine: the CLI gives discovery about a second, and under load with
forty-odd participants that is not enough -- `ros2 node list` has returned 0
nodes in a graph where a 30-second rclpy subscription found /scan, /odom and
/clock all delivering. Every playback factor in the harness is measured from
two readings of this clock, and a missed reading is a recorded run with no gif
at the end of it.

Prints nothing and exits 1 if no message arrives, so the shell can tell the
difference between a clock at zero and a clock it could not read.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import sys
import time

import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock


def main() -> int:
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
    rclpy.init()
    n = Node('clock_now')
    got: list[int] = []
    n.create_subscription(Clock, '/clock', lambda m: got.append(m.clock.sec), 10)
    end = time.time() + secs
    while time.time() < end and not got:
        rclpy.spin_once(n, timeout_sec=0.2)
    if got:
        print(got[-1])
    rclpy.shutdown()
    return 0 if got else 1


if __name__ == '__main__':
    raise SystemExit(main())
