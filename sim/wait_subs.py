"""Block until a topic has enough endpoints, and print how many it has.

    wait_subs.py <topic> <seconds> [minimum] [--publishers]

/goal_pose is not latched and `ros2 topic pub --once` exits the moment it has
written, so a goal sent before the planner has finished building its
subscription is dropped with no error anywhere: the first recorded run in this
harness was 190 seconds of a robot that had never been told where to go.

The count comes from rclpy's graph API rather than `ros2 topic info`. The CLI
gives discovery about a second and reported 0 subscribers on a run whose
planner demonstrably received the goal and drove to it, which made the wait in
drive.sh burn its whole 120-second budget on every single clip.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import sys
import time

import rclpy
from rclpy.node import Node


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    topic = args[0]
    secs = float(args[1])
    want = int(args[2]) if len(args) > 2 else 1
    pubs = '--publishers' in sys.argv[1:]

    rclpy.init()
    n = Node('wait_subs')
    count = n.count_publishers if pubs else n.count_subscribers
    end = time.time() + secs
    seen = 0
    while time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.2)
        seen = count(topic)
        if seen >= want:
            break
    print(seen)
    rclpy.shutdown()
    return 0 if seen >= want else 1


if __name__ == '__main__':
    raise SystemExit(main())
