"""Send a goal and confirm the planner answered it.

    send_goal.py <x> <y> [seconds]

Counting subscribers was the previous way of making sure a goal landed, and it
does not work on this machine.  /goal_pose is not latched and a publisher that
exits the moment it has written loses the message if matching has not finished,
so drive.sh waited for a subscription to appear in the graph before publishing
-- and the graph lied in both directions.  `ros2 topic info` reported zero
subscribers on runs whose planner demonstrably received the goal and drove to
it; rclpy's own count_subscribers then reported zero for a full 120 s on five
consecutive runs that all worked, and reported one on the single run that
failed.  A recorded 600-second capture of a stationary robot came out of
trusting it: the goal went out, nothing arrived, and the clip was 57 kB of one
still frame.

So this does not count anything.  It publishes the goal and waits for the
planner's own /plan, which is the only evidence that matters, and publishes
again if none comes.  Re-sending is free: every planner in this package treats
a goal as the current goal and replans to it, so a duplicate is a replan to the
same place.  Exits non-zero with nothing on /plan, which is worth an aborted
run rather than a capture of nothing.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import sys
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

# How long to wait for a plan before sending the goal again.  The planners here
# take 0.1 to 3.5 s to search the warehouse at 0.03 m/cell -- rrt_smac_hybrid
# is the slow end -- and the costmap callback that has to land first runs at
# 1 Hz, so 12 s is several times the worst measured case rather than a guess at
# it.
WAIT = 12.0
TRIES = 6


class Sender(Node):

    def __init__(self, x: float, y: float) -> None:
        super().__init__('send_goal')
        self.x, self.y = x, y
        self.plan: Path | None = None
        self.pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.create_subscription(Path, '/plan', self._plan, 10)

    def _plan(self, msg: Path) -> None:
        if msg.poses and self.plan is None:
            self.plan = msg

    def send(self) -> None:
        m = PoseStamped()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.position.x = self.x
        m.pose.position.y = self.y
        m.pose.orientation.w = 1.0
        self.pub.publish(m)


def main() -> int:
    x, y = float(sys.argv[1]), float(sys.argv[2])
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else WAIT * TRIES

    rclpy.init()
    n = Sender(x, y)
    end = time.time() + budget
    tries = 0
    while time.time() < end and n.plan is None and tries < TRIES:
        n.send()
        tries += 1
        wait_until = min(end, time.time() + WAIT)
        while time.time() < wait_until and n.plan is None:
            rclpy.spin_once(n, timeout_sec=0.2)

    ok = n.plan is not None
    if ok:
        print(f'goal ({x}, {y}) planned: {len(n.plan.poses)} poses, '
              f'{tries} send{"" if tries == 1 else "s"}', flush=True)
    else:
        print(f'goal ({x}, {y}) sent {tries} times and nothing appeared on '
              f'/plan in {budget:.0f}s', flush=True)
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
