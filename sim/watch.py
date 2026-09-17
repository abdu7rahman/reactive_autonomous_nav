"""Report where the robot is in the map frame, and whether it is making progress.

Odometry is reported in the odom frame, which sits at whatever offset the last
reset measured, so raw /odom numbers say nothing about whether the robot is
near its goal.  This asks tf for map->base_link instead.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import math
import sys
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
import tf2_ros


class W(Node):
    def __init__(self) -> None:
        super().__init__("watch")
        self.plan: Path | None = None
        self.buf = tf2_ros.Buffer()
        self.lis = tf2_ros.TransformListener(self.buf, self)
        self.create_subscription(Path, "/plan", self.on_plan, 10)

    def on_plan(self, m: Path) -> None:
        self.plan = m

    def where(self) -> tuple[float, float, float] | None:
        try:
            t = self.buf.lookup_transform("map", "base_link", rclpy.time.Time())
        except Exception:
            return None
        p, q = t.transform.translation, t.transform.rotation
        return p.x, p.y, math.atan2(2 * (q.w * q.z + q.x * q.y),
                                    1 - 2 * (q.y * q.y + q.z * q.z))


def main() -> int:
    gx, gy = float(sys.argv[1]), float(sys.argv[2])
    secs = float(sys.argv[3]) if len(sys.argv) > 3 else 60.0
    rclpy.init()
    n = W()
    t_end = time.time() + secs
    best = 1e9
    while time.time() < t_end:
        rclpy.spin_once(n, timeout_sec=0.2)
        w = n.where()
        if w is None:
            continue
        d = math.hypot(w[0] - gx, w[1] - gy)
        best = min(best, d)
        # /plan is not latched, so a watcher that attaches after the planner
        # has published sees nothing until the next replan.  Say "plan -" for
        # "none seen yet" rather than "0", which reads as an empty plan.
        npts = f"{len(n.plan.poses)}" if n.plan else "-"
        print(f"map ({w[0]:6.2f},{w[1]:6.2f}) yaw {w[2]:6.2f}  "
              f"goal {d:5.2f} m  best {best:5.2f}  plan {npts}", flush=True)
        time.sleep(4)
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
