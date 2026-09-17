"""Block until the transform chain the costmaps need is actually available.

reset.sh teleports the model, and for a moment afterwards the diff drive stops
publishing odom->base_link while the wheels take the load again.  The local
costmap's activation has a bounded wait for that transform; miss it and the
lifecycle manager never gets past local_costmap, so global_costmap never
activates and the planner sits on "Cannot plan: global_data is None" for the
whole run.  One clip was lost to it before this gate existed.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import sys
import time

import rclpy
from rclpy.node import Node
import tf2_ros

PAIRS = [("odom", "base_link"), ("map", "odom"), ("map", "base_link")]


def main() -> int:
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
    need = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    rclpy.init()
    n = Node("wait_tf")
    buf = tf2_ros.Buffer()
    lis = tf2_ros.TransformListener(buf, n)
    end = time.time() + secs
    steady = None
    while time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.2)
        ok = True
        for a, b in PAIRS:
            try:
                buf.lookup_transform(a, b, rclpy.time.Time())
            except Exception:
                ok = False
                break
        if not ok:
            steady = None          # any gap restarts the clock
            continue
        if steady is None:
            steady = time.time()
        elif time.time() - steady >= need:
            print(f"tf chain steady for {need:.0f}s")
            rclpy.shutdown()
            return 0
    print(f"tf chain NOT steady within {secs:.0f}s")
    rclpy.shutdown()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
