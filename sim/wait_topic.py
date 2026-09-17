"""Block until a topic delivers a message, or time out.

A sleep is a guess about how long a node takes to come up; on a world running
at a real-time factor near 0.1, with RViz loading meshes through software GL on
the same four cores, the guess is wrong in both directions.  Waiting for the
message itself is the thing that actually matters.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from nav_msgs.msg import OccupancyGrid, Path

KINDS = {"OccupancyGrid": OccupancyGrid, "Path": Path}


def main() -> int:
    topic, kind, secs = sys.argv[1], sys.argv[2], float(sys.argv[3])
    # Minimum populated cells for an OccupancyGrid.  The costmap publishes its
    # first message before the static layer has received /map, so a gate that
    # accepts any message accepts an empty one: a run once came up on an
    # 800x1200 costmap -- the size configured, not the map's 1006x1674 -- with
    # nothing in it at all, and drove a world with no obstacles in it.
    need = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    rclpy.init()
    n = Node("wait_topic")
    got: list[object] = []
    q = QoSProfile(depth=1)
    q.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
    q.reliability = QoSReliabilityPolicy.RELIABLE
    def keep(m: object) -> None:
        if kind == "OccupancyGrid" and need:
            if sum(1 for v in m.data if v > 0) < need:
                return
        got.append(m)

    n.create_subscription(KINDS[kind], topic, keep, q)
    end = time.time() + secs
    while time.time() < end and not got:
        rclpy.spin_once(n, timeout_sec=0.2)
    ok = bool(got)
    if ok and kind == "OccupancyGrid":
        g = got[0]
        known = sum(1 for v in g.data if v > 0)
        print(f"{topic}: {g.info.width}x{g.info.height}, {known} cells above zero")
    elif ok:
        print(f"{topic}: received")
    else:
        print(f"{topic}: nothing with at least {need} populated cells in {secs:.0f}s")
    rclpy.shutdown()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
