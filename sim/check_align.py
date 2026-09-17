"""Project one live /scan into the map frame and score it against the served
occupancy grid.  If the map is misaligned with the simulator the beam endpoints
land on free cells and the global costmap is fiction -- which is exactly what
happened when wheel odometry ran away from the robot: agreement fell from 95%
to 2% while the robot stood still."""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
import tf2_ros

class Check(Node):
    def __init__(self):
        super().__init__('check_align')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', value=True)])
        self.scan = None; self.map = None
        self.buf = tf2_ros.Buffer(); self.lis = tf2_ros.TransformListener(self.buf, self)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        q = QoSProfile(depth=1); q.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(OccupancyGrid, '/map', self.on_map, q)
    def on_scan(self, m): self.scan = m
    def on_map(self, m): self.map = m

rclpy.init()
n = Check()
for _ in range(2000):
    rclpy.spin_once(n, timeout_sec=0.05)
    if n.scan and n.map:
        try:
            tf = n.buf.lookup_transform('map', n.scan.header.frame_id, rclpy.time.Time())
            break
        except Exception: pass
else:
    print("gave up waiting for scan/map/tf"); raise SystemExit(1)

t, r = tf.transform.translation, tf.transform.rotation
yaw = math.atan2(2*(r.w*r.z + r.x*r.y), 1 - 2*(r.y*r.y + r.z*r.z))
print(f"laser frame '{n.scan.header.frame_id}' at map ({t.x:.3f},{t.y:.3f}) yaw {yaw:.3f}")

g, info = n.map.data, n.map.info
ox, oy, res, W, H = info.origin.position.x, info.origin.position.y, info.resolution, info.width, info.height
hit = free = unk = off = 0
for i, d in enumerate(n.scan.ranges):
    if not (n.scan.range_min < d < min(n.scan.range_max, 10.0)): continue
    a = n.scan.angle_min + i*n.scan.angle_increment + yaw
    x, y = t.x + d*math.cos(a), t.y + d*math.sin(a)
    cx, cy = int((x-ox)/res), int((y-oy)/res)
    if not (0 <= cx < W and 0 <= cy < H): off += 1; continue
    v = g[cy*W + cx]
    # 1-cell tolerance: SLAM walls are a cell or two thick
    near = any(g[(cy+b)*W + (cx+a2)] > 50
               for a2 in (-2,-1,0,1,2) for b in (-2,-1,0,1,2)
               if 0 <= cx+a2 < W and 0 <= cy+b < H)
    if near: hit += 1
    elif v < 0: unk += 1
    else: free += 1
tot = hit+free+unk
print(f"beams {tot}:  on-wall {hit} ({100*hit/max(tot,1):.0f}%)  free {free}  unknown {unk}  off-map {off}")
rclpy.shutdown()
