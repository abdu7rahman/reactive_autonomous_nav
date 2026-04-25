#!/usr/bin/env python3
"""
TEB Local Controller — reactive_autonomous_nav
  • Simplified Timed Elastic Band approach
  • Poses along the path are treated as nodes in an elastic band
  • External forces push nodes away from obstacles
  • Internal forces keep the band smooth and equidistant
  • Real-time trajectory deformation for dynamic obstacle avoidance
"""
print(''.join(chr(x-7) for x in [104,105,107,124,115,39,121,104,111,116,104,117]))

import rclpy
import rclpy.time
import math
import numpy as np
from rclpy.node     import Node
from rclpy.duration import Duration
from rclpy.qos      import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg      import Twist, Point, PoseStamped
from nav_msgs.msg           import Odometry, Path, OccupancyGrid
from std_msgs.msg           import String, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros                import TransformListener, Buffer

class TEBControllerNode(Node):

    def __init__(self):
        super().__init__('teb_controller_node')

        self.max_vel        = 0.4
        self.max_yawrate    = 1.5
        self.lookahead_wps  = 8
        self.dt             = 0.1
        self.goal_tol       = 0.15

        # Forces — consistent naming
        self.force_obstacle = 0.5   # was force_obs (typo caused runtime crash)
        self.force_smooth   = 0.3
        self.force_dist     = 0.2
        self.min_obs_dist   = 0.4
        self.desired_sep    = 0.15

        self.current_pose   = None
        self.costmap_data   = None
        self.costmap_info   = None
        self.costmap_origin = None
        self.current_path   = None
        self.goal_reached   = False
        self.band           = []

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        cmap_qos = QoSProfile(
            depth       = 1,
            reliability = ReliabilityPolicy.RELIABLE,
            durability  = DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.cmd_pub    = self.create_publisher(Twist,       '/cmd_vel_unstamped', 10)
        self.band_pub   = self.create_publisher(MarkerArray, '/teb_band',          10)
        self.status_pub = self.create_publisher(String,      '/teb_status',        10)

        self.create_subscription(Odometry,      '/odom',                  self._odom_cb,    10)
        self.create_subscription(Path,          '/plan',                  self._path_cb,    10)
        self.create_subscription(OccupancyGrid, '/local_costmap/costmap', self._costmap_cb, cmap_qos)

        self.create_timer(0.1, self._control_loop)
        self.get_logger().info('TEB Controller (Simplified) — ready')

    def _odom_cb(self, msg):
        self.current_pose = msg.pose.pose

    def _path_cb(self, msg: Path):
        self.current_path = msg.poses
        self.goal_reached = False
        self.band = []
        for ps in self.current_path[:self.lookahead_wps]:
            self.band.append([ps.pose.position.x, ps.pose.position.y])

    def _costmap_cb(self, msg: OccupancyGrid):
        self.costmap_data   = np.array(msg.data, dtype=np.int16).reshape(
            (msg.info.height, msg.info.width))
        self.costmap_info   = msg.info
        self.costmap_origin = (msg.info.origin.position.x,
                               msg.info.origin.position.y)

    def _get_tf(self, target, source):
        try:
            t = self.tf_buffer.lookup_transform(
                target, source, rclpy.time.Time(), Duration(seconds=0.2))
            tr = t.transform.translation
            rot = t.transform.rotation
            yaw = np.arctan2(2.0 * (rot.w * rot.z + rot.x * rot.y),
                             1.0 - 2.0 * (rot.y**2 + rot.z**2))
            return tr.x, tr.y, yaw
        except Exception:
            return None

    def _control_loop(self):
        if self.current_pose is None or not self.band:
            return

        pose = self._get_tf('map', 'base_link')
        if pose is None:
            return
        rx, ry, ryaw = pose

        self.band[0] = [rx, ry]

        if self.costmap_data is not None:
            self._deform_band()

        target = self.band[min(2, len(self.band)-1)]
        dx = target[0] - rx
        dy = target[1] - ry
        dist = math.hypot(dx, dy)

        goal = self.current_path[-1].pose.position
        dist_to_goal = math.hypot(goal.x - rx, goal.y - ry)

        if dist_to_goal < self.goal_tol:
            if not self.goal_reached:
                self.get_logger().info('TEB: Goal reached')
                self.status_pub.publish(String(data='REACHED'))
                self.goal_reached = True
            self.cmd_pub.publish(Twist())
            return

        angle_to_target = math.atan2(dy, dx)
        alpha = (angle_to_target - ryaw + math.pi) % (2*math.pi) - math.pi

        cmd = Twist()
        cmd.linear.x = min(self.max_vel, dist * 2.0)
        cmd.angular.z = np.clip(alpha * 3.0, -self.max_yawrate, self.max_yawrate)

        self.cmd_pub.publish(cmd)
        self._publish_band()

    def _deform_band(self):
        res = self.costmap_info.resolution
        for i in range(1, len(self.band) - 1):
            curr = self.band[i]
            prev = self.band[i-1]
            nxt  = self.band[i+1]

            force = np.zeros(2)

            col = int((curr[0] - self.costmap_origin[0]) / res)
            row = int((curr[1] - self.costmap_origin[1]) / res)
            if 0 <= col < self.costmap_info.width and 0 <= row < self.costmap_info.height:
                cost_center = self.costmap_data[row, col]

                if cost_center > 50:
                    grad_x, grad_y = 0.0, 0.0

                    if col > 0:
                        grad_x += (cost_center - self.costmap_data[row, col-1])
                    if col < self.costmap_info.width - 1:
                        grad_x += (self.costmap_data[row, col+1] - cost_center)
                    if row > 0:
                        grad_y += (cost_center - self.costmap_data[row-1, col])
                    if row < self.costmap_info.height - 1:
                        grad_y += (self.costmap_data[row+1, col] - cost_center)

                    grad_norm = math.sqrt(grad_x**2 + grad_y**2) + 1e-6
                    repulsion = np.array([-grad_x / grad_norm, -grad_y / grad_norm])
                    force += self.force_obstacle * repulsion * (cost_center / 100.0)  # fixed

            mid = 0.5 * (np.array(prev) + np.array(nxt))
            force += self.force_smooth * (mid - np.array(curr))

            d_prev = np.array(curr) - np.array(prev)
            dist_prev = np.linalg.norm(d_prev)
            if dist_prev > 1e-3:
                dir_prev = d_prev / dist_prev
                separation_force = self.force_dist * (self.desired_sep - dist_prev) * dir_prev
                force += separation_force

            self.band[i] = (np.array(curr) + force).tolist()

    def _publish_band(self):
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'band'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.05
        m.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=0.8)
        for p in self.band:
            m.points.append(Point(x=p[0], y=p[1], z=0.1))
        ma.markers.append(m)
        self.band_pub.publish(ma)

def main(args=None):
    rclpy.init(args=args)
    node = TEBControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
