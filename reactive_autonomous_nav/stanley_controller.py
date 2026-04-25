#!/usr/bin/env python3
"""
Stanley Local Controller — reactive_autonomous_nav
  • Precision path following using cross-track error + heading error
  • Compensates for lateral deviation at the front of the robot
  • Smooth, stable, and highly accurate for smooth paths (like SMAC)
"""
print(''.join(chr(x-7) for x in [104,105,107,124,115,39,121,104,111,116,104,117]))

import rclpy
import math
import numpy as np
from rclpy.node     import Node
from rclpy.duration import Duration
from rclpy.qos      import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg      import Twist, PoseStamped
from nav_msgs.msg           import Path
from std_msgs.msg           import String
from tf2_ros                import TransformListener, Buffer

class StanleyControllerNode(Node):

    def __init__(self):
        super().__init__('stanley_controller_node')

        # ── Params ───────────────────────────────────────────────────
        self.k                  = 1.5    # Cross-track gain
        self.max_vel            = 0.35
        self.min_vel            = 0.05
        self.max_yawrate        = 1.5
        self.goal_tol           = 0.15
        self.dt                 = 0.1

        # ── State ────────────────────────────────────────────────────
        self.current_path   = None
        self.goal_reached   = False

        # ── TF ───────────────────────────────────────────────────────
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── Subs/Pubs ────────────────────────────────────────────────
        self.create_subscription(Path, '/plan', self._path_cb, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel_unstamped', 10)
        self.status_pub = self.create_publisher(String, '/dwa_status', 10)
        
        # Visualization
        from visualization_msgs.msg import Marker
        self.closest_pub = self.create_publisher(Marker, '/stanley_closest_point', 10)

        # ── Timer ────────────────────────────────────────────────────
        self.create_timer(self.dt, self._control_loop)

        self.get_logger().info('Stanley Controller — ready')

    def _path_cb(self, msg: Path):
        self.current_path = msg
        self.goal_reached = False

    def _get_robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(), Duration(seconds=0.5))
            tr = t.transform.translation
            rot = t.transform.rotation
            yaw = math.atan2(2.0 * (rot.w * rot.z + rot.x * rot.y),
                             1.0 - 2.0 * (rot.y**2 + rot.z**2))
            return tr.x, tr.y, yaw
        except Exception:
            return None

    def _control_loop(self):
        if self.current_path is None or self.goal_reached or len(self.current_path.poses) < 2:
            return

        pose = self._get_robot_pose()
        if pose is None:
            return
        rx, ry, ryaw = pose

        # Goal check
        goal = self.current_path.poses[-1].pose.position
        dist_to_goal = math.hypot(goal.x - rx, goal.y - ry)
        if dist_to_goal < self.goal_tol:
            self.goal_reached = True
            self._stop()
            self.status_pub.publish(String(data='REACHED'))
            return

        # Stanley Control Logic
        # 1. Find the closest segment on the path
        idx, error, segment_yaw = self._get_closest_point(rx, ry)
        
        # 2. Heading error
        heading_error = segment_yaw - ryaw
        # Normalize to [-pi, pi]
        heading_error = (heading_error + math.pi) % (2.0 * math.pi) - math.pi

        # 3. Cross-track error
        # Slow down near goal
        v = self.max_vel if dist_to_goal > 0.5 else max(self.min_vel, self.max_vel * (dist_to_goal / 0.5))
        
        # Avoid division by zero
        v_safe = max(0.1, abs(v))
        crosstrack_steering = math.atan2(self.k * error, v_safe)

        # Total steering angle
        steer = heading_error + crosstrack_steering

        # Convert steering angle to angular velocity for differential drive
        # Using kinematic approximation: ω ≈ v/L * tan(δ) where δ is steering angle
        # For small angles: ω ≈ v * δ / L, using L = wheelbase ≈ 0.3m
        wheelbase = 0.3
        angular_vel = v * math.tan(steer) / wheelbase
        
        # Commands
        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = angular_vel
        
        # Clamping
        if abs(cmd.angular.z) > self.max_yawrate:
            cmd.angular.z = math.copysign(self.max_yawrate, cmd.angular.z)
            # Reduce linear velocity when turning sharply
            cmd.linear.x *= 0.6

        self.cmd_pub.publish(cmd)
        
        # Visualize closest point
        closest_pt = self.current_path.poses[idx].pose.position
        self._visualize_closest(closest_pt, error)

    def _get_closest_point(self, rx, ry):
        """Find the segment of the path closest to the robot."""
        best_d = float('inf')
        best_idx = 0
        
        # Simple version: find closest waypoint first
        for i, ps in enumerate(self.current_path.poses):
            p = ps.pose.position
            d = math.hypot(p.x - rx, p.y - ry)
            if d < best_d:
                best_d = d
                best_idx = i
        
        # Now find the orientation of the path at that point
        # Use next waypoint for heading
        if best_idx < len(self.current_path.poses) - 1:
            p1 = self.current_path.poses[best_idx].pose.position
            p2 = self.current_path.poses[best_idx+1].pose.position
            yaw = math.atan2(p2.y - p1.y, p2.x - p1.x)
        else:
            # Use orientation from current pose to goal
            p1 = self.current_path.poses[best_idx-1].pose.position
            p2 = self.current_path.poses[best_idx].pose.position
            yaw = math.atan2(p2.y - p1.y, p2.x - p1.x)

        # Cross-track error: sign depends on which side of the path the robot is on
        # d = (x-x1)(y2-y1) - (y-y1)(x2-x1) / dist(p1,p2)
        if best_idx < len(self.current_path.poses) - 1:
            p1 = self.current_path.poses[best_idx].pose.position
            p2 = self.current_path.poses[best_idx+1].pose.position
        else:
            p1 = self.current_path.poses[best_idx-1].pose.position
            p2 = self.current_path.poses[best_idx].pose.position

        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dist = math.hypot(dx, dy)
        if dist > 0.001:
            cross_err = ((rx - p1.x) * dy - (ry - p1.y) * dx) / dist
        else:
            cross_err = 0.0

        return best_idx, cross_err, yaw
    
    def _visualize_closest(self, closest_pt, error):
        """Visualize the closest path point being tracked."""
        from visualization_msgs.msg import Marker
        from std_msgs.msg import ColorRGBA
        
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'stanley'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = closest_pt.x
        marker.pose.position.y = closest_pt.y
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.12
        marker.scale.y = 0.12
        marker.scale.z = 0.12
        # Color based on error: green (small) to red (large)
        error_norm = min(abs(error) / 0.5, 1.0)
        marker.color = ColorRGBA(r=error_norm, g=1.0-error_norm, b=0.0, a=0.8)
        
        self.closest_pub.publish(marker)

    def _stop(self):
        self.cmd_pub.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = StanleyControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
