#!/usr/bin/env python3
"""
Pure Pursuit Local Controller — reactive_autonomous_nav
  • Simple geometric path following
  • Chases a "lookahead point" on the global path
  • Smooth, stable, and computationally very lightweight
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

class PurePursuitControllerNode(Node):

    def __init__(self):
        super().__init__('pure_pursuit_controller_node')

        # ── Params ───────────────────────────────────────────────────
        self.lookahead_dist     = 0.4    # m
        self.max_vel            = 0.35   # m/s
        self.min_vel            = 0.05
        self.max_yawrate        = 1.5    # rad/s
        self.goal_tol           = 0.15
        self.dt                 = 0.1    # 10 Hz control loop

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
        self.lookahead_pub = self.create_publisher(Marker, '/pure_pursuit_lookahead', 10)

        # ── Timer ────────────────────────────────────────────────────
        self.create_timer(self.dt, self._control_loop)

        self.get_logger().info('Pure Pursuit Controller — ready')

    def _path_cb(self, msg: Path):
        self.current_path = msg
        self.goal_reached = False
        self.get_logger().info(f'Received new path: {len(msg.poses)} waypoints')

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
        if self.current_path is None or self.goal_reached:
            return

        pose = self._get_robot_pose()
        if pose is None:
            return
        rx, ry, ryaw = pose

        # Check goal distance
        goal = self.current_path.poses[-1].pose.position
        dist_to_goal = math.hypot(goal.x - rx, goal.y - ry)
        if dist_to_goal < self.goal_tol:
            self.get_logger().info('Goal reached!')
            self.goal_reached = True
            self._stop()
            self.status_pub.publish(String(data='REACHED'))
            return

        # Find lookahead point
        target = self._get_lookahead_point(rx, ry)
        if target is None:
            return

        # Transform target to robot frame
        dx = target.x - rx
        dy = target.y - ry
        
        # Local coordinates
        cos_yaw = math.cos(ryaw)
        sin_yaw = math.sin(ryaw)
        lx =  cos_yaw * dx + sin_yaw * dy
        ly = -sin_yaw * dx + cos_yaw * dy

        # Pure Pursuit curvature calculation: κ = 2y / L^2
        dist_sq = lx**2 + ly**2
        if dist_sq < 0.001:
            curvature = 0.0
        else:
            curvature = (2.0 * ly) / dist_sq

        # Commands
        cmd = Twist()
        # Slow down near goal
        v = self.max_vel if dist_to_goal > 0.5 else max(self.min_vel, self.max_vel * (dist_to_goal / 0.5))
        
        cmd.linear.x = v
        cmd.angular.z = v * curvature
        
        # Clamp angular
        if abs(cmd.angular.z) > self.max_yawrate:
            cmd.angular.z = math.copysign(self.max_yawrate, cmd.angular.z)
            # If we're turning too sharp, slow down linear
            cmd.linear.x *= 0.8

        self.cmd_pub.publish(cmd)
        
        # Visualize lookahead point
        self._visualize_lookahead(target)

    def _get_lookahead_point(self, rx, ry):
        """Find the first point on the path that is >= lookahead_dist away."""
        if not self.current_path.poses:
            return None
            
        best_pt = None
        for ps in self.current_path.poses:
            p = ps.pose.position
            d = math.hypot(p.x - rx, p.y - ry)
            if d >= self.lookahead_dist:
                return p
        
        # If no point is far enough, use the last one (the goal)
        return self.current_path.poses[-1].pose.position
    
    def _visualize_lookahead(self, target):
        """Publish lookahead point as a sphere marker."""
        from visualization_msgs.msg import Marker
        from std_msgs.msg import ColorRGBA
        
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'pure_pursuit'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = target.x
        marker.pose.position.y = target.y
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        marker.color = ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.8)  # magenta
        
        self.lookahead_pub.publish(marker)

    def _stop(self):
        self.cmd_pub.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitControllerNode()
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
