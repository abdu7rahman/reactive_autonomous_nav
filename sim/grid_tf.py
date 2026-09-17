"""Publish the course: the odom pins, the chicane and the name labels.

    grid_tf.py                publish them
    grid_tf.py --lanes        print the course as shell assignments
    grid_tf.py --gates        print one gz EntityFactory request per wall

Where the course is defined is not here -- it is
reactive_autonomous_nav/race_course.py, which the launch file reads too.  This
is what puts it into the graph: race_up.sh spawns the walls from --gates and
the robots from --lanes, and the node pins each robot's odom origin into the
shared `map` frame the way a localiser would.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import math
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

from reactive_autonomous_nav.race_course import (
    COLOUR, CONTROLLER, COURSE, GATE_H, GATE_T, GATE_W, LANES, RACE_LENGTH,
    START_Y, YAW, wall_poses)

RATE = 10.0


def _sdf(name, x, y):
    """A static box.  Flat faces: the corridor beside a wall is a corridor
    because the wall has a face, and lidar returns off a curve at a shallow
    angle are the sparsest returns there are."""
    return (f'<?xml version="1.0"?><sdf version="1.7">'
            f'<model name="{name}"><static>true</static>'
            f'<pose>{x:.3f} {y:.3f} {GATE_H / 2:.3f} 0 0 0</pose>'
            f'<link name="link">'
            f'<collision name="c"><geometry><box><size>'
            f'{GATE_W} {GATE_T} {GATE_H}</size></box></geometry></collision>'
            f'<visual name="v"><geometry><box><size>'
            f'{GATE_W} {GATE_T} {GATE_H}</size></box></geometry>'
            f'<material><ambient>0.85 0.45 0.1 1</ambient>'
            f'<diffuse>0.85 0.45 0.1 1</diffuse></material></visual>'
            f'</link></model></sdf>')


class GridTf(Node):

    def __init__(self) -> None:
        super().__init__('grid_tf')
        self.pub = self.create_publisher(TFMessage, '/tf', 10)
        self.msg = TFMessage()
        for ns, x in LANES.items():
            t = TransformStamped()
            t.header.frame_id = 'map'
            t.child_frame_id = f'{ns}/odom'
            t.transform.translation.x = x
            t.transform.translation.y = START_Y
            t.transform.rotation.z = math.sin(YAW / 2)
            t.transform.rotation.w = math.cos(YAW / 2)
            self.msg.transforms.append(t)

        latched = QoSProfile(depth=1)
        latched.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        # The chicane, drawn where it was placed. The walls are Gazebo models
        # and this race serves no map, so the only other way they reach RViz is
        # as lidar returns off whichever robot is near one -- which means the
        # opening frame of a clip shows five robots and no course.
        self.wall_pub = self.create_publisher(MarkerArray, '/chicane', latched)
        self.wall_pub.publish(self._walls())

        # And who is who. Five coloured trails with no legend is five coloured
        # trails: the clip is 560 px wide and carries no caption, so the name
        # has to be in the scene. Each label lives in its own robot's base
        # frame, so it follows the robot without anyone publishing its pose.
        self.name_pub = self.create_publisher(MarkerArray, '/race_labels', latched)
        self.name_pub.publish(self._labels())

        self.create_timer(1.0 / RATE, self._tick)
        self.get_logger().info(
            f'course {COURSE}: {len(LANES)} odom pins at {RATE:.0f} Hz, '
            f'{len(wall_poses())} chicane walls, {len(LANES)} labels')

    def _walls(self) -> MarkerArray:
        ma = MarkerArray()
        for i, (wx, gy) in enumerate(wall_poses()):
            m = Marker()
            m.header.frame_id = 'map'
            m.ns, m.id = 'chicane', i
            m.type, m.action = Marker.CUBE, Marker.ADD
            m.pose.position.x = wx
            m.pose.position.y = gy
            m.pose.position.z = GATE_H / 2
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = GATE_W, GATE_T, GATE_H
            m.color.r, m.color.g, m.color.b, m.color.a = 0.85, 0.45, 0.1, 0.9
            ma.markers.append(m)
        return ma

    def _labels(self) -> MarkerArray:
        ma = MarkerArray()
        for i, ns in enumerate(LANES):
            m = Marker()
            m.header.frame_id = f'{ns}/base_link'
            m.ns, m.id = 'labels', i
            m.type, m.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            m.pose.position.z = 0.62      # clear of the robot's own tower
            m.pose.orientation.w = 1.0
            m.scale.z = 0.30              # cap height, legible at 560 px
            r, g, b = COLOUR[ns]
            m.color.r, m.color.g, m.color.b = r / 255.0, g / 255.0, b / 255.0
            m.color.a = 1.0
            m.text = CONTROLLER[ns]
            ma.markers.append(m)
        return ma

    def _tick(self) -> None:
        now = self.get_clock().now().to_msg()
        for t in self.msg.transforms:
            t.header.stamp = now
        self.pub.publish(self.msg)


def main() -> None:
    if '--lanes' in sys.argv[1:]:
        lanes = ' '.join(f'{ns}:{x}' for ns, x in LANES.items())
        print(f'COURSE={COURSE} LANES="{lanes}" START_Y={START_Y} YAW={YAW!r} '
              f'RACE_LENGTH={RACE_LENGTH} N_WALLS={len(wall_poses())}')
        return

    if '--gates' in sys.argv[1:]:
        for k, (wx, gy) in enumerate(wall_poses()):
            print(f'gate_{k}\t{_sdf(f"gate_{k}", wx, gy)}')
        return

    rclpy.init()
    n = GridTf()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
