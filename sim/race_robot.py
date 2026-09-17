#!/usr/bin/env python3
"""Emit a race-ready TurtleBot 4 URDF for one lane.

    race_robot.py <namespace>   > r1.urdf

Same description as every other run in this harness -- the standard TurtleBot 4
xacro, same meshes, same masses, same wheel geometry -- with the Create 3
control stack taken out and Gazebo's own DiffDrive put in its place.

Why not the stock turtlebot4_spawn.launch.py, five times over.  That was tried
first and the log says exactly what happened: r1 came up, r2 loaded into the
physics engine but its controller_manager sat on "Waiting for data on
'/r2/robot_description'" for eleven minutes, and r3, r4 and r5 never reached
the engine at all while the server printed "SceneBroadcaster: Timed out waiting
for state".  Each stock robot is about forty-five nodes -- hazard vectors, IR
vectors, a UI manager, a kidnap estimator, a mock publisher, a 1000 Hz
ros2_control loop -- and twelve gpu_lidars.  Five of those is 225 nodes and 60
raycast sensors on four cores with software rendering, and FastDDS ran out of
shared-memory ports underneath it ("Failed init_port fastrtps_port7173").

None of that is what a race measures.  What comes out of here is the same robot
with the parts a controller comparison uses:

  - DiffDrive in place of ros2_control, carrying the same numbers the Create 3
    controller was configured with (irobot_create_control/config/control.yaml):
    0.233 m wheel separation, 0.03575 m radius, 0.46 m/s and 1.9 rad/s ceilings,
    0.9 m/s^2 and 7.725 rad/s^2 acceleration limits.  Identical dynamics, no
    controller_manager and no robot_description handshake to lose.
  - the rplidar, and none of the eleven cliff and IR gpu_lidars.  The planners
    and controllers in this package read one LaserScan; the other eleven exist
    for the Create 3 reflex layer, which is also what latched a false CLIFF in
    this world and had to be killed on every single-robot run.  Taking the
    sensors out removes the reflex problem at its root rather than killing the
    node that acts on it.
  - no model-level Sensors system.  Each robot carrying its own spins up its
    own render context; the world file gets one instead (see race_up.sh).

Frames are prefixed with the namespace, so five trees share one /tf without
colliding and RViz can draw the whole grid at once.
"""

from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import subprocess
import sys
import xml.etree.ElementTree as ET

XACRO = ('/root/ros2_ws/tb4_overlay/share/turtlebot4_description/'
         'urdf/standard/turtlebot4.urdf.xacro')

# The one sensor this package's nodes actually read.  Everything else in the
# description is a Create 3 reflex input.
KEEP_SENSOR = 'rplidar'

# The wheel-drop suspension, welded.  These are 30 mm prismatic joints that
# exist so the Create 3 can tell when a wheel has left the floor, and the
# hazard sensor that reads them is one of the eleven this file removes.  Left
# movable they break the transform tree instead: robot_state_publisher
# publishes a movable joint only when it has a joint state for it, Gazebo's
# joint state publisher is asked for the two drive joints, and the result is
# base_link with no transform to wheel_drop_left -- RViz drew five robots with
# a RobotModel error on each and no wheels, and tf2 reported "two or more
# unconnected trees" for base_link -> left_wheel.  Fixed, the pair lands on
# /tf_static at startup and the drive joints still turn.
WELD_JOINTS = ('wheel_drop_left_joint', 'wheel_drop_right_joint')

# Straight out of irobot_create_control/config/control.yaml, so the robot that
# races has the dynamics of the robot that was recorded ten times already.
WHEEL_SEPARATION = 0.233
WHEEL_RADIUS = 0.03575
MAX_LINEAR = 0.46
MAX_ANGULAR = 1.9
MAX_LINEAR_ACCEL = 0.9
MAX_ANGULAR_ACCEL = 7.725
PUBLISH_RATE = 62.0     # the stock diffdrive_controller publish_rate


def _gazebo_blocks(root):
    return [e for e in root.findall('gazebo')]


def strip(root):
    """Take out the Create 3 control stack, the reflex sensors and the
    per-model Sensors system."""
    for gz in _gazebo_blocks(root):
        for plug in list(gz.findall('plugin')):
            fn = plug.get('filename', '')
            if ('gz_ros2_control' in fn or 'sensors-system' in fn
                    or 'contact-system' in fn or 'pose-publisher-system' in fn):
                gz.remove(plug)
        for sensor in list(gz.findall('sensor')):
            if sensor.get('name') != KEEP_SENSOR:
                gz.remove(sensor)
        if len(gz) == 0:
            root.remove(gz)


def weld(root):
    """Make the wheel-drop suspension rigid -- see WELD_JOINTS."""
    for joint in root.findall('joint'):
        if joint.get('name') in WELD_JOINTS:
            joint.set('type', 'fixed')
            for tag in ('axis', 'limit', 'dynamics'):
                for e in joint.findall(tag):
                    joint.remove(e)


def add_systems(root, ns):
    """DiffDrive and a joint state publisher, in place of ros2_control."""
    gz = ET.SubElement(root, 'gazebo')

    dd = ET.SubElement(gz, 'plugin',
                       filename='gz-sim-diff-drive-system',
                       name='gz::sim::systems::DiffDrive')
    fields = [
        ('left_joint', 'left_wheel_joint'),
        ('right_joint', 'right_wheel_joint'),
        ('wheel_separation', f'{WHEEL_SEPARATION}'),
        ('wheel_radius', f'{WHEEL_RADIUS}'),
        ('max_linear_velocity', f'{MAX_LINEAR}'),
        ('min_linear_velocity', f'{-MAX_LINEAR}'),
        ('max_angular_velocity', f'{MAX_ANGULAR}'),
        ('min_angular_velocity', f'{-MAX_ANGULAR}'),
        ('max_linear_acceleration', f'{MAX_LINEAR_ACCEL}'),
        ('min_linear_acceleration', f'{-MAX_LINEAR_ACCEL}'),
        ('max_angular_acceleration', f'{MAX_ANGULAR_ACCEL}'),
        ('min_angular_acceleration', f'{-MAX_ANGULAR_ACCEL}'),
        ('odom_publish_frequency', f'{PUBLISH_RATE}'),
        ('topic', f'/{ns}/cmd_vel'),
        ('odom_topic', f'/{ns}/odom'),
        ('tf_topic', f'/{ns}/pose_tf'),
        ('frame_id', f'{ns}/odom'),
        ('child_frame_id', f'{ns}/base_link'),
    ]
    for tag, text in fields:
        ET.SubElement(dd, tag).text = text

    jsp = ET.SubElement(gz, 'plugin',
                        filename='gz-sim-joint-state-publisher-system',
                        name='gz::sim::systems::JointStatePublisher')
    ET.SubElement(jsp, 'topic').text = f'/{ns}/joint_states'
    for j in ('left_wheel_joint', 'right_wheel_joint'):
        ET.SubElement(jsp, 'joint_name').text = j


def name_lidar_frame(root, ns):
    """Report the scan in the frame robot_state_publisher publishes.

    Gazebo stamps a sensor message with <model>/<link>/<sensor>, which is a
    frame nothing else in the graph knows about; the stock bringup closes that
    with an identity static transform.  gz_frame_id says it directly instead,
    so there is one fewer node per robot and one fewer place for the chain to
    break.  race_up.sh publishes the identity transform as well, because a
    transform to a frame nobody stamps costs nothing and a missing one drops
    every scan.
    """
    for gz in _gazebo_blocks(root):
        for sensor in gz.findall('sensor'):
            if sensor.get('name') == KEEP_SENSOR:
                ET.SubElement(sensor, 'gz_frame_id').text = f'{ns}/rplidar_link'


def main():
    if len(sys.argv) != 2:
        sys.exit(f'usage: {sys.argv[0]} <namespace>')
    ns = sys.argv[1]

    urdf = subprocess.run(['xacro', XACRO, 'gazebo:=ignition', 'namespace:=/'],
                          check=True, capture_output=True, text=True).stdout
    root = ET.fromstring(urdf)

    strip(root)
    weld(root)
    add_systems(root, ns)
    name_lidar_frame(root, ns)

    sys.stdout.write(ET.tostring(root, encoding='unicode'))


if __name__ == '__main__':
    main()
