#!/bin/bash
# Put the robot back on the warehouse origin between runs, facing $1 (radians,
# default -pi/2, which is south, roughly along the route to the goal).
#
# The heading matters more than it looks.  DWA's dynamic window is the measured
# yaw rate plus or minus max_dyawrate * dt = 0.1 rad/s, so it can only turn as
# fast as it has already begun turning.  Dropped in facing east with the goal
# due south it never opened that window: every tick logged omega = -0.10, the
# robot curved away at a 3 m radius and drove 5 m in the wrong direction.  A
# robot that starts pointed along its route is also just the normal case.
#
# No transform bookkeeping here: localize.py derives map->odom from ground
# truth every tick, so teleporting the model is enough and the frame follows.
# The dock is moved clear once at bringup -- it spawns 0.157 m in front of the
# robot, pointing at it, and the robot drove into it and sat there with its
# wheels turning while odometry invented ten metres of travel.
. /opt/rosenv.sh
YAW=${1:--1.5707963}
read Z W < <(python3 -c "import math;y=$YAW;print(math.sin(y/2), math.cos(y/2))")
timeout 30 gz service -s /world/warehouse/set_pose \
  --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 5000 \
  --req "name: \"turtlebot4\", position: {x: 0, y: 0, z: 0.01}, orientation: {x: 0, y: 0, z: $Z, w: $W}" \
  2>&1 | tail -1
sleep 6
