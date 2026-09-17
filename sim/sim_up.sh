#!/bin/bash
# Bring up the TurtleBot4 warehouse simulation and everything the repo's
# planners and controllers need to run against it.  Run once; drive.sh then
# runs against it as many times as you like.
#
# sim/README.md explains why each of these steps is here.  In short: the stock
# top-level launch renders a GUI nobody is looking at, so the server and the
# spawn are started separately -- which leaves out the clock bridge, without
# which every use_sim_time node sits frozen at t=0.  The global costmap wants a
# map frame with a static layer in it.  The robot spawns 0.157 m in front of
# its dock, pointing at it.  And the Create 3 reflex layer latches a false
# CLIFF in this world and overrides every command with a backward escape.

. /opt/rosenv.sh
G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. $G/lib.sh
export DISPLAY=:99
export GZ_SIM_RESOURCE_PATH=/opt/ros/jazzy/share/turtlebot4_gz_bringup/worlds:/opt/ros/jazzy/share
RUN=$G/run; SIM=$RUN/sim; mkdir -p "$SIM" "$RUN/log"
W=/opt/ros/jazzy/share/turtlebot4_gz_bringup/worlds/warehouse.sdf
MAP=/opt/ros/jazzy/share/turtlebot4_navigation/maps/warehouse.yaml

start "$SIM/xvfb.pid" Xvfb :99 -screen 0 1280x720x24 -nolisten tcp
sleep 3

start "$SIM/gz.pid" gz sim -s -r -v 1 "$W"
echo "gazebo started; loading the warehouse"
sleep 30

start "$SIM/spawn.pid" ros2 launch turtlebot4_gz_bringup turtlebot4_spawn.launch.py \
      rviz:=false use_sim_time:=true
sleep 10

start "$SIM/clock.pid" ros2 run ros_gz_bridge parameter_bridge \
      /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock
sleep 5

start "$SIM/map.pid" ros2 run nav2_map_server map_server \
      --ros-args -p yaml_filename:=$MAP -p use_sim_time:=true
sleep 8
timeout 40 ros2 lifecycle set /map_server configure > /dev/null 2>&1
timeout 40 ros2 lifecycle set /map_server activate  > /dev/null 2>&1
echo "map served"

echo "waiting for the robot to finish spawning"
sim_wait 12

MC=$(pgrep -f "irobot_create_nodes/motion_control" | head -1)
[ -n "$MC" ] && { echo "stopping motion_control ($MC)"; kill -TERM "$MC"; sleep 5; }

start "$SIM/relay.pid" python3 $G/cmd_relay.py --ros-args -p use_sim_time:=true
start "$SIM/localize.pid" python3 $G/localize.py --ros-args -p use_sim_time:=true
sleep 5

# The dock spawns 0.157 m in front of the robot, facing it.  Left there, every
# run begins with the robot driving into it and stalling on the ramp.
timeout 30 gz service -s /world/warehouse/set_pose \
  --reqtype gz.msgs.Pose --reptype gz.msgs.Boolean --timeout 5000 \
  --req 'name: "standard_dock", position: {x: 0, y: 11, z: 0}, orientation: {x: 0, y: 0, z: 0, w: 1}' \
  > /dev/null 2>&1
echo "dock moved clear of the spawn point"

bash $G/reset.sh > /dev/null 2>&1

echo "topics: $(timeout 25 ros2 topic list 2>/dev/null | wc -l)"
have=$(timeout 25 ros2 topic list 2>/dev/null | grep -cE "^/(scan|odom|clock|map)$")
echo "core topics present: $have/4"
echo "--- scan against the served map ---"
timeout 220 python3 $G/check_align.py --ros-args -p use_sim_time:=true 2>&1 | tail -2
echo SIMUP
