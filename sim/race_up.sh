#!/bin/bash
# Bring up the warehouse with five TurtleBot 4s on a start line.
#
# Five stock turtlebot4_spawn stacks does not fit on this machine -- see the
# header of race_robot.py for what the logs said when it was tried.  This
# brings up the same robot with the parts a controller comparison uses: the
# same description, the same wheel geometry and the same velocity limits,
# Gazebo's own DiffDrive instead of ros2_control, and one lidar instead of
# twelve.  Four processes per robot rather than forty-five.
#
# Frames.  Every robot's tree is prefixed with its namespace -- r1/base_link,
# r1/rplidar_link -- so all five share one /tf without colliding, and one
# static transform per robot pins its odom origin into a common `map` at the
# pose it was spawned at.  That is exactly what a localiser would publish, and
# it lets RViz draw the whole grid in one view.  No map server: the costmaps
# roll with the robot and the only obstacles on the straight are the other four
# robots, which the lidar sees.
#
# Lanes 1.3 m apart.  The robot is 0.34 m across and carries a 0.30 m
# inflation, so anything tighter starts the race with every robot already
# inside its neighbour's forbidden zone.

. /opt/rosenv.sh
G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. $G/lib.sh
export DISPLAY=:99
# The overlay first: it carries the same description with the OAK-D stripped
# out, and five depth cameras rendering through llvmpipe is most of the cost of
# a frame and none of what these planners read.
export GZ_SIM_RESOURCE_PATH=/root/ros2_ws/tb4_overlay/share:/opt/ros/jazzy/share
RUN=$G/run; SIM=$RUN/sim; mkdir -p "$SIM" "$RUN/log"

# The grid geometry lives in grid_tf.py, which also publishes the transforms
# that pin each odom origin into `map`.  Read rather than repeated: a lane
# value that disagreed with its own odom pin would put a robot somewhere the
# whole stack believes it is not.
eval "$(python3 $G/grid_tf.py --lanes)"
SPAWN_Z=0.01

# One Sensors system for the whole world.  The stock world has it commented out
# because every turtlebot4 URDF carries its own, and five models each starting
# a render context is five OpenGL contexts through llvmpipe.  race_robot.py
# takes the per-model copy out; this puts one back at world level.  ogre, not
# ogre2 -- the TurtleBot 4 description asks for ogre and ogre2 is what the
# world's commented-out block named, which is a default nobody ran here.
W=$SIM/race_warehouse.sdf
python3 - "$W" <<'PY'
import sys
src = '/opt/ros/jazzy/share/turtlebot4_gz_bringup/worlds/warehouse.sdf'
old = """<!--<plugin name="gz::sim::systems::Sensors" filename="gz-sim-sensors-system">
        <render_engine>ogre2</render_engine>
    </plugin>-->"""
new = """<plugin name="gz::sim::systems::Sensors" filename="gz-sim-sensors-system">
        <render_engine>ogre</render_engine>
    </plugin>"""
text = open(src).read()
assert old in text, 'warehouse.sdf no longer carries the commented-out Sensors block'
open(sys.argv[1], 'w').write(text.replace(old, new))
PY
echo "world written with one Sensors system"

start "$SIM/xvfb.pid" Xvfb :99 -screen 0 1280x720x24 -nolisten tcp
sleep 3
start "$SIM/gz.pid" gz sim -s -r -v 1 "$W"
echo "gazebo starting"; sleep 30
start "$SIM/clock.pid" ros2 run ros_gz_bridge parameter_bridge \
      /clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock
sleep 5

# The dock is not spawned here at all -- it belongs to the stock bringup -- but
# the warehouse's own shelving is, and the start line sits on open floor.
for lane in $LANES; do
  ns=${lane%%:*}; x=${lane##*:}
  python3 $G/race_robot.py "$ns" > "$SIM/$ns.urdf"
  echo "spawning $ns at x=$x"
  timeout 90 ros2 run ros_gz_sim create -world warehouse \
      -file "$SIM/$ns.urdf" -name "$ns" \
      -x "$x" -y "$START_Y" -z "$SPAWN_Z" -Y "$YAW" > "$SIM/spawn_$ns.log" 2>&1
  echo "  $(tail -1 "$SIM/spawn_$ns.log")"
  sleep 4
done

# All five map -> <ns>/odom pins from one node on /tf, which is where a
# localiser publishes map -> odom.  Five static_transform_publisher processes
# on /tf_static was the first arrangement and grid_tf.py's header records what
# it measured: with fifteen transient-local publishers on that topic a fresh
# listener got some latched samples and not others, and the nav2 costmaps
# failed to activate on one to four robots per attempt.  There is no lidar
# identity transform either -- race_robot.py sets gz_frame_id, and the scan was
# measured arriving stamped r3/rplidar_link.
start "$SIM/gridtf.pid" python3 $G/grid_tf.py --ros-args -p use_sim_time:=true
sleep 3

for lane in $LANES; do
  ns=${lane%%:*}

  # robot_state_publisher gives the tree below base_link, prefixed.  It reads
  # the same URDF the model was spawned from, so what RViz draws and what the
  # physics engine steps cannot drift apart.
  # The URDF goes in as a file rather than as a -p robot_description:= value:
  # a parameter on the command line is parsed as YAML, and 38 kB of XML full of
  # double quotes is not YAML.
  start "$SIM/rsp_$ns.pid" ros2 run robot_state_publisher robot_state_publisher \
        "$SIM/$ns.urdf" --ros-args -p use_sim_time:=true \
        -p frame_prefix:="$ns/" -r __ns:="/$ns"

  # One bridge process per robot rather than one per topic.  The stock bringup
  # runs twenty-one of them per robot and they were 4-5% of a core each.
  start "$SIM/bridge_$ns.pid" ros2 run ros_gz_bridge parameter_bridge \
        "/$ns/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist" \
        "/$ns/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry" \
        "/$ns/pose_tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V" \
        "/$ns/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model" \
        "/world/warehouse/model/$ns/link/rplidar_link/sensor/rplidar/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan" \
        --ros-args -p use_sim_time:=true \
        -r "/$ns/cmd_vel:=/$ns/cmd_vel_unstamped" \
        -r "/$ns/pose_tf:=/tf" \
        -r "/world/warehouse/model/$ns/link/rplidar_link/sensor/rplidar/scan:=/$ns/scan"
done
sleep 10
sim_wait 6

echo "topics: $(timeout 25 ros2 topic list 2>/dev/null | wc -l)"
ok=0
for lane in $LANES; do
  ns=${lane%%:*}
  # >/dev/null, not 2>/dev/null: wait_topic.py reports on stdout, so without
  # this the substitution captures its sentence as well as the digit and the
  # gate fails with all five robots publishing -- which is what it did.
  n=$(timeout 30 python3 $G/wait_topic.py "/$ns/scan" LaserScan 40 0 >/dev/null 2>&1 && echo 1 || echo 0)
  m=$(timeout 30 python3 $G/wait_topic.py "/$ns/odom" Odometry 40 0 >/dev/null 2>&1 && echo 1 || echo 0)
  printf "  %s scan=%s odom=%s\n" "$ns" "$n" "$m"
  [ "$n$m" = "11" ] && ok=$((ok + 1))
done
echo "robots live: $ok/5"
[ "$ok" = "5" ] && echo RACEUP || echo "RACEUP FAILED"
