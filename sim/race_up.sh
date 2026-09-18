#!/bin/bash
# Bring up the warehouse with five TurtleBot 4s on a start line.
#
#   RACE_COURSE=straight|chicane RACE_FIELD=ours|versus|nav2|bench-py|bench-cpp \
#     race_up.sh
#
# The course -- where the lanes are, how far apart, and what is standing on
# them -- is reactive_autonomous_nav/race_course.py.  Nothing here knows: the
# lanes come out of grid_tf.py --lanes and the walls out of grid_tf.py --gates,
# and both are read rather than repeated, because a lane value that disagreed
# with its own odom pin would put a robot somewhere the whole stack believes it
# is not.
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
# Lane spacing is the course's, and it is not a free choice: the robot is
# 0.34 m across and carries a 0.30 m inflation, so 1.3 m is about the tightest
# start line that does not put every robot inside its neighbour's forbidden
# zone, and a course with a wall blocking the middle of each lane needs 1.7 to
# leave a corridor beside it.

. /opt/rosenv.sh
G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. $G/lib.sh
export DISPLAY=:99
# Exported, not defaulted per process: every python3 below resolves the course
# from the environment, and five processes each falling back to their own
# default is five chances to disagree.
export RACE_COURSE=${RACE_COURSE:-chicane}
# The field too, because the grid is now the field's size rather than the
# course's: the two bench fields enter four implementations, and a grid built
# without knowing that spawns a fifth robot with no controller to drive it and
# stands a set of gates in an empty lane. race.sh reads the same variable, so
# a grid brought up for one field and raced as another would put every label
# on the wrong robot -- which is the failure grid_tf.py's own header records.
export RACE_FIELD=${RACE_FIELD:-ours}

# The overlay first: it carries the same description with the OAK-D stripped
# out, and five depth cameras rendering through llvmpipe is most of the cost of
# a frame and none of what these planners read.
export GZ_SIM_RESOURCE_PATH=/root/ros2_ws/tb4_overlay/share:/opt/ros/jazzy/share
RUN=$G/run; SIM=$RUN/sim; mkdir -p "$SIM" "$RUN/log"

# Tear the old world down first, always. This is a bringup, and a bringup run
# against a world that is already up silently reuses it: `start` skips a
# process whose pid file is live, the spawns land as new entities in the old
# world or fail on the name that is already there, and the gate at the foot of
# this file reports "robots live: 4/4" because the *previous* run's robots are
# still answering on those topics.
#
# Which is exactly what it did. A grid brought up for the bench-cpp field
# after the bench-py race had every robot parked where bench-py left it --
# map->r1/base_link at y = 5.994 and r2 at 6.025, which are bench-py's own
# finishing distances of 5.99 m and 6.03 m -- and race.sh refused the race one
# step later with "only 2/4 robots are on the start line". The bringup said
# RACEUP; nothing between the two knew the world was four minutes old.
bash $G/sim_down.sh 2>&1 | sed 's/^/  /'

eval "$(python3 $G/grid_tf.py --lanes)"
echo "course $COURSE: lanes $LANES, start y=$START_Y, $N_WALLS walls"
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

# Is there a simulator at all?  This is not ceremony.  `gz sim` segfaulted on
# start once -- five seconds after a teardown, so probably before the previous
# server had released its shared memory -- and everything after it carried on
# regardless: the wall loop reported "18/18 walls spawned" against a world that
# did not exist, because `gz service` against a dead server exits zero, and the
# five robot spawns then burned their 90-second timeouts one after another. A
# clock reading is the one thing that cannot be true without a running world.
if [ -z "$(sim_now)" ]; then
  echo "no /clock -- gazebo did not come up; aborting"
  tail -3 "$SIM/gz.log" 2>/dev/null | sed 's/^/    /'
  exit 1
fi
echo "simulator clock at $(sim_now)s"

# The dock is not spawned here at all -- it belongs to the stock bringup -- but
# the warehouse's own shelving is, and the start line sits on open floor.
# The chicane first, so the walls are in the world before the lidars are.
# Spawned as static SDF models rather than edited into the world file: the
# world is turtlebot4_gz_bringup's and this leaves it alone, and
# race_course.py stays the only place the course is described.
gates=0
while IFS=$'\t' read -r name sdf; do
  [ -z "$name" ] && continue
  if timeout 60 gz service -s /world/warehouse/create \
      --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 20000 \
      --req "sdf: '$sdf', name: '$name', allow_renaming: false" > /dev/null 2>&1; then
    gates=$((gates + 1))
  fi
done < <(python3 $G/grid_tf.py --gates)
# What the world has, not what the service calls returned.  `gz service`
# exits zero whether or not a server answered, so $gates counts attempts.
if [ "$N_WALLS" -gt 0 ]; then
  inworld=$(timeout 30 gz model --list 2>/dev/null | grep -c -- "- *gate_" || true)
  echo "course walls spawned: $gates/$N_WALLS (in the world: ${inworld:-0})"
  gates=${inworld:-0}
else
  echo "course has no walls"
fi

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
N=$(echo $LANES | wc -w)
echo "robots live: $ok/$N"
# $N_WALLS and $N, not literals: the straight has no walls and the chicane
# eighteen, and the first chicane run passed a gate written for fifteen with
# three walls missing. The robot count went the same way the moment a field
# entered four -- the grid came up correctly, all four live, and the literal
# 5 failed it.
if [ "$ok" = "$N" ] && [ "$gates" = "$N_WALLS" ]; then
  echo RACEUP
else
  echo "RACEUP FAILED -- robots $ok/$N, course walls $gates/$N_WALLS"
  exit 1
fi
