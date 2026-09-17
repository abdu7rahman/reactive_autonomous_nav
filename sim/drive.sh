#!/bin/bash
# Record one planner/controller pair against the running simulator.
#
#   drive.sh <planner> <controller> <tag> [wall_seconds]
#
# Every run starts from the same state: clean.sh kills anything left over from
# the last one, reset.sh puts the robot back on the warehouse origin facing
# south.  Both matter.  A stale publisher from an earlier run once sat on
# /cmd_vel_unstamped at 10 Hz and the robot followed it instead of the
# controller under test, which looks exactly like a controller that cannot
# steer.
#
# Timing comes from a measured run: astar with dwa took 139 wall seconds from
# its first plan to "Goal REACHED", the warehouse running at a real-time factor
# near 0.1 under software rendering.  190 seconds covers that with room for a
# slower controller to finish inside the clip.
#
# The capture is cropped to RViz's 3D viewport.  The Displays dock cannot be
# hidden from a config file -- RViz restores dock visibility from its
# QMainWindow State blob and an empty blob means "defaults", not "hidden" -- so
# it is cropped out rather than fought with.

. /opt/rosenv.sh
G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. $G/lib.sh
export DISPLAY=:99

PLANNER=$1; CONTROLLER=$2; TAG=$3; WALL=${4:-190}
GX=${GX:-2.0}; GY=${GY:--5.0}
RUN=$G/run; JOB=$RUN/job; mkdir -p "$JOB" "$RUN/log"
trap 'stop $JOB/*.pid' EXIT

echo "=== $TAG : planner=$PLANNER controller=$CONTROLLER"
bash $G/clean.sh
bash $G/reset.sh > /dev/null 2>&1


# RViz first, and given time to finish loading.  Started alongside the nav
# stack it competes for the same four cores while it compiles shaders and loads
# the robot meshes through software GL, and the global costmap -- which has a
# bounded wait for base_link->map at activation -- loses that race and comes up
# with no data at all.  The symptom downstream is the planner logging
# "Cannot plan: global_data is None" for the whole run.
start "$JOB/rviz.pid" rviz2 -d $G/nav.rviz --ros-args -p use_sim_time:=true
sleep 35

# Gate on tf here, with RViz already up, rather than before it.  The costmaps
# have a bounded wait for base_link->map at activation and nav2's lifecycle
# manager aborts the whole bringup permanently when that wait expires -- it
# does not retry.  Checked before RViz loads, the chain looks fine and then is
# not; this measures the moment that actually matters.
if ! timeout 140 python3 $G/wait_tf.py 120 5; then
  echo "    tf chain never settled -- aborting $TAG"
  exit 1
fi

start "$JOB/nav.pid" ros2 launch reactive_autonomous_nav nav_launch.py \
      planner:="$PLANNER" controller:="$CONTROLLER" use_sim_time:=true
sleep 20

# Gate on the costmap actually delivering, not on a guessed sleep
if ! timeout 200 python3 $G/wait_topic.py /global_costmap/costmap OccupancyGrid 180; then
  echo "    global costmap never published -- aborting $TAG"
  exit 1
fi

# /goal_pose is not latched and `pub --once` exits the moment it has written,
# so a goal sent before the planner has finished building its subscription is
# dropped with no error anywhere.  The first recorded run was 190 seconds of a
# robot that had never been told where to go: RViz starting alongside the nav
# stack slowed the planner's startup past the fixed sleep that used to be here.
subs=0
for _ in $(seq 1 40); do
  subs=$(timeout 20 ros2 topic info /goal_pose 2>/dev/null | awk '/Subscription count/{print $3}')
  [ "${subs:-0}" -ge 1 ] && break
  sleep 3
done
n=$(timeout 25 ros2 topic info /cmd_vel_unstamped 2>/dev/null | awk '/Publisher count/{print $3}')
echo "    /goal_pose subscribers: ${subs:-0}, /cmd_vel_unstamped publishers: ${n:-?}"

# Capture opens before the goal so the clip starts on a robot at rest and the
# plan appears inside the recording.  1.2 frames a second of wall clock is
# about 12 frames a second of simulated time, so the GIF plays at the speed the
# robot is actually moving.
C0=$(sim_now)
start "$JOB/ff.pid" ffmpeg -loglevel error -y -f x11grab -draw_mouse 0 \
      -video_size 912x624 -framerate 6 -i :99+366,65 -t "$WALL" "$RUN/$TAG.mp4"
sleep 4

ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'map'}, pose: {position: {x: $GX, y: $GY}, orientation: {w: 1.0}}}" \
  > /dev/null 2>&1
echo "    goal ($GX, $GY) sent; capturing ${WALL}s"

while [ -f "$JOB/ff.pid" ] && kill -0 "$(cat $JOB/ff.pid)" 2>/dev/null; do sleep 5; done
rm -f "$JOB/ff.pid"
C1=$(sim_now)

# Play the clip at the speed the robot is actually moving.  The warehouse runs
# far slower than real time under software rendering, so the factor is measured
# from the simulator's own clock across this capture rather than assumed -- it
# varies with what else is competing for the four cores, and a fixed guess
# would make one controller look quicker than another for no reason.
SPEED=$(python3 -c "
c0, c1, wall = ${C0:-0}, ${C1:-0}, $WALL
d = c1 - c0
print(f'{wall/d:.3f}' if d > 0 else '')")
if [ -z "$SPEED" ]; then
  echo "    could not read the clock across this capture; $TAG.mp4 kept, no gif"
  echo "    (a made-up playback factor would look exactly like a measured one)"
  exit 1
fi
echo "    sim ${C0}s to ${C1}s over ${WALL}s wall -> playing back ${SPEED}x"

# 48 colours is plenty for RViz's flat fills; 560 px keeps a ten-clip set to a
# size a repository can carry and still reads at a glance.
ffmpeg -loglevel error -y -i "$RUN/$TAG.mp4" -vf \
  "setpts=PTS/$SPEED,fps=10,scale=560:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=48[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3" \
  -loop 0 "$G/gif/$TAG.gif" < /dev/null

# start() writes each process's output beside its pid file
cp -f "$JOB/nav.log" "$RUN/log/$TAG.nav.log" 2>/dev/null
hits=$(grep -c "Goal REACHED" "$RUN/log/$TAG.nav.log" 2>/dev/null)
echo "    goal reached: ${hits:-0}"
ls -la "$G/gif/$TAG.gif" 2>/dev/null | awk '{printf "    gif %.1f MB\n", $5/1048576}'
