#!/bin/bash
# Record the five-robot race against the running simulator.
#
#   RACE_COURSE=straight|chicane RACE_FIELD=ours|versus|nav2 race.sh [wall_seconds]
#
# The course has to be the one race_up.sh spawned -- it decides where the lanes
# are -- so both read RACE_COURSE from the environment.  RACE_FIELD picks who
# is racing: this package's five controllers, or its DWA against nav2's own.
# The output is named after both, so gif/race.gif is the straight,
# gif/race-chicane.gif the chicane and gif/race-chicane-nav2.gif the nav2
# field on it.  Three artefacts, not one overwritten three times.
#
# race_up.sh puts the grid on the start line; this brings up five costmap
# pairs and five controllers -- plus five planners on the straight, where each
# robot plans its own way to its own goal -- opens RViz on the whole grid, and
# releases all five at once.
#
# The capture window is wall seconds, not simulated ones.  Everything that sets
# it -- how long the slowest controller takes, what the real-time factor is
# with five physics bodies and five raycast sensors on four cores -- is
# measured by the run itself and reported at the end, so this is deliberately
# generous and the clip is trimmed to the race afterwards.

. /opt/rosenv.sh
G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. $G/lib.sh
export DISPLAY=:99
export RACE_COURSE=${RACE_COURSE:-chicane}
export RACE_FIELD=${RACE_FIELD:-ours}


WALL=${1:-600}
eval "$(python3 $G/grid_tf.py --lanes)"
TAG=race; [ "$COURSE" != "straight" ] && TAG="race-$COURSE"
[ "$RACE_FIELD" != "ours" ] && TAG="$TAG-$RACE_FIELD"
RUN=$G/run; JOB=$RUN/job; mkdir -p "$JOB" "$RUN/log" "$G/gif"
trap 'stop $JOB/*.pid' EXIT

echo "=== $TAG"
bash $G/clean.sh

# The reference path before anything else.  On a course with gates it is what
# all five controllers are handed, and a reference that clipped a wall would
# not read as a bad path in the clip -- it would read as five robots stopping
# for no visible reason, because each one's own costmap refuses to follow a
# path through its inflated band.  Twenty minutes of simulator to find out
# what this measures in a tenth of a second.
if ! python3 $G/race_path.py --check > "$JOB/path.log" 2>&1; then
  sed 's/^/    /' "$JOB/path.log"
  echo "    the reference path is not clear of the course -- aborting"
  exit 1
fi
sed -n '1,4p' "$JOB/path.log" | sed 's/^/    /'

# RViz config generated from the same course module, so the framing suits the
# lane spacing and the colours cannot disagree with the labels.
python3 $G/race_rviz.py > "$RUN/$TAG.rviz" || exit 1

# Gate on the transform chain with the robots settled.  nav2's costmaps wait a
# bounded time for base_frame->global_frame when they activate and the
# lifecycle manager turns an expired wait into a permanent abort -- it logs
# "Failed to bring up all requested nodes" and never retries -- so the planner
# would spend the whole race on "Cannot plan: global_data is None".
for lane in $LANES; do
  ns=${lane%%:*}
  if ! timeout 90 ros2 run tf2_ros tf2_echo map "$ns/base_link" \
       --ros-args -p use_sim_time:=true 2>/dev/null | grep -q Translation; then
    echo "    no map->$ns/base_link -- aborting"
    exit 1
  fi
done
echo "    all five transform chains up"

# And on the grid being a grid.  Odometry is where it was when the robot
# spawned, and nothing in this stack can put it back: a teleport moves the body
# and leaves the odom frame six metres up the straight, so map->base_link then
# says the robot is somewhere it is not.  A race replayed without a fresh
# race_up.sh therefore starts with every robot already past the finish line,
# which is what the first attempt at timing this recorded -- 5.8 m of progress
# at the moment the grid was released, and three robots "crossing" 0.2 s later.
onstart=$(timeout 150 python3 $G/race_timer.py 1 --no-start \
          --ros-args -p use_sim_time:=true 2>/dev/null \
          | grep -cE "reached 0\.[0-9]+ m")
if [ "${onstart:-0}" -lt 5 ]; then
  echo "    only ${onstart:-0}/5 robots are on the start line -- run race_up.sh first"
  exit 1
fi
echo "    all five on the start line"

# Bring the nav stack up, then walk the ten costmaps through their lifecycle
# with lifecycle_up.py rather than leaving it to five nav2 lifecycle managers.
# Its header says why: the managers abort the whole bringup permanently when a
# single service call fails, which on this machine happened on most attempts,
# and three consecutive full bringups were thrown away by relaunching around
# it.  Relaunching is still here as a second line, but it should now be rare
# rather than routine.
#
# The status, not a grep for a count.  This used to test the output for
# "costmaps active: 10/10", which lifecycle_up.py has never printed -- it
# prints "active: 10/10" -- so the condition could not be true however well the
# bringup went, and the next chicane race would have relaunched the whole stack
# three times and aborted with ten live costmaps.  It exits non-zero unless
# every node on its list reached active, which is the same question without a
# number in it.  The output goes to a file rather than through `tee`, because a
# pipeline's status is its last command's and tee always succeeds.
up=0
for attempt in 1 2 3; do
  start "$JOB/race.pid" ros2 launch reactive_autonomous_nav race_launch.py
  sleep 45
  if timeout 900 python3 $G/lifecycle_up.py 30 > "$JOB/costmaps.log" 2>&1; then
    up=1
  fi
  echo "    costmaps $(grep -m1 '^active:' "$JOB/costmaps.log" 2>/dev/null \
        | sed 's/^active: //') (attempt $attempt)"
  [ "$up" = "1" ] && break
  grep FAILED "$JOB/costmaps.log" 2>/dev/null | sed 's/^/      /'
  stop "$JOB/race.pid"
  # Sweep before relaunching, not just stop.  `ros2 launch` does not always
  # take its children with it, and a surviving controller_server answers the
  # next attempt's lifecycle calls from whatever state it was already in --
  # which is how an attempt that had brought two nav2 lanes all the way to
  # ACTIVE reported both as failures and then relaunched on top of them.
  bash $G/clean.sh | sed 's/^/      /'
  sleep 10
done
if [ "$up" != "1" ]; then
  echo "    the costmaps never all activated -- aborting"
  exit 1
fi

# One trail node for all five lanes, before RViz has anything to draw.  The
# controllers' own /driven_path is not what the clip shows any more: the nav2
# plugins publish nothing of the kind, and five controllers each drawing their
# own idea of where they had been is five measurements on one picture.
start "$JOB/trail.pid" python3 $G/trail.py --ros-args -p use_sim_time:=true
sleep 3

# RViz last.  It is the heaviest thing on this machine and the costmaps'
# activation wait is what it used to walk on.
start "$JOB/rviz.pid" rviz2 -d "$RUN/$TAG.rviz" --ros-args -p use_sim_time:=true
sleep 45

# Capture opens before the start so the clip begins on a grid at rest.
#
# mpegts, not mp4.  The capture is stopped the moment the race is decided
# rather than run to the full window, and an mp4 whose writer was signalled
# instead of reaching its own -t has no moov atom: the first race recorded this
# way produced "moov atom not found" and no gif at all, with the run itself
# perfectly good.  A transport stream has no index to finish writing.
PREROLL=6
C0=$(sim_now)
T0=$(date +%s.%N)
start "$JOB/ff.pid" ffmpeg -loglevel error -y -f x11grab -draw_mouse 0 \
      -video_size 912x624 -framerate 6 -i :99+366,65 -t "$WALL" \
      -f mpegts "$RUN/$TAG.ts"
sleep $PREROLL

# race_timer.py releases the grid and times it; it exits when the last robot
# crosses or its simulated budget runs out, whichever comes first.  240
# simulated seconds: the single-robot runs took 119 to 213 seconds of capture
# at a real-time factor near 0.11, which is 13 to 23 simulated seconds for a
# 3.8 m diagonal through shelving, so a 6 m course has an order of magnitude
# of room here.
start "$JOB/timer.pid" python3 $G/race_timer.py 240 --ros-args -p use_sim_time:=true
echo "    grid released; capturing ${WALL}s"

while [ -f "$JOB/ff.pid" ] && kill -0 "$(cat $JOB/ff.pid)" 2>/dev/null; do
  sleep 10
  if [ -f "$JOB/timer.pid" ] && ! kill -0 "$(cat $JOB/timer.pid)" 2>/dev/null; then
    grep -q "finish order" "$JOB/timer.log" 2>/dev/null && break
  fi
done
C1=$(sim_now)
T1=$(date +%s.%N)
stop "$JOB/ff.pid"

cp -f "$JOB/timer.log" "$RUN/log/$TAG.timer.log" 2>/dev/null
cp -f "$JOB/race.log"  "$RUN/log/$TAG.nav.log"   2>/dev/null
echo "--- result ---"
sed -n '/finish order/,$p' "$RUN/log/$TAG.timer.log" 2>/dev/null

# A race nobody started is not a clip.  The first chicane race published five
# reference paths to five subscribers that turned out to be RViz's Path
# displays, and encoded 3.3 MB of a still grid: every robot at 0.00 m of 5.80.
# race_timer.py confirms movement against each robot's own odometry now and
# says how many moved, and a race where any of them did not, or where any of
# them needed its path sent twice, is worth re-running rather than keeping.
moved=$(grep -oE "moving after the release: [0-9]+" "$RUN/log/$TAG.timer.log" \
        2>/dev/null | grep -oE "[0-9]+$")
if [ "${moved:-0}" != "5" ]; then
  echo "    only ${moved:-0}/5 robots moved -- $TAG.ts kept, no gif"
  exit 1
fi
if grep -q "not a fair one" "$RUN/log/$TAG.timer.log" 2>/dev/null; then
  echo "    $(grep -m1 'needed the path sent again' "$RUN/log/$TAG.timer.log")"
  echo "    -- $TAG.ts kept, no gif"
  exit 1
fi

# Play the clip at the speed the robots are actually moving, measured from the
# simulator's own clock across this capture rather than assumed: it varies with
# what else is competing for the four cores, and a fixed guess would make the
# race look quicker or slower than it was.
#
# Wall seconds come from the clock either side of the capture, not from $WALL.
# $WALL is the cap, and this capture stops when the race is decided: the first
# timed race ran 370 wall seconds against a 600 second window and reported
# 20.0x for a factor that measured 12.3x.
SPEED=$(python3 -c "
c0, c1, t0, t1 = ${C0:-0}, ${C1:-0}, ${T0:-0}, ${T1:-0}
sim, wall = c1 - c0, t1 - t0
print(f'{wall/sim:.3f}' if sim > 0 and wall > 0 else '')")
if [ -z "$SPEED" ]; then
  echo "    could not read the clock across this capture; $TAG.ts kept, no gif"
  echo "    (a made-up playback factor would look exactly like a measured one)"
  exit 1
fi
echo "    sim ${C0}s to ${C1}s over $(python3 -c "print(f'{${T1:-0}-${T0:-0}:.0f}')")s wall"\
     " -> playing back ${SPEED}x"

# Trim to the last finisher rather than the first: the point of the clip is who
# arrives when, and cutting at the winner throws four of the five results away.
# Trimming to the last crossing alone cut the nav2 field's clip at 14.5 s with
# three robots still driving up the course, two of which stopped within 0.06 m
# of the line: the interesting part of that race was entirely after the last
# finisher.
#
# So it is the later of the last crossing and the last moment any robot gained
# ground.  It used to be the later of the last crossing and "race over", which
# was the same thing only because the stall rule fired early -- it compared a
# robot's position against a value updated on the same odometry message, so it
# ended every race at t0 + STALL.  With that fixed a field holding a robot
# that stops short runs STALL seconds past the point where anything moves, and
# trimming there gave the nav2 and versus clips 25 s of parked robots and 9.6
# MB apiece against 5.
LAST=$(grep -oE "crossed at [0-9.]+ s|last progress at [0-9.]+s" \
       "$RUN/log/$TAG.timer.log" 2>/dev/null \
       | grep -oE "[0-9.]+" | sort -g | tail -1)
TRIM=""
if [ -n "$LAST" ]; then
  # LAST is simulated seconds from the release, and the trim is in capture wall
  # seconds: the pre-roll, plus the race, plus three simulated seconds of
  # everyone stopped at the far end.  The padding is in simulated seconds
  # rather than wall ones so it is the same three seconds of gif however slowly
  # the world happened to be running.
  END=$(python3 -c "print(f'{$PREROLL + ($LAST + 3) * $SPEED:.1f}')")
  TRIM="-t $END"
  echo "    last finisher at ${LAST}s sim; trimming the capture at ${END}s"
fi

# crop before scale: the capture window starts one pixel inside RViz's 3D
# viewport, which puts its left and right dock-splitter handles in the frame --
# a few coloured pixels at each edge of the gif, at the middle height, that
# look like world geometry and are not.  16 px off each side and 8 off the
# bottom clears them and costs nothing at the ends of the course.
ffmpeg -loglevel error -y $TRIM -i "$RUN/$TAG.ts" -vf \
  "crop=880:616:16:0,setpts=PTS/$SPEED,fps=8,scale=560:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=none" \
  -loop 0 "$G/gif/$TAG.gif" < /dev/null
ls -la "$G/gif/$TAG.gif" 2>/dev/null | awk '{printf "    gif %.1f MB\n", $5/1048576}'
