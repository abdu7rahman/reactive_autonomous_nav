#!/bin/bash
# Record all ten clips: five planners against a fixed controller, five
# controllers against a fixed planner.
#
# dwa holds the controller slot while the planners vary and astar holds the
# planner slot while the controllers vary, so each half changes one thing.
# astar-with-dwa is therefore the same configuration twice; it is recorded once
# and the second name is a copy rather than a second run, because two runs of
# the same pair would differ only by simulator noise and inviting a comparison
# between them would be inviting a reader to read noise as a result.
. /opt/rosenv.sh
G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. $G/lib.sh
mkdir -p $G/gif $G/run/log
# 600, not 260.  The capture window has to outlast the slowest run at
# whatever real-time factor the machine happens to be giving, and the slowest
# run is now much slower than it was: theta_star with dwa orbits the shelf
# corner three times, recovers three times and arrives about 46 simulated
# seconds after the goal goes out, against astar's 17. At a measured playback
# factor of 7.5 that is 350 to 420 wall seconds, and a 400 s window closed on
# it at wp=70/71, 0.25 m from its goal at 0.42 m/s. drive.sh trims each clip
# to its own arrival, so a window with room in it costs capture time and not a
# byte of gif.
SECS=${SECS:-600}

# Skip a clip that is already recorded, so a batch stopped halfway can be
# restarted without paying again for the runs that already worked.  Delete the
# gif to force a re-record.
run() {  # planner controller tag
  if [ -s "$G/gif/$3.gif" ]; then
    echo "=== $3 : already recorded, skipping" | tee -a $G/run/log/batch.txt
    return
  fi
  stdbuf -oL bash $G/drive.sh "$1" "$2" "$3" "$SECS" 2>&1 \
    | stdbuf -oL grep -vE "^\[[0-9]+\]" | tee -a $G/run/log/batch.txt
}

# Refuse to start against a broken simulator. A batch is two hours; a bringup
# that came up without a lidar cost exactly that once, and the only sign was a
# line in sim_up.sh's log that nothing read.
# `if ! sim_ok | tee ...` is not this check: a pipeline's status is its last
# command's and tee always succeeds, so the gate could never fail. It was
# written that way once and the batch started against a simulator with no
# /map, launching a nav stack per run for as long as nobody looked.
echo "=== simulator check" | tee -a $G/run/log/batch.txt
sim_ok > $G/run/log/simcheck.txt 2>&1; ok=$?
tee -a $G/run/log/batch.txt < $G/run/log/simcheck.txt
if [ "$ok" != "0" ]; then
  echo "simulator is not delivering -- run sim_up.sh first" | tee -a $G/run/log/batch.txt
  exit 1
fi

touch $G/run/log/batch.txt

for p in astar theta_star smac rrt rrt_smac_hybrid; do
  run "$p" dwa "planner-$p"
done

for c in pure_pursuit stanley teb mppi; do
  run astar "$c" "controller-$c"
done

cp -f $G/gif/planner-astar.gif $G/gif/controller-dwa.gif
cp -f $G/run/log/planner-astar.nav.log $G/run/log/controller-dwa.nav.log
echo "controller-dwa: copied from planner-astar (same pair)" | tee -a $G/run/log/batch.txt

echo
echo "=== results ==="
# Four controllers, four arrival wordings -- the same pattern drive.sh trims
# on.  Counting "Goal REACHED" alone is dwa's wording, and it reported the
# other eight runs as reached=0 with their robots parked on the goal.
#
# gif/ also holds race.gif, which is not one of these ten.
for p in planner-astar planner-theta_star planner-smac planner-rrt \
         planner-rrt_smac_hybrid controller-dwa controller-pure_pursuit \
         controller-stanley controller-teb controller-mppi; do
  f=$G/gif/$p.gif
  [ -s "$f" ] || { printf "%-26s missing\n" "$p"; continue; }
  r=$(grep -cE "Goal REACHED|Goal reached|stopping replanning" \
      "$G/run/log/$p.nav.log" 2>/dev/null)
  printf "%-26s %5s KB   reached=%s\n" "$p" "$(( $(stat -c%s "$f") / 1024 ))" "${r:-0}"
done
