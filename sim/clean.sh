#!/bin/bash
# Kill every per-run process before a run, whether or not this harness started
# it.  Two controllers, two costmap pairs and an hour-old manual publisher were
# once all writing to /cmd_vel_unstamped at once; the stale one was sending
# v=0.25 at 10 Hz and that is what the robot followed, so every reading taken
# about the controller while it ran was taken about the wrong command stream.
#
# The ancestor guard is not paranoia.  A match pattern also appears in the argv
# of the shell that launched this script, so an unguarded sweep kills its own
# caller; it has happened twice, both times reported only as exit code 144.
. /opt/rosenv.sh
G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. $G/lib.sh

# every pid from here up to init -- never a candidate
mine() {
  local p=$$
  while [ "$p" -gt 1 ]; do
    echo "$p"
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    [ -z "$p" ] && break
  done
}
SAFE=" $(mine | tr '\n' ' ') "

zap() {
  local sig=$1 pat p
  shift
  for pat in "$@"; do
    for p in $(ps -eo pid,args | grep -- "$pat" | awk '{print $1}'); do
      case "$SAFE" in *" $p "*) continue;; esac
      kill -"$sig" "$p" 2>/dev/null
    done
  done
}

stop $G/run/job/*.pid

# Patterns carry the installed path, not just the node name.  A bare
# "dwa_controller" matches any process whose argv mentions the file -- an
# editor, a grep, a colcon build, the shell running this script's caller -- and
# this swept away a shell that was in the middle of patching those very files,
# reported as exit code 144 with the work half done.  The ancestor guard below
# only protects this script's own ancestry; it cannot protect a sibling.
L=lib/reactive_autonomous_nav
PATS=("$L/dwa_controller" "$L/pure_pursuit_controller" "$L/stanley_controller"
      "$L/teb_controller" "$L/mppi_controller" "$L/astar_planner"
      "$L/theta_star_planner" "$L/smac_planner" "$L/rrt_planner"
      "$L/rrt_smac_hybrid_planner"
      "nav2_costmap_2d/nav2_costmap_2d" "nav2_lifecycle_manager/lifecycle_manager"
      "bin/rviz2")
zap TERM "${PATS[@]}"
sleep 6
zap KILL "${PATS[@]}"
sleep 2

# pgrep leaves itself out of its own results; the plain ps-and-grep version
# counted the grep as a survivor and reported 13 leftovers of a sweep that had
# in fact left nothing running.  A check that always fails is not a check.
left=0
for pat in "${PATS[@]}"; do
  for p in $(pgrep -f -- "$pat" 2>/dev/null); do
    case "$SAFE" in *" $p "*) continue;; esac
    left=$((left + 1))
  done
done
echo "clean: $left per-run processes left"
