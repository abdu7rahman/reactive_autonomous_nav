#!/bin/bash
# Tear the simulator down completely.  Paired with sim_up.sh.
#
# Patterns live in this file, never on a command line: a pattern typed into a
# shell also lands in that shell's own argv, and an unguarded sweep then kills
# its own caller.  The ancestor guard is the second belt.
. /opt/rosenv.sh
G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. $G/lib.sh

mine() {
  local p=$$
  while [ "$p" -gt 1 ]; do
    echo "$p"
    p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    [ -z "$p" ] && break
  done
}
SAFE=" $(mine | tr '\n' ' ') "

bash $G/clean.sh > /dev/null 2>&1
stop $G/run/sim/*.pid

PATS=("gz sim" "parameter_bridge" "turtlebot4_spawn" "create3_nodes"
      "irobot_create" "robot_state_publisher" "ros_gz_sim" "map_server"
      "cmd_relay.py" "localize.py" "static_transform_publisher"
      "turtlebot4_node" "ros_gz_bridge" "Xvfb"
      # joint_state_publisher was missing from this list for ten runs.  Six of
      # them survived every teardown -- one from the first single-robot bringup
      # and one per robot from the five-robot attempt -- and were still in
      # `ros2 node list` three hours later, two of them claiming the same node
      # name in the same namespace.
      "joint_state_publisher" "spawner")
for sig in TERM KILL; do
  for pat in "${PATS[@]}"; do
    for p in $(pgrep -f -- "$pat" 2>/dev/null); do
      case "$SAFE" in *" $p "*) continue;; esac
      kill -"$sig" "$p" 2>/dev/null
    done
  done
  sleep 5
done

left=0
for pat in "${PATS[@]}"; do
  for p in $(pgrep -f -- "$pat" 2>/dev/null); do
    case "$SAFE" in *" $p "*) continue;; esac
    left=$((left + 1))
  done
done
echo "sim down: $left simulator processes left"
