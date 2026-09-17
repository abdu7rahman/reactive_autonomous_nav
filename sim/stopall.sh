#!/bin/bash
# Stop the chained bringup-and-batch runner and everything under it, leaving
# the simulator alone.  Paired with stopbatch.sh, which stops only the batch.
#
# The patterns live in this file.  A pattern typed at a prompt is also in that
# prompt's own argv, so `pkill -f` from a command line kills the command line:
# it has now happened three times in this session, each one surfacing only as
# exit code 144 with whatever was queued behind it lost.
G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

mine() {
  local p=$$
  while [ "$p" -gt 1 ]; do
    echo "$p"; p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' '); [ -z "$p" ] && break
  done
}
SAFE=" $(mine | tr '\n' ' ') "

for pat in "rerecord.sh" "full_race.sh" "all.sh" "drive.sh" "race.sh"; do
  for p in $(pgrep -f -- "$pat" 2>/dev/null); do
    case "$SAFE" in *" $p "*) continue;; esac
    kill -TERM "$p" 2>/dev/null
  done
done
sleep 4
bash $G/clean.sh
