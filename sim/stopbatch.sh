#!/bin/bash
# Stop a running batch without touching the simulator.
# Patterns live here rather than on a command line, and the ancestor guard
# keeps the sweep off its own caller.
G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mine() {
  local p=$$
  while [ "$p" -gt 1 ]; do
    echo "$p"; p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' '); [ -z "$p" ] && break
  done
}
SAFE=" $(mine | tr '\n' ' ') "
for pat in "all.sh" "drive.sh"; do
  for p in $(pgrep -f -- "$pat" 2>/dev/null); do
    case "$SAFE" in *" $p "*) continue;; esac
    kill -TERM "$p" 2>/dev/null
  done
done
sleep 5
bash $G/clean.sh
