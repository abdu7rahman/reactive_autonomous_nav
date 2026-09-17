#!/bin/bash
# Stop a race.sh run and everything it started, leaving the simulator up.
#
# The pattern lives in this file rather than on a command line.  A pattern
# typed into a shell also appears in that shell's own argv, so `pkill -f` from
# the prompt kills the prompt: it happened again this session, reported only as
# exit code 144, and took the edit that was queued behind it with it.
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

# Anchored, not a substring.  `race.sh` matched every process whose argv
# merely contained those characters, and a wrapper script called rerace.sh was
# enough: this killed the shell that had just launched it, reported only as
# exit code 144.  A slash or a space before it and a space or the end after it
# is the difference between the script and a name that ends in it.
PAT='(^|[ /])race\.sh( |$)'
for p in $(pgrep -f -- "$PAT" 2>/dev/null); do
  case "$SAFE" in *" $p "*) continue;; esac
  kill -TERM "$p" 2>/dev/null
done
sleep 3
stop $G/run/job/*.pid
bash $G/clean.sh
