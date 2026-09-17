# Shared helpers for the simulation harness.

export PYTHONUNBUFFERED=1

# Fast DDS over UDP only.  fastdds_udp.xml says what the shared-memory
# transport does to a fifty-participant graph on this machine and how it was
# measured; this is where every process in the harness picks the profile up.
export FASTRTPS_DEFAULT_PROFILES_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fastdds_udp.xml"
#
# start <pidfile> <command...>
#   `setsid cmd & echo $!` records setsid's own pid, and when setsid forks --
#   which it does whenever the shell has already made it a group leader -- the
#   child gets a different pid and survives every later kill.  One orphaned
#   `ros2 topic pub` left that way drove the robot 1.4 m between two resets and
#   cost an afternoon of wrong conclusions about cliff sensors.  Having the
#   subshell write its own pid before exec'ing is exact.
# Output goes to <pidfile-without-.pid>.log.  A background process that keeps
# the caller's stdout open holds the pipe open too, so a script that starts one
# never appears to finish -- the first version of this hung reset_tf.sh for six
# minutes with all its work already done.
start() {
  local pf=$1; shift
  local lg="${pf%.pid}.log"
  setsid bash -c 'echo $$ > "$1"; lg="$2"; shift 2; exec stdbuf -oL -eL "$@" > "$lg" 2>&1 < /dev/null' \
      _ "$pf" "$lg" "$@" > /dev/null 2>&1 < /dev/null &
  sleep 1
}

# stop <pidfile...> -- terminate the process group, then the process
stop() {
  local f p
  for f in "$@"; do
    [ -f "$f" ] || continue
    p=$(cat "$f" 2>/dev/null)
    if [ -n "$p" ]; then
      kill -TERM -"$p" 2>/dev/null
      kill -TERM  "$p" 2>/dev/null
    fi
    rm -f "$f"
  done
}

# sim_now -- the simulator's clock, in whole seconds
#
# ros2 topic echo writes "A message was lost!!!" to stdout, not stderr, so the
# first line is not reliably the number.  Taking it unchecked put that sentence
# into a playback-speed calculation and ffmpeg failed to build its filtergraph;
# the run itself had been fine.  Only a line that is entirely digits counts.
sim_now() {
  timeout 30 ros2 topic echo /clock --once --field clock.sec 2>/dev/null \
    | grep -oE '^[0-9]+$' | head -1
}

# sim_wait <seconds> -- block until the simulator's clock has advanced that far
sim_wait() {
  local want=$1 t0 t
  t0=$(sim_now); [ -z "$t0" ] && { sleep "$want"; return; }
  while :; do
    sleep 4
    t=$(sim_now); [ -z "$t" ] && continue
    [ $((t - t0)) -ge "$want" ] && return
  done
}
