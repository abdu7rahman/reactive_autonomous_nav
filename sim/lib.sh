# Shared helpers for the simulation harness.

export PYTHONUNBUFFERED=1

# Fast DDS over UDP only, in one place for every run.
#
# fastdds_udp.xml records what the shared-memory transport does to a
# fifty-participant race graph, which is where this is load-bearing: with
# shared memory on, every new participant logged "Failed init_port
# fastrtps_port7000" and the costmap lifecycle managers hung for eight
# minutes; with it off, none did.
#
# For a fourteen-node single-robot graph it is neither help nor harm -- the
# /map failures that cost most of an afternoon happened identically with and
# without it, and their cause was `ros2 lifecycle set` not finding a live node
# (see lifecycle_up.py). It was scoped to the race scripts for a while on the
# belief that it was what made four of ten clips miss their goal; that was
# wrong too, those were dwa_controller's dynamic window. One export beats two
# copies, and loopback UDP costs throughput nothing here is near using.
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
# clock_now.py rather than `ros2 topic echo /clock --once`, for the reason its
# header gives: the CLI gives discovery about a second and that is not enough
# on this machine under load, and every playback factor in the harness is two
# readings of this clock.  The CLI also wrote "A message was lost!!!" to
# stdout, not stderr, so the first line was not reliably the number -- that
# sentence went into a playback-speed calculation once and ffmpeg failed to
# build its filtergraph on a run that had been perfectly good.
sim_now() {
  local G
  G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  timeout 40 python3 $G/clock_now.py 25 2>/dev/null | grep -oE '^[0-9]+$' | head -1
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

# sim_ok -- do the four topics a run cannot start without actually deliver?
#
# `ros2 topic list` is not enough and neither is a publisher count: a bringup
# that looks complete in the graph and delivers nothing is the failure this
# harness keeps hitting. sim_up.sh printed "core topics present: 2/4" and then
# printed SIMUP anyway, and all.sh spent two hours recording a world with no
# lidar in it. /clock is the one that matters most -- every use_sim_time node
# sits frozen at t=0 without it -- so it is checked by reading the clock rather
# than by counting publishers.
sim_ok() {
  local G bad=0
  G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  [ -z "$(sim_now)" ] && { echo "  /clock: no reading"; bad=1; }
  timeout 40 python3 $G/wait_topic.py /scan LaserScan 30 0 > /dev/null 2>&1 \
    || { echo "  /scan: nothing in 30s"; bad=1; }
  timeout 40 python3 $G/wait_topic.py /odom Odometry 30 0 > /dev/null 2>&1 \
    || { echo "  /odom: nothing in 30s"; bad=1; }
  timeout 60 python3 $G/wait_topic.py /map OccupancyGrid 45 10000 > /dev/null 2>&1 \
    || { echo "  /map: no grid with 10000 populated cells in 45s"; bad=1; }
  return $bad
}
