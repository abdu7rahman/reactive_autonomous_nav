#!/bin/bash
# Re-encode one clip from its retained capture, finding the arrival by the
# marker every controller produces rather than by dwa_controller's own wording.
#
#   reencode.sh <tag>
#
# drive.sh looked for "Goal REACHED", which only dwa_controller logs. The other
# four publish REACHED on their status topic without printing that string, so a
# pure_pursuit run that finished was reported as reached: 0 and its clip was
# left untrimmed. What all five share is the planner's reply to that status:
# "Goal reached - stopping replanning".
#
# The capture's start is not recorded, so the trim is taken relative to the
# first plan instead: the capture opens, waits four seconds, the goal goes out,
# and the planner answers within about a second. delta + 9 covers that lead-in
# and leaves four seconds of tail after arrival.
. /opt/rosenv.sh
G=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TAG=$1
LOG=$G/run/log/$TAG.nav.log
MP4=$G/run/$TAG.mp4
[ -f "$MP4" ] || { echo "$TAG: no capture"; exit 1; }
[ -f "$LOG" ] || { echo "$TAG: no log"; exit 1; }

stamp() { grep -oE "\[[0-9]{10}\.[0-9]+\]" | head -1 | tr -d "[]"; }
FIRST=$(grep -m1 "Path:" "$LOG" | stamp)
REACH=$(grep -m1 -E "Goal REACHED|Goal reached|stopping replanning" "$LOG" | stamp)
SPEED=$(grep -oE "playing back [0-9.]+x" "$G/run/log/$TAG.speed" 2>/dev/null \
        | grep -oE "[0-9.]+" | head -1)
[ -z "$SPEED" ] && SPEED=$2
[ -z "$SPEED" ] && { echo "$TAG: no measured playback speed; pass it as arg 2"; exit 1; }

TRIM=""
if [ -n "$FIRST" ] && [ -n "$REACH" ]; then
  END=$(python3 -c "print(f'{max(8.0, $REACH - $FIRST + 9):.1f}')")
  TRIM="-t $END"
  echo "$TAG: arrival ${END}s into the capture, playback ${SPEED}x"
else
  echo "$TAG: no arrival found, keeping the whole capture"
fi

ffmpeg -loglevel error -y $TRIM -i "$MP4" -vf \
  "setpts=PTS/$SPEED,fps=10,scale=560:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=48[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3" \
  -loop 0 "$G/gif/$TAG.gif" < /dev/null
ls -la "$G/gif/$TAG.gif" | awk '{printf "  %s  %d KB\n", $9, $5/1024}'
