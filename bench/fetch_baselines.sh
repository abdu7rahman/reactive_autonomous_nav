#!/usr/bin/env bash
# Pulls the third-party DWA implementations this repo is benchmarked against.
# They are fetched rather than vendored, so bench/.third_party is gitignored.
#
# The Python baselines (PythonRobotics, kmilo7204) are fetched by
# bench/dwa_compare.py at run time instead, since it needs to strip their
# plotting before importing them.
set -euo pipefail
D="$(dirname "$0")/.third_party"; mkdir -p "$D"
base=https://raw.githubusercontent.com

curl -sfL "$base/goktug97/DynamicWindowApproach/master/src/dwa.c" -o "$D/dwa.c"
curl -sfL "$base/goktug97/DynamicWindowApproach/master/src/dwa.h" -o "$D/dwa.h"

curl -sfL "$base/onlytailei/CppRobotics/master/src/dynamic_window_approach.cpp" \
  | sed '/opencv2/d' | sed -n '1,/^cv::Point2i cv_offset(/p' | sed '$d' > "$D/cpprobotics_core.h"

# amslabtech is a ROS node, so its scoring core is transcribed into
# bench/baseline_amslabtech.cpp rather than included. Its upstream is fetched
# here so the transcription can be checked against it -- see the comment at the
# top of that file for exactly which functions were lifted.
curl -sfL "$base/amslabtech/dwa_planner/master/src/dwa_planner.cpp" -o "$D/amslabtech_dwa_planner.cpp"

echo "fetched: goktug97 (C), CppRobotics (C++), amslabtech (C++, reference copy) -> $D"
