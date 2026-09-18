#!/usr/bin/env bash
# Regenerates every number in bench/README.md. Run from the repo root.
set -euo pipefail

python3 bench/maps.py
g++ -O2 -std=c++17 -Wall -o bench/bench_astar bench/bench_astar.cpp
g++ -O2 -std=c++17 -Wall -o bench/bench_dwa   bench/bench_dwa.cpp

# The four-way C++ comparison, which had no build line anywhere: the table in
# the README was produced by a command nobody wrote down, against a README that
# opens "everything below is reproducible from bench/".  goktug97's is C rather
# than C++ -- g++ rejects its implicit void* conversions -- so it is compiled
# by gcc and linked in, which is also how its own build does it.
if [ -f bench/.third_party/dwa.c ]; then
  gcc -O2 -std=c11 -I bench/.third_party -c bench/.third_party/dwa.c -o bench/dwa_goktug.o
  g++ -O2 -std=c++17 -I bench -o bench/dwa_compare_cpp \
      bench/dwa_compare_cpp.cpp bench/baseline_cpprobotics.cpp \
      bench/baseline_goktug.cpp bench/baseline_amslabtech.cpp bench/dwa_goktug.o
else
  echo "bench/.third_party is empty -- run bench/fetch_baselines.sh for the"
  echo "C and C++ comparison; skipping it"
fi

echo; python3 bench/bench_astar.py
echo; ./bench/bench_astar bench/maps.bin 5  | tee bench/cpp_astar.json
echo; python3 bench/bench_dwa.py
echo; ./bench/bench_dwa   bench/local.bin 25 | tee bench/cpp_dwa.json
echo; [ -x bench/dwa_compare_cpp ] && ./bench/dwa_compare_cpp

# The three tables this script used to leave out, against a README that opens
# "everything below is reproducible from bench/": the Python four-way, the
# Nav2-geometry planner comparison, and the two closed-loop figures.
echo; python3 bench/dwa_compare.py
echo; python3 bench/nav2_compare.py

# python3.12 for the figures: they render with matplotlib, and the one in
# dist-packages is built for 3.12 while python3 here is 3.11.  Skipped rather
# than failed if that interpreter is absent, since every table above is
# independent of it.
if command -v python3.12 > /dev/null && \
   python3.12 -c "import matplotlib" 2> /dev/null; then
  echo; python3.12 bench/gif_compare.py
  echo; [ -x bench/dwa_compare_cpp ] && python3.12 bench/gif_compare.py --cpp
else
  echo; echo "no python3.12 with matplotlib -- skipping the closed-loop figures"
fi

echo; python3 bench/report.py
