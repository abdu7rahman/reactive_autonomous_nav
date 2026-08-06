#!/usr/bin/env bash
# Regenerates every number in bench/README.md. Run from the repo root.
set -euo pipefail

python3 bench/maps.py
g++ -O2 -std=c++17 -Wall -o bench/bench_astar bench/bench_astar.cpp
g++ -O2 -std=c++17 -Wall -o bench/bench_dwa   bench/bench_dwa.cpp

echo; python3 bench/bench_astar.py
echo; ./bench/bench_astar bench/maps.bin 5  | tee bench/cpp_astar.json
echo; python3 bench/bench_dwa.py
echo; ./bench/bench_dwa   bench/local.bin 25 | tee bench/cpp_dwa.json
echo; python3 bench/report.py
