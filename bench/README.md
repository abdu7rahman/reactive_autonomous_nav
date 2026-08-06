# Benchmarks

The README claims the C++ ports are faster than the Python ones. This measures
by how much, so the claim is a number instead of an impression.

```bash
./bench/run.sh
```

Nothing here reimplements a planner. `bench_astar.py` and `bench_dwa.py` load
the real modules from `reactive_autonomous_nav/` — `rclpy` and the message
packages are stubbed just far enough for the import to succeed — and call
`_astar` and `_score_trajectories` on a bare instance with the grid attributes
wired up by hand. `bench_astar.cpp` and `bench_dwa.cpp` copy the search and the
rollout out of `cpp/src/` verbatim, swapping `nav_msgs::msg::OccupancyGrid` for
a struct with the same fields. Both languages read the same map bytes from
`maps.bin` / `local.bin`.

Measured on an Intel Xeon @ 2.10 GHz, g++ 13.3 `-O2`, Python 3.11 with
numpy 2.4. Absolute numbers will move on other hardware; the ratios are the
point.

## A\* global planner

Median of 5 runs. Maps are 8-connected costmaps with a verified-reachable
start and goal.

| Map | Python | C++ | Speedup | Nodes expanded, Python | C++ |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 × 128 | 7.50 ms | 0.046 ms | **163×** | 500 | 421 |
| 256 × 256 | 110.26 ms | 1.115 ms | **99×** | 8,186 | 4,814 |
| 384 × 384 | 884.21 ms | 4.598 ms | **192×** | 61,631 | 21,015 |

Two things are happening at once, and it is worth separating them.

The expansion counts diverge because the C++ port added a closed set. The
Python `_astar` pushes a node whenever it finds a cheaper `g`, and re-pops it
later — on the 384 × 384 map it expands 2.9× the nodes C++ does. That is an
algorithmic difference, not a language one, and it means the honest reading of
these rows is roughly 3× from the closed set and the rest from the language.

The remaining gap is what a `heapq` of Python tuples costs against a
`priority_queue` of PODs. It grows with map size because the Python version's
per-node overhead is constant while its node count grows faster.

## DWA rollout + score

Median of 25 runs, horizon `T = 25` (2.5 s at `dt = 0.1`). The Python
implementation rolls out all trajectories at once with numpy; the C++ one
sweeps `(v, ω)` in a scalar double loop with an early break on collision.

| Window | Trajectories | Python | C++ | Per-trajectory speedup |
| --- | ---: | ---: | ---: | ---: |
| Accel-limited (one control cycle) | 36 / 30 | 0.3037 ms | 0.0098 ms | **26×** |
| Full velocity space | 2,626 | 4.2121 ms | 1.0570 ms | **4×** |

The interesting result is that the gap *shrinks* as the batch grows. At the
window the controller actually evaluates each cycle, numpy is paying fixed
per-call overhead across a few dozen trajectories and loses badly. Give it
2,626 trajectories and that overhead amortises, and vectorisation closes the
gap to 4×.

So the C++ port buys the most where it matters least — both versions clear a
20 Hz budget on the per-cycle window with room to spare — and buys the least
in the wide-sweep case that would actually benefit. The global planner is where
the port earns its keep.

The two windows sample slightly different counts (36 vs 30) because
`_dynamic_window` builds its ranges with `np.arange(lo, hi + res, res)`, which
overshoots the upper bound by one sample. The table compares per-trajectory
cost to account for it. The full-velocity-space row samples 2,626 in both.

## Files

| | |
| --- | --- |
| `maps.py` | Deterministic costmaps, BFS-verified reachable, dumped for both harnesses |
| `bench_astar.py` / `.cpp` | A\* timing |
| `bench_dwa.py` / `.cpp` | DWA rollout + score timing |
| `report.py` | Prints the tables above from the result JSON |
| `run.sh` | Regenerates everything |
