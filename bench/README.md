# Benchmarks and tests

Two things live here: correctness suites that every planner and controller has
to pass, and timing comparisons against published baselines.

```bash
python3 bench/test_planners.py     # correctness, ~8 min
python3 bench/test_dwa_window.py   # DWA's velocity window and stuck detector
python3 bench/test_chicane.py      # all five controllers down the race reference path
python3 bench/test_views.py        # every RViz display shows a topic something publishes
python3 bench/dwa_compare.py       # vs PythonRobotics DWA
python3 bench/nav2_compare.py      # vs Nav2 Smac Planner paper
./bench/run.sh                     # Python vs C++ latency

python3 bench/sweep_stanley.py     # Stanley's k, k_soft, wheelbase
python3 bench/sweep_teb.py         # TEB's acceleration edges and vertex spacing
python3 bench/sweep_mppi.py        # MPPI's temperature, against effective sample size
python3 bench/chase_mppi.py       # the site's cursor-chase plate, both weightings
```

The sweeps exist because a controller that carries a published name should
carry its published constants, and picking those by eye is how Stanley's gain
ended up at 4.0 compensating for a reference point in the wrong place.

Nothing is reimplemented. `rig.py` stubs `rclpy` and the message packages just
far enough for the modules to import, then builds each node with
`object.__new__` and wires the grid and state attributes it reads. Parameters
come from the node's own `__init__` by AST extraction, so a test can never
silently drift from the shipped tuning.

## What "passing" means

A global plan passes only if it is connected, in bounds, ends at the goal, and
**no segment enters a blocked cell**. Checking only the waypoints is exactly
what let a Bresenham line-of-sight return paths through walls — see below.

A controller passes only if driving its own `_control_loop` around a unicycle
plant reaches the goal without touching a lethal cell.

Maps are chosen so a straight start-to-goal line is always blocked, which
catches a planner that quietly returns the trivial path instead of flattering
it.

## Bugs this found

Every one of these was live before the suite existed.

| Component | Bug | Effect |
| --- | --- | --- |
| Theta\* | Bresenham line-of-sight walks a *thin* line and skips cells the segment enters | Returned paths **straight through walls**. 13 of 18 segments on a maze were invalid |
| RRT | `_is_free` used `v < LETHAL_COST` with no lower bound | `OccupancyGrid.data` is int8, so cost 254 arrives as −2 and **every wall read as free** |
| RRT | Line-of-sight sampled at half a cell | Stepped over a 2.15 cm clip with 2.7 cm sampling |
| RRT | Goal link connected on distance alone | Final hop onto the goal was never collision-checked |
| SMAC | Straight motion primitive returned only its endpoint | 0.15 m move spans 3 cells at 5 cm; 1 was checked |
| Hybrid | Goal gate needed distance **and** heading simultaneously on a 0.3 m lattice | Never terminated — tree stalled 0.53 m from the goal at any iteration count |
| Hybrid | Smoother checked the moved point, not the legs into it | Waypoint slid somewhere free while its leg cut a corner through a wall |
| Hybrid | `_get_merged_cost` / `_is_arc_collision_free` defined twice | 42 lines of dead code; same int8 sign bug as RRT |
| Pure Pursuit | Lookahead scanned from index 0 | Once a lookahead from the start, the start itself qualified — the robot **turned around and chased its own path start** |
| Stanley | Closest-point search scanned the whole path | Could snap the reference onto an earlier leg |
| TEB | Elastic band built once from the head of the path, never advanced | Robot orbited waypoint 2 forever |
| DWA | Lookahead was a fixed waypoint count, so where the plan wrapped an obstacle it aimed at a waypoint on the far side of the wall | Drove into the near face of a pillar and sat there for 600 steps, **4.57 m from a reachable goal**. Clamping the lookahead to a *visible* waypoint also took rooms-200 from 827 steps to 469 |
| TEB | Band carried no time intervals at all — the velocity came from `min(max_vel, dist * 2.0)` against band index 2 | Commanded speed was a function of how finely the plan happened to be sampled rather than of any limit the robot has. 39–41% more steps than the timed band needs |
| MPPI | Softmax `lambda` fixed at 0.3 while the critics sum to about 300 | `exp(-300/0.3)` collapses the weighting onto one rollout: effective sample size **1.00 of 1000**, which is random shooting, not MPPI. Reverse commands appeared in 19 of 20 chase runs |
| bench | `_sig()` called above its own definition in every entry point | `test_planners.py` — the suite that gates a push — died with `NameError` before printing a line |

A\* came through clean: zero true corner-cuts across every map, and the 14
diagonal steps that squeeze past one blocked orthogonal are legal for a point
robot on an inflated costmap.

## MPPI's temperature has an optimum, which is the point

Once `lambda` is scaled to the units of the cost, `temperature` is a ratio and
the weighting has somewhere sensible to sit. Effective sample size is
`1 / sum(w^2)` — how many of the 1000 rollouts the update actually averages
over. One is argmin; a thousand is the prior with the costs ignored.

| temperature | effective samples | steps over both maps |
| ---: | ---: | ---: |
| 0.05 | 1.0 | 1125 |
| 0.10 | 2.0 | 1108 |
| **0.30** | **34.4** | **1082** |
| 0.60 | 210.9 | 1119 |
| 1.00 | 493.9 | 1163 |

It degrades at both ends and the interior minimum sits where the effective
sample size is a few dozen. That shape is the corroboration: before the units
were fixed there was no optimum to find, because every temperature in this range
collapsed to a single sample and the controller was doing the same thing at all
of them.

Every row passes both maps, so the suite's pass/fail says nothing here — which
is why `sweep_mppi.py` prints the effective sample size beside the step count.

## The chase demo's two symptoms were not one bug

The site's cursor chase showed MPPI reversing at the start and settling 0.33 m
out against a 0.15 m goal tolerance. Neither reproduced on the controller
suite, whose maps are 153 and 200 cells wide and whose reference path is
decimated by two. The chase is an 86x44 plate at 0.05 m a cell and hands the
controller the raw A\* cell path, one waypoint every five centimetres.
`chase_mppi.py` rebuilds those conditions, and they turn out to be two
different things.

**The reversal was the softmax.** Running the version from before
`fix(mppi): scale the softmax temperature to the units of the cost` against the
version after it, on the same five goals and the same seed:

| goal | wps | min commanded v, before | after |
| --- | --- | --- | --- |
| (2.50, 1.10) | 39 | **−0.229** m/s | +0.066 |
| (1.50, 1.10) | 19 | **−0.211** | +0.041 |
| (1.20, 1.60) | 12 | **−0.158** | −0.011 |
| (3.40, 0.60) | 57 | **−0.089** | +0.027 |
| (1.00, 1.10) | 9 | **−0.044** | +0.068 |

Every goal reverses before the fix and none of them meaningfully does after.
The −0.089 m/s row is the figure the symptom was first reported at.

**The 0.33 m was the plate's goal snap, not the controller.** A cursor inside
the inflation band is moved to the nearest cell at or below `FREE_COST` before
A\* sees it. Against nav2's exponential falloff at a four-cell radius, the
first such cell is five cells from a wall, so a cursor in a corner moves the
goal by up to **0.346 m** — and the robot then arrives at the goal it was
given while the readout measures to the cursor. Any controller shows it. The
demo now names the offset instead of reporting a tolerance it is not measuring
against.

## Known bounds, not bugs

The kinematic planners carry a 0.22 m minimum turning radius, so they need
about 0.44 m to come about. A maze with 0.5 m corridors sits at that bound and
the hybrid fails it at 8k, 30k and 60k iterations alike — geometry, not search
budget. They are scored on maps whose corridors fit, and the tight mazes score
the holonomic planners only.

RRT's default was raised from 2,000 to 20,000 iterations. Uniform sampling
needs far more draws to thread a narrow passage, and the loop breaks as soon as
the goal connects, so open maps still finish in ~20 ms.

## Local controller vs other DWA implementations

Six implementations, same dynamic window, same trajectory count, same 25-step
horizon. Baselines are fetched, not vendored: `bench/fetch_baselines.sh` for
the C and C++ ones, and `bench/dwa_compare.py` pulls the Python ones at run
time. Only their plotting is stripped; the planner functions are theirs.

Every number below is the median of three full runs of the comparison, and the
run-to-run spread is given with it, because it is wide enough to matter: at
these durations the difference between two runs of the same binary is up to 14
percent on the C++ side and up to 18 on the Python side. Nothing smaller than
that is being claimed.

**C and C++** — `bench/dwa_compare_cpp.cpp`, built by `bench/run.sh`

| Trajectories | This repo | [CppRobotics](https://github.com/onlytailei/CppRobotics) | [goktug97](https://github.com/goktug97/DynamicWindowApproach) (C) | [amslabtech](https://github.com/amslabtech/dwa_planner) |
| ---: | ---: | ---: | ---: | ---: |
| 36 | **0.012 ms** | 0.015 ms | 0.094 ms | 0.314 ms |
| 100 | **0.032 ms** | 0.038 ms | 0.317 ms | 0.910 ms |
| 400 | **0.136 ms** | 0.169 ms | 1.451 ms | 3.636 ms |
| 900 | **0.298 ms** | 0.368 ms | 3.380 ms | 8.519 ms |
| 2,500 | **0.876 ms** | 1.092 ms | 9.481 ms | 22.647 ms |

Worst spread over the three runs: 6% for this repo, 9% for CppRobotics, 14%
for goktug97, 11% for amslabtech.

**Python** — `bench/dwa_compare.py`

| Trajectories | This repo | [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) | [kmilo7204](https://github.com/kmilo7204/dwa_python) |
| ---: | ---: | ---: | ---: |
| 36 | **0.31 ms** | 2.66 ms | 4.12 ms |
| 100 | **0.36 ms** | 8.53 ms | 11.43 ms |
| 400 | **0.77 ms** | 37.69 ms | 46.93 ms |
| 900 | **1.83 ms** | 86.40 ms | 103.52 ms |
| 2,500 | **4.21 ms** | 246.36 ms | 284.50 ms |

Worst spread: 18% for this repo at 900 trajectories, 7% everywhere else.

Read the CppRobotics column first. Against another C++ DWA this repo is 20 to
25 percent quicker across the range, which is a margin and not a rout, and it
is the row that makes the rest of the table worth reading: the gaps elsewhere
are structural rather than a faster inner loop.

An earlier version of this table had this repo *losing* the 36-trajectory row,
0.017 ms against 0.013. Three fresh runs of both put it at 0.012 against
0.015, which is outside the spread in the other direction. The old numbers are
not reproducible from this tree and have been replaced rather than explained.

The structure is that every baseline here keeps an explicit obstacle list and
measures each rollout point against every obstacle, while this repo reads one
costmap cell. amslabtech does that per point in C++ with no vectorisation,
which is the 26× at 2,500 trajectories. goktug97 walks a point cloud per
sample, which is the 8–11×. PythonRobotics vectorises the comparison over
numpy and kmilo7204 does not, which is why they sit where they do relative to
each other.

It shows up directly as flat scaling in clutter (Python side, 400
trajectories):

| Obstacles | This repo | PythonRobotics | kmilo7204 |
| ---: | ---: | ---: | ---: |
| 20 | 0.73 ms | 33.86 ms | 41.26 ms |
| 100 | 0.73 ms | 41.42 ms | 48.91 ms |
| 500 | 0.73 ms | 77.59 ms | 88.37 ms |
| 2,000 | **0.73 ms** | 320.63 ms | **426.93 ms** |

Flat versus linear, and flat to the 1% spread of the measurement. A costmap has
to be built and maintained by something else first, so this is a trade rather
than a free win.

`amslabtech/dwa_planner` is a ROS node, so its scoring core is transcribed into
`bench/baseline_amslabtech.cpp` rather than included — `motion`,
`generate_trajectory`, `calc_dynamic_window`, `calc_to_goal_cost`,
`calc_obs_cost` and `calc_speed_cost`, with only Eigen and the message types
stubbed. `python3 bench/verify_amslabtech.py` diffs each of those bodies
against the upstream file and reports which substitutions were made; on the
current upstream it reports every lifted body identical, one of them after the
`Eigen::Vector3d` stub swap.

`nav2_dwb_controller` and the other nav2 local controllers are not in this
table and cannot be: they need a live ROS 2 graph and costmap plugins to run at
all, so any number taken here would be measuring the harness. They are measured
in that live graph instead, against this repo's DWA on the same robot, the same
costmap settings and the same path — see **The nav2 field** in `sim/README.md`.

## Global planner vs Nav2

Reference is Table I of Macenski et al., [*Cost-Aware Kinematically Feasible
Planning for Mobile and Surface Robotics*](https://arxiv.org/abs/2401.13078).
`nav2_maps.py` rebuilds their map and query geometry: 10,000 m² random
occupancy maps at 5 cm resolution (2000 × 2000 cells), ~50 m paths.

| Density | This repo, C++ A\* | Smac 2D-A\* | NavFn | Hybrid-A\* | SBPL ARA\* |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | **3.3 ms** | 66.2 ms | 71.1 ms | 39.1 ms | 5,640 ms |
| 15% | **5.8 ms** | 85.6 ms | 66.5 ms | 40.7 ms | 6,587 ms |
| 20% | **12.4 ms** | 88.8 ms | 61.0 ms | 38.8 ms | 6,633 ms |

Read with the caveats. Their CPU (Ryzen 5 5600X) is considerably faster than
the one these came off, which flatters this repo. Against that, Smac 2D-A\* is
cost-aware and returns a smoothed path, and NavFn solves a full navigation
function — both do more per call than a plain octile A\*. The honest claim is
same order of magnitude on equivalent maps, not that it beats Nav2.

## Python vs C++ in this repo

| Map | Python A\* | C++ A\* | Speedup |
| --- | ---: | ---: | ---: |
| 128 × 128 | 7.50 ms | 0.046 ms | 163× |
| 256 × 256 | 110.26 ms | 1.115 ms | 99× |
| 384 × 384 | 884.21 ms | 4.598 ms | 192× |

Not all language: the C++ port also added a closed set, so it expands 21,015
nodes where Python expands 61,631 on the same map. Roughly 3× is algorithmic.

For the DWA rollout the gap *shrinks* with batch size — 26× per trajectory at
the accel-limited window the controller actually evaluates, down to 4× at
2,626 trajectories, because numpy's fixed per-call overhead amortises away.

## Files

| | |
| --- | --- |
| `rig.py` | ROS stubs, node loader, path validators, closed-loop driver |
| `maps.py` | Mazes, room maps, costmap inflation |
| `test_planners.py` | The correctness suite |
| `test_dwa_window.py` | DWA's reachable window, orbit detection, closed loop |
| `test_chicane.py` | The five-robot race's reference path, driven before the simulator sees it |
| `test_views.py` | Every RViz display against the package's own publishers |
| `chase_mppi.py` | The site's chase plate: both MPPI weightings, and the goal snap |
| `dwa_compare.py` | vs PythonRobotics |
| `nav2_maps.py`, `nav2_compare.py` | vs the Nav2 Smac Planner paper |
| `bench_astar.*`, `bench_dwa.*` | Python vs C++ latency |

Measured on an Intel Xeon @ 2.10 GHz, g++ 13.3 `-O2`, Python 3.11, numpy 2.4.
Absolute numbers move with hardware; the ratios are the point.
