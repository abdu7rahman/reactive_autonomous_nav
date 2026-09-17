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

Four implementations, same dynamic window, same trajectory count, same 25-step
horizon. Baselines are fetched by `bench/fetch_baselines.sh`, not vendored.

| Trajectories | This repo (C++) | [CppRobotics](https://github.com/onlytailei/CppRobotics) (C++) | [goktug97](https://github.com/goktug97/DynamicWindowApproach) (C) | [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) (Py) |
| ---: | ---: | ---: | ---: | ---: |
| 36 | **0.011 ms** | 0.013 ms | 0.091 ms | 2.30 ms |
| 100 | **0.030 ms** | 0.032 ms | 0.303 ms | 7.16 ms |
| 400 | **0.120 ms** | 0.142 ms | 1.353 ms | 33.84 ms |
| 900 | **0.275 ms** | 0.315 ms | 3.159 ms | 78.25 ms |
| 2,500 | **0.760 ms** | 0.884 ms | 8.950 ms | 219.36 ms |

Read the middle column first. Against another C++ DWA this repo is within
10–20 percent — near parity, not a win. Most of the 60× over PythonRobotics is
Python versus C++, and most of the 8–12× over goktug97 is that it walks a point
cloud per sample where this repo does an O(1) costmap lookup.

That lookup is the one structural difference, and it shows up as flat scaling
in clutter (Python side, 400 trajectories):

| Obstacles | This repo | PythonRobotics |
| ---: | ---: | ---: |
| 20 | 0.64 ms | 29.40 ms |
| 500 | 0.64 ms | 68.17 ms |
| 2,000 | 0.64 ms | **326.99 ms** |

Flat versus linear. A costmap has to be built and maintained, so this is a
trade rather than a free win.

ROS-coupled implementations (`nav2_dwb_controller`, `amslabtech/dwa_planner`,
`teb_local_planner`) are not in the table: they need a live ROS 2 graph and
costmap plugins to run at all, so any number taken outside that would be
measuring the harness. The Nav2 comparison below uses their published figures
instead.

## Global planner vs Nav2

Reference is Table I of Macenski et al., [*Cost-Aware Kinematically Feasible
Planning for Mobile and Surface Robotics*](https://arxiv.org/abs/2401.13078).
`nav2_maps.py` rebuilds their map and query geometry: 10,000 m² random
occupancy maps at 5 cm resolution (2000 × 2000 cells), ~50 m paths.

| Density | This repo, C++ A\* | Smac 2D-A\* | NavFn | Hybrid-A\* | SBPL ARA\* |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | **5.0 ms** | 66.2 ms | 71.1 ms | 39.1 ms | 5,640 ms |
| 15% | **6.8 ms** | 85.6 ms | 66.5 ms | 40.7 ms | 6,587 ms |
| 20% | **15.1 ms** | 88.8 ms | 61.0 ms | 38.8 ms | 6,633 ms |

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
