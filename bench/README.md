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

Seven implementations, same dynamic window, same sampling resolution, same
25-step horizon. Baselines are fetched, not vendored: `bench/fetch_baselines.sh` for
the C and C++ ones, and `bench/dwa_compare.py` pulls the Python ones at run
time. Only their plotting is stripped; the planner functions are theirs.

Every number below is the median of four full runs of the comparison, and the
run-to-run spread is given with it.

**C and C++** — `bench/dwa_compare_cpp.cpp`, built by `bench/run.sh`

| Trajectories | This repo | [CppRobotics](https://github.com/onlytailei/CppRobotics) | [goktug97](https://github.com/goktug97/DynamicWindowApproach) (C) | [amslabtech](https://github.com/amslabtech/dwa_planner) |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.027 ms | **0.015 ms** | 0.100 ms | 0.377 ms |
| 110 | 0.071 ms | **0.038 ms** | 0.330 ms | 0.982 ms |
| 420 | 0.277 ms | **0.167 ms** | 1.470 ms | 4.117 ms |
| 930 | 0.619 ms | **0.362 ms** | 3.406 ms | 8.799 ms |
| 2,550 | 1.714 ms | **1.024 ms** | 9.692 ms | 24.312 ms |

Per-cell spread across the four runs is 2.4% median. Two cells are far wider
and both are in the two smallest rows, where a call takes 15 to 70 µs and the
steady_clock resolution shows: 41% for this repo at 110 trajectories and 38%
for amslabtech at 42. The gap to CppRobotics is outside the spread at every
count.

The trajectory counts are 42 and 110 rather than 36 and 100 because the sweep
now samples the window the way the controller does — `samples()`, a lattice of
multiples of the resolution with both bounds included — and the bounds add a
sample to each axis. Which means **the four do not evaluate the same number of
trajectories**, even given the same window and the same resolution, because
four loop constructions disagree about their own bounds: this repo's lattice
includes both, CppRobotics accumulates while `v <= dw[1]`, goktug97 truncates
an integer division and never reaches its upper bound, and amslabtech walks
`side × side` exactly. `dwa_compare_cpp` prints the counts above the table so
this cannot drift again:

| side | This repo | CppRobotics | goktug97 | amslabtech |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 42 | 36 | 25 | 36 |
| 10 | 110 | 90 | 81 | 100 |
| 20 | 420 | 400 | 361 | 400 |
| 30 | 930 | 870 | 841 | 900 |
| 50 | 2,550 | 2,450 | 2,401 | 2,500 |

So the per-call table above overstates the gap to CppRobotics slightly and
understates the gap to goktug97, and the honest comparison is per trajectory:

| side | This repo | CppRobotics | goktug97 | amslabtech |
| ---: | ---: | ---: | ---: | ---: |
| 6 | 0.643 µs | **0.417 µs** | 4.000 µs | 10.472 µs |
| 10 | 0.645 µs | **0.422 µs** | 4.074 µs | 9.820 µs |
| 20 | 0.660 µs | **0.417 µs** | 4.072 µs | 10.293 µs |
| 30 | 0.666 µs | **0.416 µs** | 4.050 µs | 9.777 µs |
| 50 | 0.672 µs | **0.418 µs** | 4.037 µs | 9.725 µs |

Every column is flat across a 61× range in trajectory count — 1.5% for
CppRobotics, 1.9% for goktug97, 4.6% for this repo, 7.7% for amslabtech — which
is the check that these are per-trajectory costs and not per-call overhead
divided by a number. Against another straightforward C++ DWA this repo costs
1.6× per trajectory, steady across the range; goktug97 costs 6.0-6.3× this
repo and amslabtech 14.5-16.3×.

**This repo loses to CppRobotics at every trajectory count, by 1.6× per
trajectory.** The previous table in this file claimed the opposite, 20 to 25 percent
quicker across the range, and that number was wrong for a reason worth
recording: `mine_sweep` in `dwa_compare_cpp.cpp` was still scoring

    heading_gain * (pi - yaw_err) + obstacle_gain * clear + speed_gain * (v / max_vel)

read at the last rollout sample, which is the form `cpp/src/dwa_controller.cpp`
*replaced* when that controller was found crawling at 0.10 to 0.14 m/s against
a 0.46 m/s limit. The controller now accumulates a normalised inflation penalty
per rollout step, truncates at the first sample within `WP_TOL` of the
waypoint, and scores `heading_gain * h_score + speed_gain * s_score -
obstacle_gain * o_cost`. That is more arithmetic per trajectory, not less, and
the honest column is the slower one. The bench had drifted from the controller
it claims to measure — the second time in this repo, after both DWA harnesses
were found reporting trajectory counts the controllers no longer used.

So the useful reading of this table is no longer the CppRobotics row. It is
that a costmap lookup and an obstacle list are different algorithms: this repo
is a constant factor behind another straightforward C++ DWA on the same window,
and one to two orders of magnitude ahead of the two that measure every rollout
point against every obstacle.

**Python** — `bench/dwa_compare.py`

| Trajectories | This repo | [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) | [kmilo7204](https://github.com/kmilo7204/dwa_python) |
| ---: | ---: | ---: | ---: |
| 36 | **0.29 ms** | 2.66 ms | 4.14 ms |
| 100 | **0.35 ms** | 8.55 ms | 11.64 ms |
| 400 | **0.73 ms** | 39.07 ms | 47.06 ms |
| 900 | **1.79 ms** | 90.36 ms | 105.87 ms |
| 2,500 | **4.32 ms** | 260.08 ms | 296.76 ms |

Per-cell spread across the four runs: 5.4% median, 11% worst.

The structure is that every baseline here keeps an explicit obstacle list and
measures each rollout point against every obstacle, while this repo reads one
costmap cell. amslabtech does that per point in C++ with no vectorisation,
which is the 14.5-16.3× per trajectory. goktug97 walks a point cloud per
sample, which is the 6.0-6.3×. PythonRobotics vectorises the comparison over
numpy and kmilo7204 does not, which is why they sit where they do relative to
each other.

It shows up directly as flat scaling in clutter (Python side, 400
trajectories):

| Obstacles | This repo | PythonRobotics | kmilo7204 |
| ---: | ---: | ---: | ---: |
| 20 | 0.74 ms | 35.23 ms | 42.62 ms |
| 100 | 0.74 ms | 41.97 ms | 50.59 ms |
| 500 | 0.74 ms | 79.22 ms | 94.16 ms |
| 2,000 | **0.74 ms** | 334.02 ms | **439.39 ms** |

Flat versus linear, and flat across a hundredfold change in obstacle count
against a 4.9% median spread on the same cells. A costmap has
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

## What each of them does, not just how long it takes

`python3.12 bench/gif_compare.py` and `--cpp` drive every implementation in
closed loop across one obstacle field and draw it. The tables above time one
call; these say where each one goes, which is the half a timing table cannot
show. Run them on python3.12 — the rendering needs matplotlib, and the one in
dist-packages here is built for 3.12 while `python3` is 3.11, so the ms/tick in
these two figures is 3.12 with numpy 1.26.4 and is not the same measurement as
the tables above.

![Python DWA implementations compared](gif/dwa-python.gif)

![C and C++ DWA implementations compared](gif/dwa-cpp.gif)

Held equal for all seven: the field, the plant and its accelerations
(0.50 m/s, 2.0 rad/s, 0.9 m/s², 7.725 rad/s² — the robot this repo drives), the
sampling resolutions (0.02 and 0.04, the controller's own), the 25-step horizon,
and the clearance, which is 0.425 m measured off this repo's own costmap rather
than the 0.470 m it was asked for: the obstacle raster tests cell centres and
the inscribed band is `int(0.22 / 0.05) = 4` cells, so the nearest free cell to
an obstacle centre is 45 mm closer than the analytic radius, and handing the
baselines the analytic one let this repo drive nearer every obstacle than they
were allowed to.

What is *not* equal is the input, because these are not all the same kind of
thing. This repo's controller tracks a path and gets one, from this repo's own
A* across the same field, with its own eight-waypoint lookahead. The six
baselines are goal seekers — their cost is the bearing to a goal and their own
demos drive at one — so they get the goal. The label in each figure says which.
That asymmetry is the comparison and not a flaw in it: half of this repo is a
global planner, and the distance column is where a planner and a controller
together show against a controller alone.

| | given | time | driven | ms/tick | rollouts | outcome |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| this repo (Python) | route | 28.1 s | 13.29 m | 0.59 | 246 | arrived |
| PythonRobotics | goal | 55.0 s | 19.16 m | 24.15 | 234 | arrived |
| kmilo7204 | goal | 71.4 s | 18.70 m | 25.59 | 234 | arrived |
| this repo (C++) | route | 28.3 s | 13.31 m | 0.14 | 246 | arrived |
| CppRobotics | goal | 90.1 s | 3.87 m | 0.27 | 190 | 9.72 m short |
| goktug97 | goal | 90.1 s | 3.20 m | 0.31 | 152 | 10.37 m short |
| amslabtech | goal | 29.5 s | 13.63 m | 0.72 | 234 | arrived |

The route is 13.83 m. The two implementations of this repo's controller agree
to 2 cm and 0.2 s on the same field, which is the cross-check that matters
most here, and the C++ one does it in 0.14 ms against 0.59.

Two of the four C and C++ ones cross about 3 m and then crawl, and the reason
is worth more than the outcome. CppRobotics scores `to_goal_cost + speed_cost
+ ob_cost` with `ob_cost = 1 / min_r` and no gain on it, against a speed cost
of `max_speed - v`. Instrumented at tick 300 it sits 1.14 m from an obstacle
with `ob_cost` 0.876 against a speed cost of 0.480: the entire speed term is
worth less than the clearance it gives up by moving, so crawling is optimal.
goktug97's clearance term is the same reciprocal. `--density --cpp` rules out
clutter — both report the same distance to the centimetre at 6, 10, 14 and 18
obstacles, because the fields are nested and the obstacle that decides it is in
all four.

That is the pathology this repo's own C++ controller was fixed for: its
obstacle term was the raw margin to lethal, up to 253, *added* to a heading
term worth at most pi, so a clear trajectory scored 1265 for clearance against
15.7 for pointing the right way and it maximised room instead of making
progress. The fix — a normalised penalty, capped at 10 and subtracted — is the
comment beside the scoring in `cpp/src/dwa_controller.cpp`. PythonRobotics has
the same `1 / min_r` but ships `to_goal_cost_gain` 0.15 against
`speed_cost_gain` 1.0, and it arrives.

Three things had to be done to the baselines to measure them at all rather
than to measure the harness, each in the comment above the function that does
it:

- **amslabtech** generates every rollout from a zero state and scores against
  `obs_list_`, so its obstacles and goal are in the base frame, where its node
  reads them from a scan. The field is transformed per tick.
- **CppRobotics**' `calc_to_goal_cost` is the angle between the goal and the
  trajectory endpoint measured *from the world origin*, which is a bearing only
  while the robot is at the origin — where its own demo starts. At this field's
  coordinates, (1, 1) to (10.6, 10.6), both lie on one ray from the origin, the
  cost is zero everywhere along the diagonal, and it drifted 26.49 m to finish
  14.48 m away, outside the field. Translated once at the start it was still
  circling at (3.06, 2.71) after 90 s. It is translated per tick.
- **goktug97**'s `planning` keeps a candidate only on `cost < total_cost` from
  an initial `FLT_MAX`, so when every candidate collides it returns
  `bestVelocity` uninitialised. Reading it is undefined; here it came back zero
  and the trail stopped dead with an obstacle 1.04 m away. It gets the same
  turn-in-place fallback the amslabtech trace gives its own node. Its gains are
  a harness choice either way — `dwa.h`'s `Config` is a plain C struct with no
  initialisers and its README documents the fields without values — so its
  clearance gain was swept, with its footprint, over 18 combinations: it never
  arrives at any of them, and the footprint changes nothing. Two steps, because
  the sweep reads the same field file the figures do:

  ```bash
  python3.12 bench/gif_compare.py --field /tmp/field
  ./bench/dwa_compare_cpp --sweep-goktug /tmp/field
  ```

`--seeds` prints the other two fields that were measured. Seed 7 is the one
drawn because all three Python implementations arrive on it; on seed 3
kmilo7204 finishes 7.86 m short and on seed 11 both references do. Picking one
of those would have flattered this repo by a factor of three.

| seed | this repo | PythonRobotics | kmilo7204 |
| ---: | --- | --- | --- |
| 3 | 27.6 s, 13.39 m | 64.9 s, 23.52 m | 7.86 m short |
| 7 | 28.1 s, 13.29 m | 55.0 s, 19.16 m | 71.4 s, 18.70 m |
| 11 | 27.6 s, 13.39 m | 7.16 m short | 9.64 m short |

Four arrangements were measured and rejected before this one, and they are in
the docstring of `gif_compare.py` because each is a way to make this figure
say something untrue. The shortest version: a field that leaves the straight
line to the goal clear gives this repo nothing to avoid and reported 26.5 s
against 33.5 s, which measured the field; a field that blocks the route stalls
a path tracker, correctly, because routing round a blockage is the global
planner's job; and a parked obstacle freezes a stale route's waypoint index, so
all three orbited it at 0.5 m for the rest of the run.

## Global planner vs Nav2

Reference is Table I of Macenski et al., [*Cost-Aware Kinematically Feasible
Planning for Mobile and Surface Robotics*](https://arxiv.org/abs/2401.13078).
`nav2_maps.py` rebuilds their map and query geometry: 10,000 m² random
occupancy maps at 5 cm resolution (2000 × 2000 cells), ~50 m paths.

| Density | This repo, C++ A\* | Smac 2D-A\* | NavFn | Hybrid-A\* | SBPL ARA\* |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | **4.6 ms** | 66.2 ms | 71.1 ms | 39.1 ms | 5,640 ms |
| 15% | **6.1 ms** | 85.6 ms | 66.5 ms | 40.7 ms | 6,587 ms |
| 20% | **14.8 ms** | 88.8 ms | 61.0 ms | 38.8 ms | 6,633 ms |

Read with the caveats. Their CPU (Ryzen 5 5600X) is considerably faster than
the one these came off, which flatters this repo. Against that, Smac 2D-A\* is
cost-aware and returns a smoothed path, and NavFn solves a full navigation
function — both do more per call than a plain octile A\*. The honest claim is
same order of magnitude on equivalent maps, not that it beats Nav2.

## Python vs C++ in this repo

| Map | Python A\* | C++ A\* | Speedup |
| --- | ---: | ---: | ---: |
| 128 × 128 | 7.58 ms | 0.046 ms | 165× |
| 256 × 256 | 117.65 ms | 1.118 ms | 105× |
| 384 × 384 | 904.84 ms | 4.607 ms | 196× |

Not all language: the C++ port also added a closed set, so it expands 21,015
nodes where Python expands 61,631 on the same map. Roughly 3× is algorithmic.

For the DWA rollout, on the same window:

| Window | Trajectories | Python | C++ | Speedup |
| --- | ---: | ---: | ---: | ---: |
| accel-limited | 410 | 0.961 ms | 0.139 ms | 7× |
| full velocity space | 2,626 | 4.595 ms | 1.021 ms | 4× |

The gap shrinks with batch size, because numpy's fixed per-call overhead
amortises away.

"On the same window" is new, and it is the whole point of the row. Both
harnesses kept their own copy of the controller's tuning, and both had drifted
from it: the accelerations were corrected to the plant's 0.90 m/s² and
7.725 rad/s² and neither copy followed, so the row labelled "accel-limited"
was timing 36 trajectories on the Python side against 30 on the C++ side and
reporting the ratio as a per-trajectory speedup. It read 26×. Neither number
described a window the robot searches. `bench_dwa.py` now takes every constant
off the controller by the same AST extraction `rig.py` uses, and
`bench_dwa.cpp` shares the accelerations with `cpp/src/dwa_controller.cpp`;
both report 410.

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
| `dwa_compare.py` | vs PythonRobotics and kmilo7204, in Python |
| `dwa_compare_cpp.cpp`, `baseline_*.cpp` | vs CppRobotics, goktug97 and amslabtech, and `--trace` for the figures |
| `trace.h` | one closed loop and one clamped plant, shared by the four C++ traces |
| `gif_compare.py` | drives all seven across one field and draws it |
| `nav2_maps.py`, `nav2_compare.py` | vs the Nav2 Smac Planner paper |
| `bench_astar.*`, `bench_dwa.*` | Python vs C++ latency |

Measured on an Intel Xeon @ 2.10 GHz, g++ 13.3 `-O2`, Python 3.11, numpy 2.4,
except the two figures in **What each of them does**, which are python3.12 with
numpy 1.26.4 because matplotlib here is built for 3.12. Absolute numbers move
with hardware; the ratios are the point.
