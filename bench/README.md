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

Every number below is the median of four full runs taken in one sequential
sweep, with the per-cell spread beside it. One sweep matters: on this host the
same scripts against the same code measured 1.52 to 1.69 times faster earlier
the same day, while every trail in the closed-loop figures came out identical
to the centimetre, so the absolutes drift with the machine and the ratios do
not. Read the ratios.

### The comparison was not comparing

Every baseline here returns from its obstacle cost on the first rollout point
inside its own clearance radius, which is a sound optimisation and was
wrecking the table. The harness left CppRobotics' `robot_radius` at its 1.0 m
default against 60 obstacles in a 10 × 10 m square, so every trajectory it
scored was already in collision at its first point. Measured, per trajectory,
at side 20:

| | rollout points | obstacle checks | bailed on the first hit |
| --- | ---: | ---: | ---: |
| This repo | 24.8 of 25 | 24.8 | 5% |
| CppRobotics | **1.0** | 15.0 | **100%** |

It was timing one point against twenty-five. There is one field for all four
now — the same 0.5 m blocks this repo's costmap rasterises, handed to the
baselines as centres with a radius of half a block, and a start pose kept clear
of them — and `dwa_compare_cpp` prints what each one evaluates above its table
every run, because nothing in a column of milliseconds would have shown this:

| | rollout points | obstacle checks | bailed |
| --- | ---: | ---: | ---: |
| This repo | 25.0 | 25.0 | 0% |
| CppRobotics | 14.0 | 840.0 | 0% |
| goktug97 | 25.0 | 1,500.0 | 0% |
| amslabtech | 25.0 | 1,500.0 | 0% |

CppRobotics evaluates every second rollout point — its own `skip_n = 2` — which
is why it reads 14 against 25.

### Taking the arithmetic out of the rollout loop

Four changes, none of which alter what the controller decides:

- cos and sin of the heading are advanced by one rotation per step instead of
  recomputed from the angle. For a constant *w* the heading turns by *w·dt*
  every step, so *(c, s) ← (c·cw − s·sw, s·cw + c·sw)* with *cw* and *sw*
  computed once per trajectory. That removes two transcendental calls per step,
  fifty per trajectory, about twenty-one thousand a tick.
- the arrival test compares squared distances rather than calling `hypot`.
- the inflation penalty multiplies by a precomputed `1 / (LETHAL − WARN)`.
- the costmap index multiplies by `1 / resolution`.

| Trajectories | Before | After | |
| ---: | ---: | ---: | ---: |
| 42 | 0.037 ms | **0.008 ms** | 4.6× |
| 420 | 0.400 ms | **0.079 ms** | 5.1× |
| 2,550 | 2.392 ms | **0.495 ms** | 4.8× |

`mine_pick_check()` runs both forms over 1,944 start states and lookahead
points and reports the chosen command in each: **identical in all 1,944**,
worst velocity difference 0, worst yaw-rate difference 0, worst score
difference 2.6 × 10⁻¹⁴ — which is the drift the rotation increment actually
costs, and it is a gate, not a note: the binary refuses to print a table if
the two disagree.

`cpp/src/dwa_controller.cpp` had one more thing wrong that the bench could not
see, because the bench does not draw: it built a `LINE_STRIP` of twenty-five
`geometry_msgs/Point` for every kept trajectory and published all of them,
about 420 markers and 10,500 Point constructions a tick, for a picture that is
a grey smear past a few dozen lines. That is exactly what `dwa_controller.py`
was fixed for and carries a comment about. It takes the same `traj_draw` of 48
now, tested before the rollout so the points are never built.

### The table

**C and C++** — `bench/dwa_compare_cpp.cpp`

| Trajectories | This repo | [CppRobotics](https://github.com/onlytailei/CppRobotics) | [goktug97](https://github.com/goktug97/DynamicWindowApproach) (C) | [amslabtech](https://github.com/amslabtech/dwa_planner) |
| ---: | ---: | ---: | ---: | ---: |
| 42 | **0.008 ms** | 0.086 ms | 0.144 ms | 0.560 ms |
| 110 | **0.023 ms** | 0.241 ms | 0.504 ms | 1.510 ms |
| 420 | **0.092 ms** | 0.931 ms | 2.248 ms | 6.256 ms |
| 930 | **0.202 ms** | 2.152 ms | 5.343 ms | 13.875 ms |
| 2,550 | **0.514 ms** | 6.171 ms | 15.412 ms | 38.168 ms |

The four do not evaluate the same number of trajectories from one window,
because four loop constructions disagree about their own bounds: this repo's
lattice includes both, CppRobotics accumulates while `v <= dw[1]`, goktug97
truncates an integer division and never reaches its upper bound, and
amslabtech walks `side × side` exactly. At side 20 that is 420, 400, 361 and
400, printed above the table every run. So per trajectory:

| Trajectories | This repo | CppRobotics | goktug97 | amslabtech |
| ---: | ---: | ---: | ---: | ---: |
| 42 | **0.190 µs** | 2.389 µs | 5.760 µs | 15.569 µs |
| 110 | **0.209 µs** | 2.678 µs | 6.216 µs | 15.100 µs |
| 420 | **0.220 µs** | 2.328 µs | 6.227 µs | 15.640 µs |
| 930 | **0.217 µs** | 2.474 µs | 6.354 µs | 15.417 µs |
| 2,550 | **0.201 µs** | 2.519 µs | 6.419 µs | 15.267 µs |

**10.6 to 12.8× CppRobotics, 28 to 32× goktug97, 71 to 82× amslabtech**, and
every column flat across a 61× range in trajectory count, which is the check
that these are per-trajectory costs and not per-call overhead divided by a
number. Per-cell spread is 2 to 9% at 420 trajectories and above; the 42 and
110 rows reach 25 and 126% because a call there takes 8 to 23 µs and the
`steady_clock` resolution shows.

Two earlier versions of this table are worth recording, because both were
wrong and neither was wrong at random:

1. It claimed this repo **20 to 25 percent quicker** than CppRobotics.
   `mine_sweep` was still scoring the form `cpp/src/dwa_controller.cpp`
   *replaced* — `heading_gain * (pi - yaw_err)` plus a raw clearance margin
   read at the last rollout sample, the pair of mistakes that had that
   controller crawling at 0.10 to 0.14 m/s against a 0.46 m/s limit.
2. Corrected to the current scoring, it claimed this repo **1.6× slower**.
   That was the field above: one rollout point against twenty-five.

Three copies of this loop in this repo have now been found drifted from the
controller they measure — `mine_sweep` here, `bench_dwa.cpp`, and both
harnesses' idea of the dynamic window. The lesson is in the work table: a
timing harness needs a gate on *what it evaluated*, not only on how long it
took.

**Python** — `bench/dwa_compare.py`

| Trajectories | This repo | [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) | [kmilo7204](https://github.com/kmilo7204/dwa_planner) |
| ---: | ---: | ---: | ---: |
| 36 | **0.47 ms** | 4.17 ms | 6.58 ms |
| 100 | **0.58 ms** | 13.24 ms | 18.24 ms |
| 400 | **1.19 ms** | 59.59 ms | 73.70 ms |
| 900 | **2.87 ms** | 140.91 ms | 166.51 ms |
| 2,500 | **7.29 ms** | 396.58 ms | 463.89 ms |

Per-cell spread: 5.5% median, 22% worst.

It shows up directly as flat scaling in clutter (Python side, 400
trajectories):

| Obstacles | This repo | PythonRobotics | kmilo7204 |
| ---: | ---: | ---: | ---: |
| 20 | 1.17 ms | 53.32 ms | 66.55 ms |
| 100 | 1.17 ms | 65.29 ms | 86.59 ms |
| 500 | 1.17 ms | 129.35 ms | 153.00 ms |
| 2,000 | **1.17 ms** | 538.28 ms | **703.04 ms** |

Flat versus linear, and flat across a hundredfold change in obstacle count
against a 10.3% median spread on the same cells. A costmap has
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
| this repo (Python) | route | 28.1 s | 13.29 m | 1.01 | 246 | arrived |
| PythonRobotics | goal | 55.0 s | 19.16 m | 39.01 | 234 | arrived |
| kmilo7204 | goal | 71.4 s | 18.70 m | 42.19 | 234 | arrived |
| this repo (C++) | route | 28.3 s | 13.31 m | **0.04** | 246 | arrived |
| CppRobotics | goal | 90.1 s | 3.87 m | 0.25 | 190 | 9.72 m short |
| goktug97 | goal | 90.1 s | 3.20 m | 0.35 | 152 | 10.37 m short |
| amslabtech | goal | 29.5 s | 13.63 m | 1.10 | 234 | arrived |

The route is 13.83 m. The two implementations of this repo's controller agree
to 2 cm and 0.2 s on the same field, which is the cross-check that matters
most here, and the C++ one does it in 0.04 ms a tick against 1.01. Those two
trails were identical on every seed before and after the rollout loop was
optimised, and identical again on a host running 1.6× slower, which is what
says the optimisation changed the cost and not the controller.

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

Every cell in that table is unchanged from the run before the optimisation and
before the host slowed down. The figures' ms/tick moved; nothing else did.

Four arrangements were measured and rejected before this one, and they are in
the docstring of `gif_compare.py` because each is a way to make this figure
say something untrue. The shortest version: a field that leaves the straight
line to the goal clear gives this repo nothing to avoid and reported 26.5 s
against 33.5 s, which measured the field; a field that blocks the route stalls
a path tracker, correctly, because routing round a blockage is the global
planner's job; and a parked obstacle freezes a stale route's waypoint index, so
all three orbited it at 0.5 m for the rest of the run.

### And the same seven in the simulator

A drawn field is still a drawing: one plant, one obstacle list, one process.
`RACE_FIELD=bench-py` and `RACE_FIELD=bench-cpp` in `sim/` put the same seven
on TurtleBot 4s in Gazebo -- same robot, same nav2_costmap_2d settings, same
reference path, every lane released in the same instant, each implementation's
own scoring function choosing the commands from its own lidar and costmap.

The C and C++ three call `baseline_*.cpp`'s `step_*`, which is the per-tick
function `trace_*` loops over here, so the clip and the figure above cannot
diverge below that call. The Python two are loaded by `baselines_py.py`, which
is the same loader `dwa_compare.py` uses.

| field | controller | finished |
| --- | --- | --- |
| bench-py | dwa-py | 13.9 s, 6.03 m |
| bench-py | dwa-c++ | 13.9 s, 5.99 m |
| bench-py | PythonRobotics | did not cross: 0.32 m of 5.80 |
| bench-py | kmilo7204 | did not cross: 0.32 m of 5.80 |
| bench-cpp | dwa-c++ | 14.0 s, 5.97 m |
| bench-cpp | amslabtech | 23.4 s, 5.87 m |
| bench-cpp | CppRobotics | 96.2 s, 5.80 m |
| bench-cpp | goktug97 | did not cross: 0.09 m of 5.80 |

The C and C++ order is this page's order -- amslabtech arrives, CppRobotics is
far behind it, goktug97 does not get there -- reached through a different
plant, different obstacles and a different goal.

The Python pair is where the two disagree, and the disagreement is the
finding. Both arrive on the drawn field, at seed 7 in 55.0 s over 19.16 m and
71.4 s over 18.70 m, and neither reaches a third of a metre of a 5.80 m
chicane. Their scoring says why: `calc_obstacle_cost` returns an unnormalised
`1 / min_r` at `obstacle_cost_gain` 1.0, against `to_goal_cost_gain` 0.15 and
`speed_cost_gain` 1.0 -- all three their own defaults -- so with a wall half a
metre away the obstacle term is worth several times the whole of the other
two, the command that maximises clearance wins, and standing still maximises
clearance. PythonRobotics then reads `|v| < 0.001` as stuck and commands
`-max_delta_yaw_rate`, which is a spin in place that does not change anything
it is scoring. Eighteen blocks in a 12 m square leave room to swing wide of
all of it; a 1.10 m corridor does not.

This is the same shape as the pathology recorded for the C and C++ side above,
and it is the one this repo's own controller was fixed for.

`sim/README.md` has the clips, the two decisions about what a goal seeker is
handed on a tracking course, and the arrangement of them that measured worse.

## Global planner vs Nav2

Reference is Table I of Macenski et al., [*Cost-Aware Kinematically Feasible
Planning for Mobile and Surface Robotics*](https://arxiv.org/abs/2401.13078).
`nav2_maps.py` rebuilds their experiment rather than something like it: three
10,000 m² random environments at 10/15/20% obstacle density, 5 cm resolution
(2000 × 2000 cells), and the paper's own query rule, which is quoted there
because getting it slightly wrong is what went wrong before — "we generated
1,000 verified start-goal pairs with minimum separation of 3m". Uniform pairs
in a 100 × 100 m square average 52.1 m apart; these three maps measure 51.9,
52.2 and 53.7 m, which is what says the geometry matches.

| Density | This repo, C++ A\* | Smac 2D-A\* | NavFn | Hybrid-A\* | SBPL ARA\* |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10% | **12.7 ms** | 66.2 ms | 71.1 ms | 39.1 ms | 5,640 ms |
| 15% | **19.2 ms** | 85.6 ms | 66.5 ms | 40.7 ms | 6,587 ms |
| 20% | **25.2 ms** | 88.8 ms | 61.0 ms | 38.8 ms | 6,633 ms |

Mean over the 1,000 pairs, median of five runs, and the rows reproduce to
6.3%, 2.1% and 4.0% across those five. The mean is the statistic because a
1,000-pair table is an average, and because it is the steadier one here: the
distribution is heavy enough that at 10% density the per-query median is
6.4 ms against a 12.7 ms mean, and a median over a small sample of that
jumps between neighbouring queries. That is not a hypothetical. The figures
this table carried until now — 7.3, 12.0 and 21.3 ms — were a median of
**eight** pairs drawn from a 50 ± 6 m band, and all three of those choices
were wrong: the count, the sampling and the statistic. The same eight pairs
on the same seeded maps printed 9.7, 10.6, 10.6 and 13.1 ms on the 15% row
across four runs of unchanged code.

And the column that decides whether this is a fair race at all, which the
band sampling had made agree with the paper by construction:

| Density | This repo, C++ A\* | Smac 2D-A\* | NavFn | Hybrid-A\* |
| ---: | ---: | ---: | ---: | ---: |
| 10% | 54.8 m | 50.96 m | 52.60 m | 51.41 m |
| 15% | 55.5 m | 50.45 m | 52.50 m | 51.10 m |
| 20% | 57.5 m | 49.65 m | 52.25 m | 50.78 m |

These paths are 7.5, 10.0 and 15.8% longer than the ones Smac 2D-A\* returns
on the same geometry, and the gap widens with density — which is what an
8-connected octile path with no smoothing does. The planner is not solving
the same problem as well; it is solving an easier version of it faster.

Read the timings with that and the rest of the caveats. Their CPU (Ryzen 5
5600X) is considerably faster than the one these came off, which flatters
this repo. Smac 2D-A\* is cost-aware and smooths, NavFn solves a full
navigation function, and both of those cost time this one does not spend. The
honest claim is same order of magnitude on equivalent maps, at a measurably
worse path.

## Python vs C++ in this repo

| Map | Python A\* | C++ A\* | Speedup |
| --- | ---: | ---: | ---: |
| 128 × 128 | 9.84 ms | 0.080 ms | 123× |
| 256 × 256 | 157.25 ms | 1.329 ms | 118× |
| 384 × 384 | 1,244.02 ms | 5.201 ms | 239× |

Not all language: the C++ port also added a closed set, so it expands 21,015
nodes where Python expands 61,631 on the same map. Roughly 3× is algorithmic.

For the DWA rollout, on the same window:

| Window | Trajectories | Python | C++ | Speedup |
| --- | ---: | ---: | ---: | ---: |
| accel-limited | 410 | 1.360 ms | **0.0895 ms** | 15× |
| full velocity space | 2,626 | 7.607 ms | **0.5843 ms** | 13× |

The gap shrinks with batch size, because numpy's fixed per-call overhead
amortises away. It was 7× and 4× before the rollout loop was optimised, and
before `bench_dwa.cpp` was found to be a *third* copy of the loop still
scoring the form the controller replaced — so that row had been comparing the
Python controller's current scoring against the C++ controller's old one.

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

Measured on an Intel Xeon @ 2.80 GHz, g++ 13.3 `-O2`, Python 3.11, numpy 2.4,
except the two figures in **What each of them does**, which are python3.12 with
numpy 1.26.4 because matplotlib here is built for 3.12.

Absolute numbers move with hardware, and on this host they move without it:
the same scripts against the same code measured 1.52 to 1.69 times faster
earlier the same day, and every trail in the closed-loop figures came out
identical to the centimetre across that change. The clock above moved too --
it read 2.10 GHz until `/proc/cpuinfo` was checked against it rather than
copied forward. So every table above is one sequential sweep rather than a
best-of, and the ratios are the claim.
