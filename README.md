# reactive_autonomous_nav

Custom reactive autonomous navigation stack for TurtleBot4, built on ROS2 Jazzy. Implements a modular planner/controller architecture with pluggable global planners and local controllers — all from scratch, no Nav2 BT server.

---

## Architecture

```
                  ┌──────────────────────┐
                  │   Global Planner     │  /goal_pose → /plan
                  │  (A* / Theta* /      │
                  │   SMAC / RRT /       │
                  │   RRT-SMAC Hybrid)   │
                  └──────────┬───────────┘
                             │ /plan
                  ┌──────────▼───────────┐
                  │   Local Controller   │  /plan + /odom → /cmd_vel_unstamped
                  │  (DWA / Pure Pursuit │
                  │   Stanley / TEB /    │
                  │   MPPI)              │
                  └──────────────────────┘
                             │
                  ┌──────────▼───────────┐
                  │  nav2_costmap_2d     │  local + global costmaps
                  │  (lifecycle managed) │
                  └──────────────────────┘
```

**Global Planners** compute a collision-free path from robot pose to goal:

| Planner | Algorithm | Status |
|---|---|---|
| `astar` | A* with octile heuristic | Working — Laplacian smoothing, RViz heat-map |
| `smac` | SMAC Hybrid A* | Working — kinematically feasible, SE2 lattice |
| `theta_star` | Theta* (any-angle A*) | Working — any-angle, exact grid-traversal line of sight |
| `rrt` | RRT | Working — shortcut smoothing, collision-checked goal link |
| `rrt_smac_hybrid` | RRT + SMAC hybrid | Working — kinematic arcs, needs corridors wider than its 0.22 m turning radius |

**Local Controllers** track the global plan reactively:

| Controller | Algorithm | Status |
|---|---|---|
| `dwa` | Dynamic Window Approach | Working — vectorized rollout, HSV trajectory viz |
| `pure_pursuit` | Pure Pursuit | Working — monotonic lookahead, curvature-limited speed |
| `stanley` | Stanley | Working — monotonic reference point |
| `teb` | Timed Elastic Band | Working — sliding band window |
| `mppi` | MPPI | Working — 1000 samples, 2.8 s horizon re-split by the measured tick interval, measured on the node's clock so the split is the same on any host |

---

## On hardware

[![Custom Planners - Tight Space Navigation](https://i.ytimg.com/vi/4fczERUsW1Y/hqdefault.jpg)](https://youtu.be/4fczERUsW1Y)

[**Custom Planners — Tight Space Navigation**](https://youtu.be/4fczERUsW1Y)

Everything else on this page is a simulator or a bench. This is the stack on
a real robot, and it is the one thing here that is a recording rather than a
measurement: no number in any table below is taken from it, and nothing in
it re-runs from `bench/`. It is here for the question the tables cannot
answer, which is whether any of this survives a real lidar, a real floor and
a gap that does not forgive a costmap being 5 cm out.

---

## In simulation

Every planner and every controller above, driving a TurtleBot 4 through
`turtlebot4_gz_bringup`'s warehouse world on ROS 2 Jazzy and Gazebo Harmonic.
One run each, same start and same goal: out of the open floor at the world
origin, around the barrier at y = -1.8, and down the aisle between the racks to
(2.0, -3.2). Orange is the global plan and white is the robot's footprint;
green, the line it actually drove, appears only in the DWA clips, because
dwa_controller is the only one of the five that publishes /driven_path.

Rendering is software, so the world runs far slower than real time. Each clip
is played back at the factor measured from the simulator's own clock across
that capture, so the robot moves at the speed it actually moves.

### Global planners, DWA driving

| A* | Theta* |
|---|---|
| ![A*](sim/gif/planner-astar.gif) | ![Theta*](sim/gif/planner-theta_star.gif) |

| SMAC | RRT |
|---|---|
| ![SMAC](sim/gif/planner-smac.gif) | ![RRT](sim/gif/planner-rrt.gif) |

| RRT-SMAC hybrid | |
|---|---|
| ![RRT-SMAC hybrid](sim/gif/planner-rrt_smac_hybrid.gif) | |

### Local controllers, A* planning

| DWA | Pure Pursuit |
|---|---|
| ![DWA](sim/gif/controller-dwa.gif) | ![Pure Pursuit](sim/gif/controller-pure_pursuit.gif) |

| Stanley | TEB |
|---|---|
| ![Stanley](sim/gif/controller-stanley.gif) | ![TEB](sim/gif/controller-teb.gif) |

| MPPI | |
|---|---|
| ![MPPI](sim/gif/controller-mppi.gif) | |

### Five controllers, one path

Five TurtleBot 4s on a start line, a different local controller in each lane,
and one chicane translated across all five so the lane is not the variable.
On a gated course the planners are not launched at all: `race_timer.py`
publishes the same path, translated per lane, to all five controllers in the
same instant, so what is compared is tracking rather than each planner's luck
with the walls.

![the chicane race](sim/gif/race-chicane.gif)

| lane | controller | finished |
| --- | --- | --- |
| r1 | dwa | 14.0 s |
| r4 | teb | 16.8 s |
| r2 | pure_pursuit | 18.9 s |
| r3 | stanley | 21.2 s |
| r5 | mppi | 23.8 s |

The order is the straight course's order but the gaps are not: first to fourth
spans 7.2 s here against 4.1 s on the straight, and stanley, which tracks the
path most tightly of the five, pays the most for it.

All five reproduce on a host the simulator clocks at half the real-time factor
the first races were run at, but only after two faults that looked like a
missed deadline and were not. MPPI's own log through a failing race reads
`opt=16-40ms` against a 50 ms period with its goal distance still counting
down: the *race timer* was ending the run. Its stall rule compared each
odometry reading against a progress value updated on the same message, so it
asked whether one message had advanced a centimetre — 0.006 m at 0.29 m/s, and
never true — and every race ended at exactly 25 s. Separately the controller
set the timestep its rollouts integrate with from `time.perf_counter()` while
its timer fires on simulated time, so a 50 ms period read as 233 ms; fixing
that took its worst deviation from 0.174 m to 0.114 m.

Both courses, the four other fields — this repo's DWA against nav2's twice,
and the bench's seven implementations split into two — and what each one cost
to get running are in [`sim/README.md`](sim/README.md).

`astar` with `dwa` is the same configuration in both halves, so it is recorded
once and shown twice rather than run twice — two runs of one pair would differ
only by simulator noise, and putting them side by side would invite reading
that noise as a result.

Localisation is the simulator's ground truth rather than a particle filter, and
the Create 3 reflex layer is bypassed: it latches a false cliff in this world
and replaces every command with a backward escape. Neither is a shortcut around
the planners — both are described, with the measurements behind them, in
[`sim/README.md`](sim/README.md), along with the harness and the rest of what
had to be true before the robot would move at all. Running the stack in
simulation is also what turned up the two bugs fixed in `fix(costmap)` and
`fix(theta_star)`, neither of which the offline benchmarks could see.

---

## How it compares

Everything below is reproducible from `bench/` — `bash bench/run.sh` regenerates
every number on this page. Taken on an Intel Xeon @ 2.80 GHz, g++ 13.3 `-O2`,
Python 3.11 with numpy 2.4, except the two figures, which are python3.12 with
numpy 1.26.4 because matplotlib here is built for 3.12. The line above used to
read 2.10 GHz and was not re-checked against `/proc/cpuinfo` when the sweeps
were re-run; it is the ratios that carry the claims, for exactly this reason.

### Local controller vs other DWA implementations

Seven implementations, same dynamic window, same sampling resolution, same
25-step horizon, same field. Baselines are fetched, not vendored:
`bench/fetch_baselines.sh` for the C and C++ ones, and `bench/dwa_compare.py`
pulls the Python ones at run time. Only their plotting is stripped; the planner
functions are theirs.

Every number is the median of four runs in one sequential sweep. One sweep
matters: on this host the same scripts against the same code measured 1.52 to
1.69 times faster earlier the same day, while every trail in the figures below
came out identical to the centimetre. The absolutes drift with the machine and
the ratios do not.

**C and C++**, per trajectory — `bench/dwa_compare_cpp.cpp`

| Trajectories | This repo | [CppRobotics](https://github.com/onlytailei/CppRobotics) | [goktug97](https://github.com/goktug97/DynamicWindowApproach) (C) | [amslabtech](https://github.com/amslabtech/dwa_planner) |
| ---: | ---: | ---: | ---: | ---: |
| 42 | **0.190 µs** | 2.389 µs | 5.760 µs | 15.569 µs |
| 110 | **0.209 µs** | 2.678 µs | 6.216 µs | 15.100 µs |
| 420 | **0.220 µs** | 2.328 µs | 6.227 µs | 15.640 µs |
| 930 | **0.217 µs** | 2.474 µs | 6.354 µs | 15.417 µs |
| 2,550 | **0.201 µs** | 2.519 µs | 6.419 µs | 15.267 µs |

10.6 to 12.8× CppRobotics, 28 to 32× goktug97, 71 to 82× amslabtech, and every
column flat across a 61× range in trajectory count — the check that these are
per-trajectory costs and not per-call overhead divided by a number. Per
trajectory because the four do not evaluate the same number of them from one
window: four loop constructions disagree about their own bounds, 420 against
400, 361 and 400 at side 20, and the binary prints all four counts above its
table.

**Two earlier versions of this table were wrong, in opposite directions.** The
first claimed 20 to 25 percent quicker than CppRobotics: the bench was still
scoring the form `cpp/src/dwa_controller.cpp` *replaced*. Corrected, it claimed
1.6× slower — and that was the field. Every baseline returns from its obstacle
cost on the first rollout point inside its clearance radius, and the harness
left CppRobotics' `robot_radius` at its 1.0 m default against 60 obstacles in a
10 × 10 m square, so it evaluated **1.0 rollout point per trajectory against
this repo's 24.8**, with 100 percent bailing immediately. One field for all four
now, and the work each one does per trajectory is printed every run, because
nothing in a column of milliseconds would have shown it.

**And then the loop itself.** Four changes, none of which alter what the
controller decides: cos and sin of the heading advanced by one rotation per
step rather than recomputed from the angle, a squared arrival test instead of
`hypot`, and the inflation penalty and costmap index scaled by precomputed
reciprocals. 4.6× at 42 trajectories, 5.1× at 420, 4.8× at 2,550.
`mine_pick_check()` runs the plain and optimised forms over 1,944 states and
gets the same chosen command in **all 1,944**, worst score difference
2.6 × 10⁻¹⁴, and refuses to print a table if they ever disagree.

**Python** — `bench/dwa_compare.py`, per call

| Trajectories | This repo | [PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics) | [kmilo7204](https://github.com/kmilo7204/dwa_planner) |
| ---: | ---: | ---: | ---: |
| 36 | **0.47 ms** | 4.17 ms | 6.58 ms |
| 100 | **0.58 ms** | 13.24 ms | 18.24 ms |
| 400 | **1.19 ms** | 59.59 ms | 73.70 ms |
| 900 | **2.87 ms** | 140.91 ms | 166.51 ms |
| 2,500 | **7.29 ms** | 396.58 ms | 463.89 ms |

Per-cell spread: 5.5% median, 22% worst.

The structure behind the gaps is that every baseline keeps an explicit obstacle
list and measures each rollout point against every obstacle, while this repo
reads one costmap cell. amslabtech does that per point in C++ with no
vectorisation, which is the 71 to 82×. goktug97 walks a point cloud per sample,
which is the 28 to 32×. PythonRobotics vectorises the comparison over numpy and
kmilo7204 does not, which is why they sit where they do relative to each other.
It shows up directly as flat scaling in clutter (Python side, 400
trajectories):

| Obstacles | This repo | PythonRobotics | kmilo7204 |
| ---: | ---: | ---: | ---: |
| 20 | 1.17 ms | 53.32 ms | 66.55 ms |
| 100 | 1.17 ms | 65.29 ms | 86.59 ms |
| 500 | 1.17 ms | 129.35 ms | 153.00 ms |
| 2,000 | 1.17 ms | **538.28 ms** | **703.04 ms** |

Flat versus linear, across a hundredfold change in obstacle count against a
10.3% median spread on the same cells. A costmap has to be built and maintained
by something else first, so this is a trade rather than a free win.

### What each of them does, not just how long it takes

A timing table cannot tell a fast implementation from a fast one that drives
badly. `python3.12 bench/gif_compare.py` and `--cpp` drive all seven in closed
loop across one field and draw it.

![Python DWA implementations compared](bench/gif/dwa-python.gif)

![C and C++ DWA implementations compared](bench/gif/dwa-cpp.gif)

Held equal: the field, the plant and its accelerations, the resolutions, the
horizon, and the clearance — 0.425 m, measured off this repo's own costmap
rather than the 0.470 m it was asked for, because the raster and the inscribed
band together leave the nearest free cell 45 mm closer than the analytic radius
and handing the baselines the analytic one let this repo drive nearer every
obstacle than they were allowed to.

What is not equal is the input, and the label in each figure says so: this
repo's controller tracks a path and gets one from this repo's own A*, while the
six baselines are goal seekers and get the goal, because that is what each is
written for. Half of this repo is a global planner, and the distance column is
where a planner and a controller together show against a controller alone — 13.3 m
driven on a 13.83 m route, against 18.7 to 19.2 m for the two Python baselines
that arrive.

The two implementations of this repo's own controller finish within 0.2 s and
2 cm of each other, which is the cross-check that matters most: the C++ port is
the same controller. In these two runs it scores its window in 0.04 ms a tick
against 1.01 for the Python one, on the 246 trajectories the acceleration limit
leaves at this speed; the bench's own figure, on a fixed 410-trajectory window,
is in **C++ Implementations** below. Both trails are identical to the run
before the rollout loop was optimised, which is what says the optimisation
changed the cost and not the controller. Two of the four C and C++ baselines cross about 3 m and
then crawl, and `bench/README.md` has the instrumented reason — an unnormalised
`1 / min_r` clearance term that outweighs their whole speed term at about a
metre from an obstacle, which is the pathology this repo's own C++ controller
was fixed for.

#### The same seven, on the robot

A drawn field is still a drawing. `RACE_FIELD=bench-py` and
`RACE_FIELD=bench-cpp` put these implementations on TurtleBot 4s in Gazebo --
same robot, same costmap settings, same reference path, every lane released in
the same instant, each implementation's own scoring function choosing the
commands. The C and C++ three run `bench/baseline_*.cpp`'s `step_*`, which is
the same per-tick function the figures above loop over, so a clip and a figure
cannot quietly diverge.

![this repo's DWA against the two reference Python implementations](sim/gif/race-chicane-bench-py.gif)

![this repo's C++ DWA against the three C and C++ baselines](sim/gif/race-chicane-bench-cpp.gif)

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

The C and C++ half comes out in the drawn figure's order — amslabtech arrives,
CppRobotics a long way behind it, goktug97 not at all — off a different plant,
different obstacles and a different goal, which is the cross-check.

The Python half does not, and that is the more useful half. Both references
arrive in the drawn field, and in a 1.10 m corridor neither manages a third of
a metre. Eighteen blocks scattered in a 12 m square leave room to swing wide of
everything; a chicane does not, and `calc_obstacle_cost` returning an
unnormalised `1 / min_r` at gain 1.0 — against `to_goal_cost_gain` 0.15 and
`speed_cost_gain` 1.0, all three theirs — is worth several times the other two
terms combined once a wall is half a metre away. What each is handed, and the
one arrangement of it that was tried and measured worse, is in
[`sim/README.md`](sim/README.md).

### Global planner vs Nav2

Reference numbers are Table I of Macenski et al., [*Cost-Aware Kinematically
Feasible Planning for Mobile and Surface Robotics*](https://arxiv.org/abs/2401.13078),
which benchmarks the Nav2 Smac Planners against NavFn and SBPL ARA* on
10,000 m² random occupancy maps at 5 cm resolution with "1,000 verified
start-goal pairs with minimum separation of 3m". `bench/nav2_maps.py` rebuilds
that map and query geometry — 2000 × 2000 cells, 100 × 100 m, and the same
1,000 uniform pairs a density, which come out 51.9 to 53.7 m apart against the
52.1 m a uniform pair in a 100 m square averages.

| Obstacle density | This repo, C++ A\* | Nav2 Smac 2D-A\* | Nav2 NavFn | Nav2 Hybrid-A\* |
| ---: | ---: | ---: | ---: | ---: |
| 10% | **12.7 ms** | 66.2 ms | 71.1 ms | 39.1 ms |
| 15% | **19.2 ms** | 85.6 ms | 66.5 ms | 40.7 ms |
| 20% | **25.2 ms** | 88.8 ms | 61.0 ms | 38.8 ms |

Mean over the 1,000 pairs, median of five runs, reproducing to 6.3%, 2.1% and
4.0%. The rows this table carried until now — 7.3, 12.0 and 21.3 — were a
median of eight pairs drawn from a 50 ± 6 m band, which was neither the
paper's sample count, nor its sampling, nor the statistic a 1,000-pair table
reports; the same eight pairs on the same seeded maps printed 9.7, 10.6, 10.6
and 13.1 ms on the 15% row across four runs of unchanged code.

**Read that with the caveats**, and the first one is in the path length rather
than the clock. These paths run 54.8, 55.5 and 57.5 m where Smac 2D-A\*
returns 50.96, 50.45 and 49.65 on the same maps — 7.5 to 15.8% longer, and
widening with density, which is what an 8-connected octile path with no
smoothing does. Their CPU (Ryzen 5 5600X) is also considerably faster than the
one these numbers came off. Nav2's Smac 2D-A\* is cost-aware and returns that
smoothed path, and NavFn solves a full navigation function — both spend time
per call that this one does not. The honest claim is that this planner is in
the same order of magnitude on equivalent maps at a measurably worse path, not
that it beats Nav2.

`python3 bench/nav2_compare.py`

### Against nav2's controllers, in a live graph

`nav2_dwb_controller` and the other nav2 local controllers cannot go in the
tables above: they need a live ROS 2 graph and costmap plugins to run at all, so
any number taken outside one would be measuring the harness. They are raced
instead, on the same robot, the same costmap settings and the same path — see
**The nav2 field** and **The versus field** in [`sim/README.md`](sim/README.md).

![this repo's C++ and Python DWA against nav2's controllers](sim/gif/race-chicane-versus.gif)

| lane | controller | finished |
| --- | --- | --- |
| r1 | dwa-c++ (this repo) | 13.9 s, 5.97 m |
| r2 | dwa-py (this repo) | 14.0 s, 6.01 m |
| r5 | nav2 `RegulatedPurePursuitController` | 14.5 s, 5.87 m |
| r3 | nav2 `DWBLocalPlanner` | did not cross: 5.78 m of 5.80 |
| r4 | nav2 `MPPIController` | did not cross: 2.37 m of 5.80 |

Read the second column before the first: DWB drove the whole course and is not
credited with a finish because it stopped 10 mm short of a line the timer
measures by crossing. Both this race and the nav2 field ran 51 s rather than
the 25 s the race timer's stall rule used to allow, so the two that do not
cross had twice as long and did not use it — and nav2's graceful controller,
which that rule had been cutting off at 5.74 m, finishes the nav2 field in
25.8 s.

This is also where the optimised rollout loop is checked against a robot
rather than a harness: lane r1 went 13.9 s to 13.8 s across the change, which
is inside the spread the first four lanes reproduce to, so the loop got cheaper
and the controller did not move.


## Dependencies

- ROS2 Jazzy
- `nav2_costmap_2d`, `nav2_lifecycle_manager`
- `slam_toolbox`
- `tf2_ros`, `rclpy`, `nav_msgs`, `geometry_msgs`, `visualization_msgs`

Install nav2:
```bash
sudo apt install ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-slam-toolbox
```

---

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select reactive_autonomous_nav
source install/setup.bash
```

---

## Usage

```bash
# Default: A* planner + DWA controller, real robot
ros2 launch reactive_autonomous_nav nav_launch.py

# Simulation (Gazebo / Isaac Sim)
ros2 launch reactive_autonomous_nav nav_launch.py use_sim_time:=true

# Pick any planner + controller combo
ros2 launch reactive_autonomous_nav nav_launch.py \
  use_sim_time:=true \
  planner:=theta_star \
  controller:=mppi

# Available planner values: astar, theta_star, smac, rrt, rrt_smac_hybrid
# Available controller values: dwa, pure_pursuit, stanley, teb, mppi
```

Send a goal from CLI:
```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  '{header: {frame_id: "map"}, pose: {position: {x: 2.0, y: 1.0, z: 0.0}, orientation: {w: 1.0}}}'
```

Visualize in RViz:
```bash
rviz2 -d $(ros2 pkg prefix reactive_autonomous_nav)/share/reactive_autonomous_nav/config/nav_view.rviz
```

---

## Package Structure

```
reactive_autonomous_nav/
├── reactive_autonomous_nav/
│   ├── astar_planner.py          # A* global planner
│   ├── theta_star_planner.py     # Theta* any-angle planner
│   ├── smac_planner.py           # SMAC hybrid A* planner
│   ├── rrt_planner.py            # RRT sampling planner
│   ├── rrt_smac_hybrid_planner.py# RRT-SMAC hybrid planner
│   ├── dwa_controller.py         # DWA local controller
│   ├── pure_pursuit_controller.py# Pure Pursuit controller
│   ├── stanley_controller.py     # Stanley controller
│   ├── teb_controller.py         # TEB controller
│   ├── mppi_controller.py        # MPPI controller
│   └── costmap_manager.py        # Lifecycle costmap activator
├── launch/
│   └── nav_launch.py             # Pluggable launch file
├── config/
│   ├── costmap_params.yaml       # Local + global costmap config
│   ├── slam_params.yaml          # SLAM Toolbox config
│   └── nav_view.rviz             # RViz preset
└── package.xml
```

---

## Performance Notes

Things that can meaningfully speed this up:

- **A\* / Theta\***: Pre-inflating the costmap offline, so the planner sees binary free/occupied rather than re-reading cost per expansion, is the obvious next thing to try. Untested here, so it carries no figure. Also try lowering `PATH_BLOCKED_LOOKAHEAD` if your env is mostly static.
- **DWA**: The bottleneck is trajectory rollout count. Decrease `vel_res` + `yawrate_res` or shrink `predict_time` to reduce samples. Alternatively, move rollout to numpy fully (already partially done) and profile with `cProfile`.
- **MPPI**: Bump sample count only if you have a GPU or vectorized backend. On CPU, keep samples ≤ 512.
- **Costmap**: `update_frequency: 20.0` is high for CPU-only — drop to 10 Hz on real hardware if `/cmd_vel` latency spikes.
- **General**: Run planners and controllers in separate processes (already the case via launch file). Pin them to isolated CPU cores with `chrt` if latency is critical.

---

## Robot

Built and tested on **TurtleBot4** (iRobot Create 3 base + RPlidar A1).

Compatible with the custom **mobile manipulator** simulation (4-wheel differential drive + Hokuyo LiDAR + UR12 arm) running in Gazebo / Isaac Sim.

---

## C++ Implementations

The `cpp/` directory contains a separate ROS2 C++ package (`reactive_nav_cpp`) with native C++ ports of the three working components:

| Component | File |
|---|---|
| A* global planner | `cpp/src/astar_planner.cpp` |
| SMAC hybrid A* planner | `cpp/src/smac_planner.cpp` |
| DWA local controller | `cpp/src/dwa_controller.cpp` |
| Baseline host, for the bench race fields | `cpp/src/baseline_controller.cpp` |

The last of those is not a port of anything here: it hosts CppRobotics,
goktug97 and amslabtech as race lanes by calling `bench/baseline_*.cpp`'s
`step_*`, and it only builds once `bench/fetch_baselines.sh` has fetched them.

Build and run the C++ package:
```bash
# from your ros2_ws root — both packages build together
colcon build --packages-select reactive_nav_cpp
source install/setup.bash

# run C++ A* planner directly
ros2 run reactive_nav_cpp astar_planner

# or C++ DWA controller
ros2 run reactive_nav_cpp dwa_controller
```

The C++ DWA is 15× lower latency than the Python one on the same
410-trajectory acceleration-limited window — 0.0895 ms against 1.360 ms — and
13× on the full 2,626-trajectory velocity space, the gap narrowing with batch
size because numpy's fixed per-call overhead amortises away. It was 7× and 4×
before the rollout loop was optimised, and before `bench_dwa.cpp` turned out
to be a third copy of that loop still scoring the form the controller replaced,
so the row had been comparing Python's current scoring against C++'s old one.
The C++ A\* is 118-239× on the shared maps, of which roughly 3× is algorithmic
rather than language: the port also added a closed set, so it expands 21,015
nodes where Python expands 61,631 on the same map. `bench/run.sh` prints both
tables.

On the same course in Gazebo the two DWA implementations finish within 0.1 s of
each other, because a 6 m course at 0.46 m/s is nowhere near either one's
budget — see **The versus field** above. The per-tick margin is what buys
headroom on a robot with a 20 ms loop, not a faster lap.

**One bug in that port outlived the benchmarks that were supposed to catch
it.** It compared nav2's `/…/costmap` — a `nav_msgs/OccupancyGrid`, 0–100 —
against `LETHAL_COST = 253`, so its rollout's collision test could not fire on
any value a costmap can publish. The benchmarks never saw it because
`bench/maps.py` builds its grids on the raw 0–255 scale and none of the C++
bench paths goes through a ROS callback, so the bug lived exactly in the gap
between them. The Python controller had the same bug once and its
`_costs_from_grid` records it; the port never got that fix. Re-racing the
versus field after fixing it moved the lap by one centimetre — the soft
penalty above `WARN_COST` had been carrying it — so what the bug cost was the
guarantee, not the behaviour. [`sim/README.md`](sim/README.md) has the
measurement and what it did and did not invalidate.

---

## License

MIT — Mohammed Abdul Rahman, Northeastern University Seattle
