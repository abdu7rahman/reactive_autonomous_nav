# reactive_autonomous_nav

Custom reactive autonomous navigation stack for TurtleBot4, built on ROS2 Jazzy. Implements a modular planner/controller architecture with pluggable global planners and local controllers — all from scratch, no Nav2 BT server.

---

## Architecture

```
                  ┌──────────────────────┐
                  │   Global Planner     │  /goal_pose → /global_plan
                  │  (A* / Theta* /      │
                  │   SMAC / RRT /       │
                  │   RRT-SMAC Hybrid)   │
                  └──────────┬───────────┘
                             │ /global_plan
                  ┌──────────▼───────────┐
                  │   Local Controller   │  /global_plan + /odom → /cmd_vel
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
| `mppi` | MPPI | Working — 1000 samples, 56-step horizon |

---

## How it compares

Everything below is reproducible from `bench/`. Numbers were taken on an Intel
Xeon @ 2.10 GHz, g++ 13.3 `-O2`, Python 3.11 with numpy 2.4.

### Local controller vs other DWA implementations

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

### Global planner vs Nav2

Reference numbers are Table I of Macenski et al., [*Cost-Aware Kinematically
Feasible Planning for Mobile and Surface Robotics*](https://arxiv.org/abs/2401.13078),
which benchmarks the Nav2 Smac Planners against NavFn and SBPL ARA* on
10,000 m² random occupancy maps at 5 cm resolution with 1,000 start-goal pairs.
`bench/nav2_maps.py` rebuilds that map and query geometry (2000 × 2000 cells,
100 × 100 m, ~50 m paths) so the timings measure comparable work.

| Obstacle density | This repo, C++ A\* | Nav2 Smac 2D-A\* | Nav2 NavFn | Nav2 Hybrid-A\* |
| ---: | ---: | ---: | ---: | ---: |
| 10% | **5.0 ms** | 66.2 ms | 71.1 ms | 39.1 ms |
| 15% | **6.8 ms** | 85.6 ms | 66.5 ms | 40.7 ms |
| 20% | **15.1 ms** | 88.8 ms | 61.0 ms | 38.8 ms |

**Read that with the caveats.** Their CPU (Ryzen 5 5600X) is considerably faster
than the one these numbers came off, which flatters this repo. Against that,
Nav2's Smac 2D-A\* is cost-aware and returns a smoothed path, and NavFn solves a
full navigation function — both do more work per call than a plain octile A\*.
The honest claim is that this planner is in the same order of magnitude on
equivalent maps, not that it beats Nav2.

`python3 bench/nav2_compare.py`


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

- **A\* / Theta\***: Pre-inflate the costmap offline so the planner sees binary free/occupied — cuts heuristic evaluation time by ~30%. Also try lowering `PATH_BLOCKED_LOOKAHEAD` if your env is mostly static.
- **DWA**: The bottleneck is trajectory rollout count. Decrease `vel_res` + `yawrate_res` or shrink `predict_time` to reduce samples. Alternatively, move rollout to numpy fully (already partially done) and profile with `cProfile`.
- **MPPI**: Bump sample count only if you have a GPU or vectorized backend. On CPU, keep samples ≤ 512.
- **Costmap**: `update_frequency: 20.0` is high for CPU-only — drop to 10 Hz on real hardware if `/cmd_vel` latency spikes.
- **General**: Run planners and controllers in separate processes (already the case via launch file). Pin them to isolated CPU cores with `chrt` if latency is critical.

---

## Robot

Built and tested on **TurtleBot4** (iRobot Create 3 base + RPlidar A1).

Compatible with the custom **mobile manipulator** simulation (4-wheel differential drive + Hokuyo LiDAR + UR12 arm) running in Gazebo / Isaac Sim.

---

---

## C++ Implementations

The `cpp/` directory contains a separate ROS2 C++ package (`reactive_nav_cpp`) with native C++ ports of the three working components:

| Component | File |
|---|---|
| A* global planner | `cpp/src/astar_planner.cpp` |
| SMAC hybrid A* planner | `cpp/src/smac_planner.cpp` |
| DWA local controller | `cpp/src/dwa_controller.cpp` |

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

The C++ versions are faster (~2-3x lower latency on trajectory rollout) and have zero Python GIL overhead on the control loop.

---

## License

MIT — Mohammed Abdul Rahman, Northeastern University Seattle
