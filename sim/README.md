# Running the stack in Gazebo

Ten recordings of this package driving a TurtleBot 4 through the warehouse
world: five global planners against one controller, five local controllers
against one planner. Every clip is one run of `nav_launch.py` with nothing
stubbed — the same nodes, the same costmaps, the same `/goal_pose` in and
`/cmd_vel_unstamped` out as on hardware.

ROS 2 Jazzy, Gazebo Harmonic 8.15.0, `turtlebot4_gz_bringup`'s warehouse world,
TurtleBot 4 standard. Rendering is software (llvmpipe), which is why the world
runs far slower than real time; each clip is played back at the factor measured
from the simulator's own clock during that capture, so what you see is the
robot moving at the speed it actually moves.

## Running it

```bash
bash sim/sim_up.sh                       # world, robot, clock, map, localiser
bash sim/drive.sh astar dwa my-run 260   # one clip -> sim/gif/my-run.gif
bash sim/all.sh                          # all ten
```

`sim_up.sh` is a one-off; `drive.sh` may be run repeatedly against it and
resets the robot to the warehouse origin each time.

## What the simulator needed before any of this would move

Five things stood between "the package works" and "the robot drives", none of
them in the package. They are worth writing down because each one produced a
symptom that pointed somewhere else entirely.

**The clock had no publisher.** The stock `turtlebot4_gz.launch.py` starts the
Gazebo GUI whether or not you ask for a headless server, and it takes most of
four cores to render a window nobody is looking at, so the server and the robot
spawn are started separately here. What the split leaves behind is the clock
bridge, which lives in `sim.launch.py`: `/clock` ended up with 38 subscribers
and no publisher, every `use_sim_time` node sat frozen at t=0, and nothing
driven by a timer ever fired. `sim_up.sh` starts the bridge itself.

**The global costmap had no map.** `config/costmap_params.yaml` gives the
global costmap `global_frame: map` and a static layer on `/map`, and in
simulation nothing was serving either. TurtleBot 4 ships a warehouse map;
`sim_up.sh` serves it and `localize.py` supplies `map -> odom`. With the robot
standing on the world origin and an identity transform, 335 of 353 laser
returns landed on mapped wall — so the map frame and the simulator's world
frame are the same frame, and the alignment is real rather than assumed.
`check_align.py` is that measurement, and it is worth re-running after any
change to the transform chain.

**The robot was parked on its own dock.** `standard_dock` spawns at
(0.157, 0, 0) facing the robot — 0.157 m in front of it. The robot drove
into it, rode up the ramp at about five degrees of pitch, and stalled with its
wheels turning. Odometry integrated better than ten metres of travel that never
happened, the computed map pose walked off into a part of the warehouse the
robot had never visited, and laser agreement fell from 95% to 2% while the
robot sat still. `sim_up.sh` moves the dock clear.

**The vendor reflex layer overrode every command.** All four Create 3 cliff
sensors latch a CLIFF hazard in this world — `/hazard_detection` carries type 2
and so does every one of `/_internal/cliff_*/event` — although the Gazebo range
sensors under the robot read 0.0157 m against a 0.15 m maximum, which is the
floor, plainly there. With the reflex latched, `motion_control` replaces every
command with a backward escape: a direct 0.25 m/s forward command left the
robot reversing at 0.15 m/s. `reflexes.REFLEX_CLIFF` is rejected at runtime and
hardcoded in `create3_nodes.launch.py`, so `motion_control` is stopped and
`cmd_relay.py` takes its place on the one path that matters: Twist in,
TwistStamped out to the diff drive, linear clamped to the 0.306 m/s that
`motion_control` itself reported as its own `max_speed`. The reflex layer
emulates vendor firmware and is not part of what these planners and controllers
do.

**Ground truth, not a particle filter.** `localize.py` publishes
`map -> odom` as `T_map_base . T_odom_base^-1` with `T_map_base` taken from
`/sim_ground_truth_pose`. A localiser's drift is a second variable in a
measurement that is about planners and controllers, and the dock episode above
is what happens when wheel odometry is trusted alone.

## Two things the exercise found in the package itself

**Every lethal-obstacle threshold was unreachable.** The planners, DWA and MPPI
compare costmap cells against `LETHAL_COST = 253`, which is the raw nav2 0-255
scale — the scale `bench/maps.py` builds its grids on (0 free, 100 inflated,
254 lethal). But they subscribe to `/global_costmap/costmap`, which is a
`nav_msgs/OccupancyGrid`: 0-100, with -1 for unknown. Measured on the live
warehouse grid, the highest value anywhere across all 1,684,044 cells is 100
and **no cell reaches 253**. Nothing was ever lethal.

A\* survived it because it uses cost as a graded traversal penalty and routed
around the expensive cells anyway. Theta\* did not: line of sight is the one
place with a hard `>= lethal` test, so with no cell lethal it shortcut straight
from robot to goal and returned a two-waypoint plan through a wall. Tracing
that segment through the costmap gives a peak of 100 at (0.68, -1.72), which is
the barrier. DWA's forward clearance read `inf` with the robot 1.5 m from it.

The fix converts at the ROS boundary in all thirteen places a costmap is read,
inverting nav2's own forward map (`255 -> -1, 254 -> 100, 253 -> 99, else
c*99/252`). No threshold changed. `bench/` feeds grids straight into
`node.global_data` and never goes through these callbacks, so the benchmark
path is untouched — `bench/test_planners.py` still reports `all checks passed`.

**Theta\*'s paths were too sparse for the controllers that consume them.** With
the walls visible again it routed around the barrier correctly, as four
waypoints. Every local controller here takes its lookahead a fixed number of
*waypoints* ahead, and DWA's own comment states the assumption: "every global
planner in this package emits at costmap resolution, which puts eight waypoints
at ~0.4 m". That is true of A\* (199 waypoints for this route) and SMAC (206),
and false of Theta\*, whose whole point is to keep only the corners. Handed
four, DWA's eight-waypoint lookahead landed on the goal itself and it crept
into the barrier at 0.09 m/s steering at a point 4.36 m away on the far side.

Theta\*'s published path is now resampled to costmap resolution — 204 waypoints
for the same route — rather than widening any lookahead, which would have
changed tuning that was swept against dense paths. The search is untouched:
same segments, sampled along.

## What made the difference to the runs themselves

**The robot starts pointed along its route.** DWA's dynamic window is the
*measured* yaw rate plus or minus `max_dyawrate * dt` = 0.1 rad/s, so it can
only turn as fast as it has already begun turning. Dropped in facing east with
the goal due south it never opened that window: every tick logged
`omega = -0.10`, and the robot curved away at a three-metre radius and drove
five metres in the wrong direction. Starting it facing south is also just the
normal case for a robot about to drive south.

**One publisher on `/cmd_vel_unstamped`, always.** Two controllers, two costmap
pairs and an hour-old manual publisher were once all writing to that topic at
the same time. The stale one was sending 0.25 m/s at 10 Hz and that is what the
robot followed, which looks exactly like a controller that cannot steer.
`clean.sh` runs before every clip and `drive.sh` prints the publisher count, so
a run that starts dirty says so in its own log.

**RViz first, and the costmap gated on data.** Started alongside the nav stack,
RViz competes for the same four cores while it compiles shaders and loads the
robot meshes through software GL, and the global costmap — which has a bounded
wait for `base_link -> map` at activation — loses that race and comes up empty.
The symptom downstream is the planner logging `Cannot plan: global_data is
None` for a whole run. `drive.sh` now waits for `/global_costmap/costmap` to
actually deliver a message before it sends a goal, rather than sleeping and
hoping.

## The files

| | |
|---|---|
| `sim_up.sh` | world, robot, clock bridge, map server, localiser, command relay |
| `sim_down.sh` | tear all of that down again |
| `drive.sh` | one clip: clean, reset, launch, record, encode |
| `all.sh` | the ten clips |
| `clean.sh` | kill every per-run process, including ones this harness did not start |
| `reset.sh` | teleport the robot back to the origin, facing the route |
| `cmd_relay.py` | `/cmd_vel_unstamped` to the diff drive, bypassing the reflex layer |
| `localize.py` | `map -> odom` from ground truth |
| `check_align.py` | score a live scan against the served map |
| `wait_topic.py` | block until a topic delivers, instead of sleeping |
| `watch.py` | the robot's map pose and distance to goal, live |
| `nav.rviz` | one view covering every planner's and controller's own markers |
| `lib.sh` | process helpers |

`clean.sh` and `lib.sh` are more careful about killing processes than they look
like they need to be. A pattern typed on a command line also appears in the
argv of the shell running that command, so an unguarded `pkill` kills its own
caller; that happened twice here, each time surfacing only as exit code 144.
Patterns live in script files and `clean.sh` walks its own ancestry and refuses
to kill anything in it.
