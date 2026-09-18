# Running the stack in Gazebo

Twelve recordings of this package driving TurtleBot 4s through the warehouse
world: five global planners against one controller, five local controllers
against one planner, and then all five controllers at once, on five robots,
twice — up a clear six-metre straight, and through a chicane every one of them
is handed the same path through. Every clip is one run of `nav_launch.py` or
`race_launch.py` with nothing stubbed — the same nodes, the same costmaps, the
same `/goal_pose` in and `/cmd_vel_unstamped` out as on hardware.

ROS 2 Jazzy, Gazebo Harmonic 8.15.0, `turtlebot4_gz_bringup`'s warehouse world,
TurtleBot 4 standard. Rendering is software (llvmpipe), which is why the world
runs far slower than real time; each clip is played back at the factor measured
from the simulator's own clock during that capture, so what you see is the
robot moving at the speed it actually moves.

## Running it

```bash
bash sim/sim_up.sh                       # world, robot, clock, map, localiser
bash sim/drive.sh astar dwa my-run 300   # one clip -> sim/gif/my-run.gif
bash sim/all.sh                          # all ten
```

`sim_up.sh` is a one-off; `drive.sh` may be run repeatedly against it and
resets the robot to the warehouse origin each time.

## The runs

Nine runs, one per configuration, all reaching (2.6, -3.2) from the world
origin. astar with dwa is recorded once and shown in both halves, because two
runs of the same pair would differ only by simulator noise and inviting a
comparison between them would be inviting a reader to read noise as a result.

The table is printed by `sim/clip_table.py`, from the batch log and the files
on disk, rather than typed from the terminal. The first version of it was
typed, and was wrong within a day: the clips were re-recorded and the arrivals
moved by as much as forty seconds with nothing in the document knowing.

| clip | arrived | playback | size |
|---|---|---|---|
| planner-astar | 18.1 s | 7.5x | 1.33 MB |
| planner-theta_star | 56.5 s | 7.9x | 4.75 MB |
| planner-smac | 15.9 s | 7.1x | 1.07 MB |
| planner-rrt | 30.5 s | 11.5x | 2.22 MB |
| planner-rrt_smac_hybrid | 26.3 s | 7.1x | 1.74 MB |
| controller-dwa | 18.1 s | 7.5x | 1.33 MB |
| controller-pure_pursuit | 16.3 s | 11.5x | 0.96 MB |
| controller-stanley | 24.7 s | 10.7x | 1.50 MB |
| controller-teb | 16.3 s | 6.6x | 0.96 MB |
| controller-mppi | 18.8 s | 7.0x | 1.14 MB |

Simulated seconds from the goal going out to the controller reporting arrival,
which is the one quantity here that compares between runs -- the playback
factor is what the machine was doing at the time, and it varies by a factor of
two between the quickest and slowest capture of the same six cores. The clips
play at their own factor, so what you watch is the robot at the speed it
actually moved.

These are not a ranking. Each run is one run; the same configuration repeats to
a second or two at best, the goal is the same for all nine, and a planner that
took eight seconds longer to search is carrying that in its number as much as
its controller's driving.

All ten were re-recorded together, which is why every clip now shows the line
the robot actually drove: at the time of the first set `dwa_controller` was the
only one publishing `/driven_path`, so the trail appeared in half the clips and
not the other half for a reason that had nothing to do with the controllers.
The set costs 17 MB against the 5.3 MB it replaced, almost all of it in the two
long runs -- theta_star orbits a shelf corner three times before it recovers,
and rrt takes the long way round -- and the clips are trimmed to their own
arrivals rather than to the capture window, so none of it is a robot parked at
the goal.

## The five-robot race

Five TurtleBot 4s on a start line, a different local controller in each lane,
and the same distance to cover. There are two courses, chosen by
`RACE_COURSE`, and they ask different questions.

```bash
RACE_COURSE=straight race_up.sh && RACE_COURSE=straight race.sh 600
RACE_COURSE=chicane  race_up.sh && RACE_COURSE=chicane  race.sh 600
```

`race_up.sh` is a one-off per race. Odometry starts where a robot was spawned
and nothing in this stack can put it back, so a second race needs a fresh
bringup; `race.sh` refuses to start otherwise and says so. The course is
defined once, in `reactive_autonomous_nav/race_course.py`, and read by the
launch file and by every script here -- the lanes were declared twice at one
point and the copies drifted, which put a robot two metres from where the rest
of the stack believed it was.

### The straight: five nav stacks

Every robot gets its own A*, its own pair of costmaps and a goal six metres
directly ahead, so what is being compared is a whole stack rather than a
tracker.

![the five-robot race](gif/race.gif)

| lane | controller | finished |
|---|---|---|
| r1 | dwa | 13.2 s |
| r4 | teb | 14.9 s |
| r2 | pure_pursuit | 17.3 s |
| r3 | stanley | 17.3 s |
| r5 | mppi | 18.6 s |

Simulated seconds from the grid release to the finish line. The first four are
repeatable to a tenth of a second over six races -- 13.2-13.3, 14.9-15.0,
17.3-17.4, 17.3-17.4 -- which they should be: the straight is clear, the goals
are the same distance away, and nothing in the run is random. MPPI's 18.6 s is
one race, the first in which it finished at all; the five before it are in the
MPPI section below, along with what was wrong.

DWA's 13.2 s predates the velocity-window repair below, which raised its
angular acceleration limit to the plant's own. On a clear straight that fix
changes almost nothing -- there is nothing to steer around -- but the number
has not been re-measured since, and it is the one figure in this document that
belongs to an older build of the controller.

### The chicane: five trackers, one path

Three walls per lane, alternating sides, and **one reference path handed to all
five controllers**. That second part is the point. Five A* runs on five rolling
costmaps are five different curves -- each robot sees a different slice of the
walls through its own lidar and replans on its own schedule -- so a chicane
raced that way compares the planners' luck as much as the controllers'
tracking. On a gated course the planners are not launched at all and
`race_timer.py` publishes the same path, translated to each lane, to all five
controllers in the same instant.

![the chicane race](gif/race-chicane.gif)

| lane | controller | finished |
|---|---|---|
| r1 | dwa | 14.0 s |
| r4 | teb | 16.7 s |
| r2 | pure_pursuit | 18.9 s |
| r3 | stanley | 21.2 s |
| r5 | mppi | 24.8 s |

Simulated seconds from the release. The first four reproduce across five races
to a tenth of a second -- 13.9-14.0, 16.7-16.8, 18.9-19.0, 21.1-21.2 -- which
they should, because every robot is handed the same path and nothing in the run
is random. The order is the straight's order, but the gaps are not: first to
fourth spans 7.2 s here against 4.1 s on the straight, and stanley, which
tracks the path most tightly of the five, pays the most for it.

Every controller's name is drawn over its own robot in the colour of its own
trail, because a 560 px clip has no room for a legend and five coloured lines
without one are five coloured lines. Each label lives in its own robot's
`base_link` frame, so the name follows the robot and cannot end up beside the
wrong trail, and the RViz config is generated (`race_rviz.py`) from the same
module the colours come from: five near-identical display blocks with the
colours typed five times each is exactly the sort of file that ends up with a
legend disagreeing with what it labels.

**The course is measured, not drawn.** A lane has to hold the robot plus 0.30 m
of inflation on each side, so a wall blocking the middle of a 1.7 m lane can be
at most 0.76 m wide; the walls are 0.60 m. Each wall is offset 0.45 m from its
lane's centre, and the corridor beside it runs from that wall's edge to the
edge of the next lane's wall -- 1.10 m, of which the middle 0.50 m carries no
inflated cost. The path takes the centre of that 0.50 m band, which is 0.40 m
off the lane centre, and `race_course.gate_corridor()` computes it from the
wall geometry rather than carrying 0.40 as a constant.

The outermost lanes get a wall each where their missing neighbour would have
put one. Without those three the middle three robots squeeze through 1.10 m
while r1 and r5 find one gate apiece with nothing on the far side, which is a
wider gate -- measured, twelve corridors at 1.10 m and three open. The point of
translating one chicane across five lanes is that the lane must not be the
variable, and `race_path.py --check` is what asserts it: it measures how far
the path could be pushed each way at every gate and fails if any lane's six
numbers differ from the rest.

### The path was driven before the simulator saw it

`bench/test_chicane.py` builds the lane as a costmap from the course's own wall
geometry and drives all five controllers down the reference path against the
same unicycle plant the rest of the bench uses. It takes about ninety seconds
and it is what chose the path's shape:

| controller | max deviation from the reference | closest approach to a wall |
|---|---|---|
| stanley | 0.045 m | 0.521 m |
| teb | 0.083 m | 0.486 m |
| pure_pursuit | 0.093 m | 0.507 m |
| mppi | 0.119 m | 0.462 m |
| dwa | 0.171 m | 0.420 m |

The reference itself clears the walls by 0.549 m, so a tracker has 0.249 m
before it is in the inflated band and 0.329 m before its footprint is in a
wall. Every one of the five stays inside that, and the spread between them --
stanley holding the line to 45 mm, DWA cutting nearly four times as much -- is
the difference the clip exists to show.

MPPI is the exception: its 24.8 s is one race, and the four before it are in
the section below along with what was wrong.

### The nav2 field

The same chicane, with nav2's own controller plugins in three of the five
lanes. `RACE_FIELD` chooses who races; the course, the path, the plant and the
walls are identical, so the only thing that differs between lanes is the
controller.

```bash
RACE_COURSE=chicane RACE_FIELD=nav2 race_up.sh && \
RACE_COURSE=chicane RACE_FIELD=nav2 race.sh 900
```

![this repo's DWA against nav2's controllers](gif/race-chicane-nav2.gif)

| lane | controller | finished |
|---|---|---|
| r1 | dwa (this repo) | 14.0 s, 6.00 m |
| r4 | nav2 pursuit (`RegulatedPurePursuitController`) | 14.5 s, 5.87 m |
| r2 | nav2 dwb (`DWBLocalPlanner`) | did not cross: 5.79 m of 5.80 |
| r5 | nav2 graceful (`GracefulController`) | did not cross: 5.74 m of 5.80 |
| r3 | nav2 mppi (`MPPIController`) | did not cross: 2.35 m of 5.80 |

Read the second column before the first. DWB drove the whole course and is not
credited with a finish because it stopped 10 mm short of a line the timer
measures by crossing, and graceful 60 mm short; calling either a failure would
be reading the table wrong. On the distance that matters they are level with
the two that crossed. nav2's MPPI stopping at 2.35 m is the one real
difference, and it is the same 5.80 m course this repo's own MPPI took 24.8 s
to finish.

Getting the three of them to race at all took four things that are not about
control:

- **The costmap is a sibling node, not a child.** `controller_server` hosts its
  own `local_costmap` as a top-level node at `/<ns>/local_costmap/local_costmap`,
  so its parameters have to sit beside `controller_server:` in the YAML rather
  than nested inside it. Nested, they are silently ignored and the defaults
  apply. `config/race_nav2_params.yaml` mirrors the race costmap exactly: 6 × 6 m
  rolling, 0.05 m, 0.30 m inflation, 0.22 m robot.
- **DWB's `max_vel_x` defaults to 0.0.** All four plugins are given the plant's
  own limits, each in its own parameter spelling, from the same four constants
  the rest of the race uses.
- **The critic list is per-plugin.** One shared template with DWB's critics in
  it makes nav2's MPPI refuse to configure -- it rejects
  `mppi::critics::RotateToGoal` -- so the extra block is substituted per lane.
- **`SimpleProgressChecker` aborts a controller slowing into its goal.** Its
  `required_movement_radius` is 0.5 m and DWB was aborted 1 cm from the line
  for not covering it; `movement_time_allowance` is 30.0 s here.

nav2's controllers take a path only through the `FollowPath` action on
`/<ns>/follow_path`, never a topic, so `race_timer.py` dispatches per lane:
the action for a nav2 lane, `/plan` for this package's controllers, and
`/goal_pose` for a lane running a whole stack.

### The versus field

Both implementations of this repo's DWA against nav2's, in one race.

```bash
RACE_COURSE=chicane RACE_FIELD=versus race_up.sh && \
RACE_COURSE=chicane RACE_FIELD=versus race.sh 900
```

![the C++ and Python DWA against nav2's controllers](gif/race-chicane-versus.gif)

| lane | controller | finished |
|---|---|---|
| r1 | dwa-c++ | 13.9 s, 5.98 m |
| r2 | dwa-py | 13.9 s, 6.05 m |
| r5 | nav2-pursuit | 14.5 s, 5.88 m |
| r3 | nav2-dwb | did not cross: 5.79 m of 5.80 |
| r4 | nav2-mppi | did not cross: 2.35 m of 5.80 |

The C++ and Python controllers finish the same course within a tenth of a
second of each other, which is the answer to the question the lane exists to
ask: the port is the same controller, not a faster one. Where the C++ is ahead
is per tick, not per race -- 0.139 ms against 0.961 ms on the same
410-trajectory window, in `bench/README.md` -- and on a 6 m course at 0.46 m/s
neither is anywhere near its budget, so the difference does not show. It would
on a robot with a 20 ms loop and a 2,500-trajectory window.

**Four recorded versus races measured a fight, not a controller.** `clean.sh`
swept `lib/reactive_autonomous_nav/` and nothing else, so every relaunch left
the previous run's C++ `dwa_controller` alive on r1 and two processes published
to one `cmd_vel`. The lane crawled, and three separate fixes to the C++
controller were judged not to have worked when what they were being judged
against was a second copy of themselves. `clean.sh` now sweeps
`lib/reactive_nav_cpp/` and `nav2_controller/controller_server` too, and
`race.sh` runs it before every relaunch rather than after.

Two shapes were tried and rejected, both because the bench measured them worse.
Dropping the smoothing's data weight rounds the corner from a 0.25 m radius to
0.92 m, which sounds better and is not: the rounding eats 0.056 m of the swing,
so the reference clears by 0.457 m instead of 0.549 m, and the spread between
best and worst tracker narrows from 0.045-0.171 m to 0.039-0.128 m. A softer
curve is an easier curve. And removing the straight through each gate, so the
path is a single apex, costs 0.084 m of clearance for the same reason -- the
smoothed path never reaches the corridor's centre, peaking at 0.375 m of a
possible 0.400 -- and brings DWA's closest approach from 0.420 m to 0.361 m.
Lengthening it past 0.35 m buys a further millimetre of clearance and costs
0.06 m of DWA's tracking.

### MPPI was planning at 20 Hz and running at 8

This is the second time the clock has been MPPI's problem and they are not the
same fault: the section further down is about a single tick costing 337 ms,
which was fixed by not allocating a 1.9 MB temporary fifty-seven times per
tick. This one is about what the controller believes that tick's interval to
be.

The first four chicane races ended with MPPI wedged against a wall: it turned
77 degrees off a course it was meant to follow at 54, crept sideways until its
footprint was 0.13 m from a wall, and braked there for the rest of the race.
Twice, at 0.51 m and 0.52 m of 5.80 -- the same centimetre, so a fault rather
than bad luck.

The bench could not reproduce it. `test_chicane.py` drove the same controller
down the same path against the same plant and measured 0.12 m of deviation,
and three hypotheses tested there all came back clean:

| what was tested | MPPI's max deviation |
|---|---|
| nothing (the reference case) | 0.127 m |
| a plant delivering at most 0.30 / 0.20 / 0.15 m/s | 0.130 / 0.102 / 0.113 m |
| a costmap knowing only what the lidar has swept, 45% unknown | 0.126 m |
| a neighbouring robot 1.7 m to one side | 0.121 m |

What the bench cannot model is a clock, and that is where the fault was. MPPI's
timer is 50 ms. In the five-robot race its optimisation alone took 53 ms at
best, 70 at the median and 113 at the worst, so every single tick overran and
the loop ran at about 8 Hz -- while every rollout, every acceleration clamp and
the warm-start shift assumed 20. A command planned to be replaced after 50 ms
was held for 125, which is two and a half times the turn that was planned. On a
straight that is invisible, which is why the same controller finished the
straight race and reached its goal in its own single-robot clip, with a median
tick of 54 ms and nothing to diverge from.

The fix is that the model's timestep is the interval the loop is achieving, not
the one it asked for, and that the horizon is the quantity held fixed: 2.8 s,
split into however many steps that interval implies, rather than always 56.
Both halves were measured separately.

| | MPPI's progress | median tick |
|---|---|---|
| as it was | 0.51 m, 0.52 m of 5.80 | 70 ms |
| measured dt, 56 steps regardless | 2.11 m | 66 ms |
| measured dt, 2.8 s horizon | finished, 24.8 s | 19 ms |

Holding the step count is the half-fix: it gets the timestep right and stretches
the horizon from 2.8 s to 7, and the tick stays too expensive to meet the
period. Splitting the horizon instead makes the whole thing self-correcting --
a longer interval means fewer, longer steps, the rollout cost falls with the
step count, and the loop gets closer to meeting its period -- and the
optimisation dropped to 19 ms, comfortably inside it. `MIN_STEPS = 12` bounds
how far that can go: at twelve steps over 2.8 s each step is 0.12 m of travel,
still finer than the 0.30 m inflation band the rollouts are scored against, so
a wall cannot fall between two steps.

Nothing here changes what the controller optimises. Williams' weighting, the
critics and their weights are untouched; what changed is that the model now
describes the machine it is running on.

**The race is timed against the robots' own odometry, not their logs.** Each
robot's odom frame is created where it spawned with x along the heading it
spawned in, so odom x is distance up the straight -- the same quantity for all
five, with no transform and no localiser in between. `race_timer.py` sends all
five goals from one process, because a grid released over four wall seconds at
a real-time factor near 0.3 is more than a second of simulated head start,
which is a tenth of the race.

The finish line is at 5.80 m for a 6.0 m goal: the goal distance, less the
0.15 m tolerance every controller in this package stops inside, less one 0.05 m
costmap cell, because the controllers measure their arrival against the
planner's last waypoint and that waypoint is a cell centre rather than the goal
pose. A line at 6.00 - 0.15 was measured to be 0.02 m too far -- pure_pursuit
stopped at 5.84 and stanley at 5.83, both correctly at their own goals, and
both were recorded as never having finished.

### Where the grid sits, and why it moved

The first straight ran from y = 1 to y = 7 with the lanes centred on x = 0, and
there is a wall across all five of those lanes at y = 7.9 -- 0.9 m past the
finish. MPPI rolls 56 steps of 0.05 s, so 2.8 s, which at its ceiling is about
1.3 m of lookahead: the wall was inside its horizon for the whole last second
of the race. `shelf_big_3` at (3.5, 9.5) sat beside lane 5 and beside no other,
so the straight was not the same race for all five.

Scanning the warehouse's own occupancy map for the longest corridor with no
occupied and no unknown cell across the full 6.4 m the five lanes and a robot
radius need: lanes centred on x = 0 give 8.6 m of it, lanes centred on x = -3
give 9.4 m. The grid runs from y = -1 to y = 5 on the second, which leaves
2.8 m of run-off past the finish -- twice MPPI's horizon -- and 0.6 m behind
the grid. `race_course.py` holds those numbers now, and the launch file,
`grid_tf.py`, `race_up.sh`, `race_path.py`, `race_rviz.py` and `race_timer.py`
all read them from it rather than keeping copies.

It made no difference to MPPI, whose trouble was elsewhere -- see below -- but
the straight is now the same race for all five, which it was not.

### Not the stock five-robot bringup

`turtlebot4_spawn.launch.py` five times over does not fit on four cores. It was
tried first and the logs say what happened: r1 came up, r2's model loaded into
the physics engine but its `controller_manager` sat on `Waiting for data on
'/r2/robot_description'` for eleven minutes, and r3, r4 and r5 never reached
the engine at all while the server printed `SceneBroadcaster: Timed out waiting
for state`. Each stock robot is about forty-five nodes -- hazard vectors, IR
vectors, a UI manager, a kidnap estimator, a 1000 Hz ros2_control loop -- and
twelve gpu_lidars. Five of those is 225 nodes and 60 raycast sensors, and
FastDDS ran out of shared-memory ports underneath it.

`race_robot.py` emits the same robot with the parts a controller comparison
uses. Same description, same meshes, same masses, same wheel geometry; Gazebo's
own DiffDrive in place of ros2_control, carrying the numbers the Create 3
controller was configured with (0.233 m wheel separation, 0.03575 m radius,
0.46 m/s and 1.9 rad/s ceilings, 0.9 m/s² and 7.725 rad/s² acceleration
limits), so the dynamics are identical and there is no `robot_description`
handshake to lose. One lidar instead of twelve: the other eleven are Create 3
reflex inputs, and that reflex layer is also what latched a false CLIFF in this
world and had to be killed on every single-robot run, so removing the sensors
removes the problem at its root rather than killing the node that acts on it.
Four processes per robot rather than forty-five, and the real-time factor with
five robots measured 0.11 to 0.29 -- the same range a single stock robot ran at.

### Four things the five-robot graph broke that one robot did not

**The shared-memory transport ran out of ports.** Past about fifty
participants, every new one logged `Failed init_port fastrtps_port7000:
open_and_lock_file failed` and discovery stopped completing: five costmap
lifecycle managers sat on `Waiting for service
/r1/local_costmap/local_costmap/get_state` for eight minutes while `ros2
service list` listed that exact service, and a direct `ros2 service call` to it
timed out. /dev/shm was 93 MB used of 16 GB and the file limit was 20000, so it
was the transport's own port table rather than anything the machine ran out of.
`fastdds_udp.xml` turns the shared-memory transport off; loopback UDP costs
throughput that nothing here is near using, the largest message in the graph
being a 400x400 costmap at 5 Hz.

**/tf_static did not deliver.** The five `map -> <ns>/odom` pins started life as
five `static_transform_publisher` processes, which together with five
`robot_state_publisher` trees and five lidar identities put fifteen
transient-local publishers on one topic. A freshly started listener received
some of the latched samples and not others -- measured with nothing but tf2
involved, `tf2_echo map r1/base_link` resolved and r2 through r5 all reported
`Tf has two or more unconnected trees` -- and downstream that read as nav2
costmaps failing to activate with `Invalid frame ID "map" passed to
canTransform`, one to four robots per attempt, different ones each time.
`grid_tf.py` publishes all five from one node on `/tf` at 10 Hz instead, which
is where a localiser publishes `map -> odom` anyway. The lidar identities are
gone too: `race_robot.py` sets `gz_frame_id` and the scan was measured arriving
stamped `r3/rplidar_link`.

**nav2's lifecycle manager aborts permanently.** It brings its whole list up at
once and treats one failure as final -- `Failed to bring up all requested
nodes. Aborting bringup`, no retry -- so one robot losing a service call costs
the entire race. With five managers and ten costmaps the failure was not a
timeout waiting for anything, it was `async_send_request failed`, the request
never leaving the client. Three consecutive bringups were thrown away that way,
reporting two, then four, then four of five pairs active. `lifecycle_up.py`
configures and activates the ten costmaps itself, one at a time and retrying
each transition, and reports which of the ten are active. The first race after
that reported 10/10 on the first attempt.

**The costmap node's namespace is not the robot's.** `nav2_costmap_2d` puts its
lifecycle node inside a sub-namespace of its own name, so a node given
`namespace='/r1'` comes up as `/r1/local_costmap` and anything driving
`/r1/local_costmap/local_costmap` waits forever. It is also what puts the
published topics where the remappings expect them.

### MPPI was six times slower than its own control period

For the first five races MPPI was the only controller that did not finish, and
it failed the same way every time: somewhere between 3.4 m and 5.3 m of the six
it held a saturated yaw rate at a near-zero speed and spun on the spot until
the clock ran out. One run held `v=-0.02 w=-1.85` against a 1.9 rad/s limit for
the rest of the race with its goal 2.5 m ahead.

Finding it took two changes that are worth keeping on their own.

**All five control loops had branches that returned with no command and no
log** -- no transform, no lookahead point, a path or a band with fewer than two
poses, no feasible time allocation. MPPI's entire log for its first failed race
was three lines: the signature, `ready`, and `New path: 121 waypoints`. Each
branch now brakes and says why, and MPPI prints the same per-tick line DWA
does, with the robot's map pose, its yaw and the optimiser's wall cost in it.

That line said it immediately: `opt=250-660ms` on a timer set to 50 ms, with
the pose frozen across two or three consecutive ticks and yaw jumping a radian
between them. The control loop was hogging its own single-threaded executor, so
its transform listener only ran between ticks and it was steering on a pose up
to half a second old. That is why it held while the robot went straight -- a
stale pose is nearly right -- and broke down the moment a correction was
needed.

Profiled against `bench/rig.py` at K=1000, T=56 and a 121-waypoint path, two
functions were 97% of a 337 ms tick, and neither for a reason to do with
arithmetic:

| | before | after |
|---|---|---|
| `_path_angle_cost` | 216.4 ms | 18.8 ms |
| `_sample_controls` | 109.5 ms | 6.9 ms |
| whole tick | 337.2 ms | 36.8 ms |

`_path_angle_cost` built a (1000, 121, 2) difference tensor for each of 57
timesteps -- 110 MB of allocation churn for the 28 Mflops it actually needs. It
uses the squared-distance identity in one matrix product per timestep now,
dropping the term that is constant across waypoints and so cannot change an
argmin; checked against the old function on a curved path and 400 rollouts, the
two arrays are identical. `_sample_controls` ran the noise recursion as 55,000
interpreted iterations, one per sample per timestep; it is 56 numpy operations
on (K,) arrays now, for the same draw. `bench/test_planners.py` reports the same
step counts and path lengths to inside MPPI's own run-to-run spread.

Under race load the tick now measures 29 ms minimum, 62 ms median, 105 ms at the
ninetieth percentile and 191 ms worst, against 50 ms of simulated time -- which
at the measured real-time factor is about 190 ms of wall clock. MPPI finished
the next race it ran, on its lane, decelerating into its goal: `v=0.04 w=0.02
at=(-0.42,4.82) yaw=1.59` against a lane at x = -0.40 and a path heading of
1.5708.

**One hypothesis was tested and rejected** before the profile was taken, and it
is worth recording because it looked right. The noise sampler's AR(1)
coefficient was 0.015, which is nav2's default for a parameter called `gamma`
that is not this one -- nav2's `gamma` is the control-cost coefficient in
`updateControlSequence`, "a trade-off between smoothness (high) and low energy
(low)" -- so the name came across and the meaning did not. At 0.015 the noise is
white over a 2.8 s horizon and every rollout is a near-copy of the warm-started
baseline, which would explain a weighted mean that cannot leave its own warm
start. Raising it to 0.9, for a 0.475 s correlation time, measured worse and
fixed nothing: rooms-200 went 654 to 671 steps and 20.81 to 21.06 m against a
run-to-run spread of about four steps, and the race still stalled, 4.07 m before
and 3.57 m after. Reverted, with the numbers in the comment beside it. The
parameter keeps its accurate name.

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

## Three things the exercise found in the package itself

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

**TEB told nobody it had arrived.** teb_controller published its status on
/teb_status while every planner subscribes to /dwa_status -- which dwa,
pure_pursuit, stanley and mppi all publish on, the name being historical rather
than DWA-specific. So no planner ever heard TEB reach the goal, the planner's
"stopping replanning" never fired, and it went on issuing a fresh path from the
robot's position for the rest of the run. It is visible in a recording: the
plan redraws after the robot has already stopped, and the clip compresses to
eight times the size of the same route under any other controller.

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
| `../reactive_autonomous_nav/race_course.py` | the courses themselves: lanes, walls, colours, who is in which lane |
| `clean.sh` | kill every per-run process, including ones this harness did not start |
| `reset.sh` | teleport the robot back to the origin, facing the route |
| `cmd_relay.py` | `/cmd_vel_unstamped` to the diff drive, bypassing the reflex layer |
| `localize.py` | `map -> odom` from ground truth |
| `check_align.py` | score a live scan against the served map |
| `wait_topic.py` | block until a topic delivers, instead of sleeping |
| `wait_tf.py` | block until the transform chain settles and stays settled |
| `clock_now.py` | read the simulator's clock, with a real discovery window |
| `lifecycle_up.py` | configure and activate lifecycle nodes, one at a time |
| `watch.py` | the robot's map pose and distance to goal, live |
| `reencode.sh` | rebuild a gif from a capture already on disk |
| `stopbatch.sh` | stop a batch without touching the world |
| `stopall.sh` | stop the chained bringup-and-batch runner and everything under it |
| `nav.rviz` | one view covering every planner's and controller's own markers |
| `lib.sh` | process helpers, and the Fast DDS profile every process picks up |
| `race_up.sh` | the warehouse, five robots, the clock bridge and the odom pins |
| `race.sh` | the race: costmaps, planners, controllers, RViz, capture, encode |
| `stoprace.sh` | stop a race and everything it started, leaving the world up |
| `race_robot.py` | the TurtleBot 4 description, stripped to what a race uses |
| `grid_tf.py` | publishes the course: odom pins, the chicane, the name labels |
| `race_timer.py` | releases the grid and times it against the robots' odometry |
| `race_path.py` | the chicane's one reference path, and the check that it is clear |
| `race_rviz.py` | generates the race view: five robots, one colour and one label each |
| `send_goal.py` | sends a goal and waits for the plan that proves it arrived |
| `clip_table.py` | prints the ten-clip table from the batch log and the files |
| `fastdds_udp.xml` | UDP only, because shared memory ran out of ports |

### Three things this harness is careful about, and why

**Nothing reads the `ros2` command line for anything that matters.** The CLI
gives discovery about a second, and on this machine under load with forty-odd
participants that is not enough: `ros2 node list` returned 0 nodes in a graph
where a 30-second rclpy subscription found `/scan`, `/odom` and `/clock` all
delivering, and `ros2 lifecycle get /map_server` answered `Node not found` for
a process that was alive and logging normally. With the CLI daemon running it
answers from a stale cache instead -- ghosts of race robots dead for hours, two
nodes claiming the same name, 20 nodes where 45 were running. `sim_up.sh` drove
the map server's lifecycle through that CLI with both calls sent to `/dev/null`
and printed `map served` either way, and the cost was batches recorded against
a world whose global costmap had no map in it. `lifecycle_up.py` and
`clock_now.py` are rclpy nodes with real discovery windows.

**Every gate has to be able to fail, and it has to be able to pass.**
`sim_up.sh` printed `core topics present: 2/4` and then `SIMUP`; `all.sh` had
no check at all, so a two-hour batch would start against anything; and the
first version of that check was `if ! sim_ok | tee ...`, which is not a check
either, because a pipeline's status is its last command's and `tee` always
succeeds. Both run `sim_ok` now, which reads the clock and waits for `/scan`,
`/odom` and `/map` to deliver rather than counting entries in a topic list.
The other direction is just as expensive: `race.sh` tested `lifecycle_up.py`'s
output for `costmaps active: 10/10`, a string it has never printed -- it prints
`active: 10/10` -- so the condition could not be true however well the bringup
went, and the next race would have relaunched the whole stack three times and
aborted with ten live costmaps. Both now use the exit status, which is the same
question with no number in it to get wrong.

**A message that was sent is not a message that arrived.** `/goal_pose` is not
latched and a publisher that exits the moment it has written loses the message
if matching has not finished, so the goal used to go out after a wait for a
subscriber to appear in the graph. The graph lied in both directions: `ros2
topic info` reported zero subscribers on runs whose planner demonstrably drove
to the goal, and rclpy's own `count_subscribers` then reported zero through a
full 120-second window on five consecutive runs that all worked, and one on the
single run that did not. That run was a 600-second capture of a stationary
robot, encoded into a 57 kB gif of one still frame, and copied to a second clip
name as well. `send_goal.py` counts nothing: it publishes the goal, waits for
the planner's own `/plan`, and publishes again if none comes. The replacement
run needed two sends, so the loss is reproducible rather than bad luck, and
`drive.sh` now aborts before the capture rather than recording nothing.

`clean.sh`, `sim_down.sh` and the stop scripts are more careful about killing
processes than they look like they need to be. A pattern typed on a command
line also appears in the argv of the shell running that command, so an
unguarded `pkill` kills its own caller; that happened four times here, each
time surfacing only as exit code 144 with whatever edit was queued behind it
lost. The patterns also have to be specific: a bare `dwa_controller` matches
any process whose argv mentions the file, including an editor or a `grep`, and
`turtlebot4_node` does not match `turtlebot4_gz_hmi_node` -- six of those
accumulated across a day of bringups, ages 2.8 to 15.7 hours, one per teardown
that reported leaving nothing running, and six stale participants on domain 0
is enough to make the graph unreadable. Patterns live in script files, carry
the installed path, and `clean.sh` walks its own ancestry and refuses
to kill anything in it.
