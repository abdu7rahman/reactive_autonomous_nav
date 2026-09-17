print(''.join(chr(x-7) for x in [104,105,107,124,115,39,121,104,111,116,104,117]))
"""Five TurtleBot 4s, five local controllers, one race.

What comes up depends on the course, which is RACE_COURSE in the environment
and defined in reactive_autonomous_nav/race_course.py.

  RACE_COURSE=straight  five costmap pairs, five A* planners, five controllers.
                        Every robot gets a goal the same distance away and
                        plans its own way there, so the race is a nav stack
                        against a nav stack.
  RACE_COURSE=chicane   the same without the planners.  sim/race_timer.py
                        publishes one reference path (sim/race_path.py) to all
                        five controllers at once instead, because on a curve
                        the thing worth comparing is the tracking, and five A*
                        runs on five rolling costmaps are five different
                        curves -- different lidar views, different replans,
                        different corners cut.  Leaving the planners out is
                        not only about fairness: A* republishes its plan on a
                        0.5 s timer whenever the robot drifts 0.8 m off it, so
                        a planner still running would overwrite the reference
                        halfway up the course.

  RACE_COURSE=chicane ros2 launch reactive_autonomous_nav race_launch.py

Frames. The five robots share one /tf, which works because every frame carries
its robot's namespace -- r1/odom, r1/base_link, r3/rplidar_link -- and one
static transform per robot pins its odom origin into a common `map` at the pose
it was spawned at (sim/race_up.sh). So `map` is real and shared, each robot's
odom hangs off it, and nothing is remapped away into a private tree. The nodes
name their topics absolutely -- '/plan', '/odom' -- and a leading slash is
immune to a namespace push, so those are remapped explicitly; /tf is not,
deliberately.

No lifecycle manager either: sim/lifecycle_up.py configures and activates the
ten costmaps itself, one at a time and retrying each transition, and its
header records what nav2_lifecycle_manager did instead -- one lost service
call on one robot aborting the whole bringup, permanently, on most attempts.

Nothing here serves a map or runs a localiser. The costmaps roll with the
robot, the transform into `map` is the spawn pose rather than an estimate of
it, and the only obstacles on the straight are the other four robots.

The five stacks come up together.  Staggering them twenty seconds apart was
tried, on the theory that twenty-five nodes appearing at once starved
discovery; it was the wrong diagnosis and did not help.  What was actually
wrong was /tf_static delivery -- see sim/grid_tf.py -- and it is fixed at the
source.

Equal distances are as fair as a grid gets; they are not the same thing as an
equal race, because r3 in the middle has a neighbour on each side and r1 and r5
on the outside have one.  The chicane's edge walls (race_course.wall_poses)
close half of that gap -- every lane then has a wall on both sides of every
gate -- and sim/race_path.py --check is what measures that they do.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import tempfile

from reactive_autonomous_nav.race_course import CONTROLLER, COURSE, GATES

# Which controller is in which lane, and where the lanes are, both come from
# race_course.py -- the same module sim/grid_tf.py spawns the grid from and
# pins the odom origins from, and which records how the grid's position was
# measured against the warehouse's own occupancy map.  It was declared twice
# and the copies were already drifting.
GRID = [(ns, f'{ctrl}_controller') for ns, ctrl in CONTROLLER.items()]

VIZ = ['/astar_markers', '/astar_explored', '/astar_status', '/replan_request',
       '/driven_path', '/controller_status', '/dwa_trajectories',
       '/dwa_best_traj', '/dwa_goal', '/mppi_trajectories', '/teb_band',
       '/pure_pursuit_lookahead', '/stanley_closest_point']


def remaps(ns):
    """Absolute names the nodes use, pointed at this robot's own."""
    names = ['/plan', '/goal_pose', '/odom', '/cmd_vel_unstamped',
             '/global_costmap/costmap', '/local_costmap/costmap'] + VIZ
    return [(n, f'/{ns}{n}') for n in names]


def costmap_params(pkg, ns):
    """One costmap config per robot, from the template.

    The template carries a marker where the namespace goes; see the header of
    config/race_costmap_params.yaml for why a wildcarded file cannot do this.
    """
    src = os.path.join(pkg, 'config', 'race_costmap_params.yaml')
    with open(src) as f:
        text = f.read().replace('__NS__', ns)
    out = os.path.join(tempfile.gettempdir(), f'race_costmap_{ns}.yaml')
    with open(out, 'w') as f:
        f.write(text)
    return out


def generate_launch_description():
    pkg = get_package_share_directory('reactive_autonomous_nav')
    ld = LaunchDescription()

    for ns, controller in GRID:
        frames = {'map_frame': 'map',
                  'odom_frame': f'{ns}/odom',
                  'base_frame': f'{ns}/base_link'}
        common = dict(namespace=ns, output='screen',
                      parameters=[{'use_sim_time': True}, frames],
                      remappings=remaps(ns))

        cm_cfg = costmap_params(pkg, ns)
        for which in ('local_costmap', 'global_costmap'):
            # namespace is /<ns>/<which>, not /<ns>. nav2_costmap_2d puts its
            # lifecycle node inside a sub-namespace of its own name -- the
            # single-robot launch says namespace='local_costmap',
            # name='local_costmap' for the same reason -- so a node given only
            # /r1 comes up as /r1/local_costmap and the manager waits forever
            # on /r1/local_costmap/local_costmap/get_state, which is the name
            # sim/costmap_up.py drives. It also puts the
            # published topics where the remappings expect them:
            # /r1/local_costmap/costmap, /r1/local_costmap/published_footprint.
            ld.add_action(Node(
                package='nav2_costmap_2d', executable='nav2_costmap_2d',
                name=which, namespace=f'{ns}/{which}', output='screen',
                parameters=[cm_cfg, {'use_sim_time': True}]))

        if not GATES:
            ld.add_action(Node(
                package='reactive_autonomous_nav', executable='astar_planner',
                name='astar_planner_node', **common))
        ld.add_action(Node(
            package='reactive_autonomous_nav', executable=controller,
            name=f'{controller}_node', **common))

    print(f'race_launch: course {COURSE}, {len(GRID)} robots, '
          f'{"controllers only" if GATES else "planner + controller each"}')
    return ld
