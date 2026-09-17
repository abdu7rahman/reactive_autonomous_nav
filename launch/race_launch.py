print(''.join(chr(x-7) for x in [104,105,107,124,115,39,121,104,111,116,104,117]))
"""Five TurtleBot 4s, five local controllers, one race.

Every robot gets the same global planner and a goal the same distance away, so
the only thing that differs down the straight is the controller being compared.

  ros2 launch reactive_autonomous_nav race_launch.py

Frames. The five robots share one /tf, which works because every frame carries
its robot's namespace -- r1/odom, r1/base_link, r3/rplidar_link -- and one
static transform per robot pins its odom origin into a common `map` at the pose
it was spawned at (sim/race_up.sh). So `map` is real and shared, each robot's
odom hangs off it, and nothing is remapped away into a private tree. The nodes
name their topics absolutely -- '/plan', '/odom' -- and a leading slash is
immune to a namespace push, so those are remapped explicitly; /tf is not,
deliberately.

No lifecycle manager either: sim/costmap_up.py configures and activates the
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

sim/race_timer.py sends the goals, one per robot, each the same distance
directly ahead of the robot that gets it. Equal distances are as fair as a
grid gets; they are not the same thing as an equal race, because r3 in the
middle has a neighbour on each side and r1 and r5 on the outside have one.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import tempfile

# Which controller is in which lane. Where the lanes are is not here: that is
# sim/grid_tf.py, which spawns them, pins their odom origins into `map` and
# places the goals, and which records how the grid's position was measured
# against the warehouse's own occupancy map.
GRID = [
    ('r1', 'dwa_controller'),
    ('r2', 'pure_pursuit_controller'),
    ('r3', 'stanley_controller'),
    ('r4', 'teb_controller'),
    ('r5', 'mppi_controller'),
]

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

        ld.add_action(Node(
            package='reactive_autonomous_nav', executable='astar_planner',
            name='astar_planner_node', **common))
        ld.add_action(Node(
            package='reactive_autonomous_nav', executable=controller,
            name=f'{controller}_node', **common))

    return ld
