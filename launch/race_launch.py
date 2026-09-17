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

Nothing here serves a map or runs a localiser. The costmaps roll with the
robot, the transform into `map` is the spawn pose rather than an estimate of
it, and the only obstacles on the straight are the other four robots.

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

# lane, controller. Lanes are 1.3 m apart: the robot is 0.34 m across and
# carries a 0.30 m inflation, so anything tighter starts the race with every
# robot already inside its neighbour's forbidden zone. The lane x values live
# in sim/race_up.sh, which spawns them, and in sim/race_timer.py, which places
# each goal directly ahead of the robot that gets it.
GRID = [
    ('r1', -2.6, 'dwa_controller'),
    ('r2', -1.3, 'pure_pursuit_controller'),
    ('r3',  0.0, 'stanley_controller'),
    ('r4',  1.3, 'teb_controller'),
    ('r5',  2.6, 'mppi_controller'),
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

    for ns, _lane, controller in GRID:
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
            # on /r1/local_costmap/local_costmap/get_state. It also puts the
            # published topics where the remappings expect them:
            # /r1/local_costmap/costmap, /r1/local_costmap/published_footprint.
            ld.add_action(Node(
                package='nav2_costmap_2d', executable='nav2_costmap_2d',
                name=which, namespace=f'{ns}/{which}', output='screen',
                parameters=[cm_cfg, {'use_sim_time': True}]))

        ld.add_action(Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager',
            name='lifecycle_manager_costmap', namespace=ns, output='screen',
            parameters=[{'use_sim_time': True}, {'autostart': True},
                        # bond_timeout 0 disables the heartbeat, as the
                        # single-robot launch does: with twenty-five nodes on
                        # four cores a missed bond is a scheduling delay, not a
                        # dead node, and the manager's answer to a missed bond
                        # is to tear the costmaps down mid-race.
                        {'bond_timeout': 0.0},
                        # Absolute. The manager itself sits in /<ns>, so a name
                        # without the leading slash resolves to
                        # /r1/r1/local_costmap/... and it waits on a service
                        # nobody offers -- which is what it did.
                        {'node_names': [f'/{ns}/local_costmap/local_costmap',
                                        f'/{ns}/global_costmap/global_costmap']}]))

        ld.add_action(Node(
            package='reactive_autonomous_nav', executable='astar_planner',
            name='astar_planner_node', **common))
        ld.add_action(Node(
            package='reactive_autonomous_nav', executable=controller,
            name=f'{controller}_node', **common))

    return ld
