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

from reactive_autonomous_nav.race_course import (
    CONTROLLER, COURSE, ENTRANTS, FIELD, GATES)

# Who is in which lane, and where the lanes are, both come from
# race_course.py -- the same module sim/grid_tf.py spawns the grid from and
# pins the odom origins from, and which records how the grid's position was
# measured against the warehouse's own occupancy map.  It was declared twice
# and the copies were already drifting.
#
# A lane is one of two things.  A 'pkg' lane runs one of this package's
# controller nodes against a nav2_costmap_2d local costmap of its own.  A
# 'nav2' lane runs a nav2 controller_server hosting the named plugin, which
# brings its own local costmap with it -- configured, in config/
# race_nav2_params.yaml, to the same 6 x 6 m window and the same 0.30 m
# inflation the package's controllers read, so what differs between lanes is
# the controller and not the costmap.

VIZ = ['/astar_markers', '/astar_explored', '/astar_status', '/replan_request',
       '/driven_path', '/controller_status', '/dwa_trajectories',
       '/dwa_best_traj', '/dwa_goal', '/mppi_trajectories', '/teb_band',
       '/pure_pursuit_lookahead', '/stanley_closest_point']


def remaps(ns):
    """Absolute names the nodes use, pointed at this robot's own."""
    names = ['/plan', '/goal_pose', '/odom', '/cmd_vel_unstamped',
             '/global_costmap/costmap', '/local_costmap/costmap'] + VIZ
    return [(n, f'/{ns}{n}') for n in names]


# What each nav2 plugin has to be told, and nothing more.
#
# The robot's limits first.  They are the plant's own, out of
# irobot_create_control/config/control.yaml by way of sim/race_robot.py, which
# is what Gazebo's DiffDrive is given: 0.46 m/s, 1.9 rad/s, 0.9 m/s^2 and
# 7.725 rad/s^2.  Racing them on their defaults instead was tried first and is
# not a comparison of controllers: DWB's default max_vel_x is 0.0, so it sat
# on the start line for the whole race, and every plugin has a different idea
# of what a default robot is.  This package's own DWA asks for 0.50 and 2.0
# and the plant clamps it to the same 0.46 and 1.9, so all five lanes are held
# to one envelope.
#
# Everything else is the plugin's own default.  DWB's block carries the sample
# counts and granularity nav2 ships in its own DWB example, because they
# describe the search rather than the robot, and its critic list, because DWB
# is the one plugin with no default list -- it refuses to configure with "No
# critics defined for FollowPath".  The list cannot live in the template for
# everyone: nav2's MPPI reads the same `critics` key with an entirely
# different set of class names and refuses to load DWB's.
MAX_V, MAX_W, ACC_V, ACC_W = 0.46, 1.9, 0.9, 7.725

PLUGIN_EXTRA = {
    'dwb_core': f"""        min_vel_x: 0.0
        min_vel_y: 0.0
        max_vel_x: {MAX_V}
        max_vel_y: 0.0
        max_vel_theta: {MAX_W}
        min_speed_xy: 0.0
        max_speed_xy: {MAX_V}
        min_speed_theta: 0.0
        acc_lim_x: {ACC_V}
        acc_lim_y: 0.0
        acc_lim_theta: {ACC_W}
        decel_lim_x: -{ACC_V}
        decel_lim_y: 0.0
        decel_lim_theta: -{ACC_W}
        vx_samples: 20
        vy_samples: 5
        vtheta_samples: 20
        sim_time: 1.7
        linear_granularity: 0.05
        angular_granularity: 0.025
        transform_tolerance: 0.2
        short_circuit_trajectory_evaluation: True
        stateful: True
        critics: ["RotateToGoal", "Oscillation", "BaseObstacle",
                  "GoalAlign", "PathAlign", "PathDist", "GoalDist"]""",

    'nav2_mppi_controller': f"""        motion_model: "DiffDrive"
        vx_max: {MAX_V}
        vx_min: -0.35
        vy_max: 0.0
        wz_max: {MAX_W}
        ax_max: {ACC_V}
        ax_min: -{ACC_V}
        az_max: {ACC_W}""",

    'nav2_regulated_pure_pursuit_controller': f"""        desired_linear_vel: {MAX_V}
        max_angular_accel: {ACC_W}
        rotate_to_heading_angular_vel: {MAX_W}""",

    'nav2_graceful_controller': f"""        v_linear_max: {MAX_V}
        v_linear_min: 0.0
        v_angular_max: {MAX_W}""",
}


def nav2_params(pkg, ns, plugin):
    """One controller_server config per nav2 lane, from the template."""
    src = os.path.join(pkg, 'config', 'race_nav2_params.yaml')
    extra = PLUGIN_EXTRA[plugin.split('::', 1)[0]]
    with open(src) as f:
        text = (f.read().replace('__NS__', ns).replace('__PLUGIN__', plugin)
                .replace('__PLUGIN_EXTRA__', extra))
    out = os.path.join(tempfile.gettempdir(), f'race_nav2_{ns}.yaml')
    with open(out, 'w') as f:
        f.write(text)
    return out


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

    for ns, (kind, spec) in ENTRANTS.items():
        if kind == 'nav2':
            # controller_server hosts its own costmap, so there is no separate
            # nav2_costmap_2d node in this lane. cmd_vel is a plain Twist in
            # this build, which is what the Gazebo bridge takes, so it is a
            # remap rather than a relay.
            ld.add_action(Node(
                package='nav2_controller', executable='controller_server',
                name='controller_server', namespace=ns, output='screen',
                parameters=[nav2_params(pkg, ns, spec), {'use_sim_time': True}],
                remappings=[('cmd_vel', f'/{ns}/cmd_vel_unstamped')]))
            continue

        # The C++ controller is the same method in another language and
        # another package; it reads and writes the same topic names, so the
        # same remaps carry it.  It declares map_frame and base_frame and
        # nothing else of the five frame parameters, because that is all it
        # looks up.
        #
        # A 'bench' lane is somebody else's planner hosted in this package's
        # baseline_controller, which does the ROS half -- costmap, carrot,
        # plant clamp -- and calls their scoring function for the command. It
        # comes up like any other lane and takes the same remaps for the same
        # reason: what the clip compares is the scoring function, so
        # everything around it has to be the same thing.  The Python
        # references are hosted by the Python node and the C and C++ ones by
        # reactive_nav_cpp's, because that is where each already compiles.
        package, controller, extra = 'reactive_autonomous_nav', None, {}
        if kind == 'cpp':
            package, controller = 'reactive_nav_cpp', 'dwa_controller'
        elif kind == 'bench':
            controller = 'baseline_controller'
            extra = {'impl': spec}
            if spec not in ('pythonrobotics', 'kmilo7204'):
                package = 'reactive_nav_cpp'
        else:
            controller = f'{spec}_controller'
        frames = {'map_frame': 'map',
                  'odom_frame': f'{ns}/odom',
                  'base_frame': f'{ns}/base_link'}
        common = dict(namespace=ns, output='screen',
                      parameters=[{'use_sim_time': True}, frames, extra],
                      remappings=remaps(ns))

        cm_cfg = costmap_params(pkg, ns)
        # Only the local costmap when no planner is running: every controller
        # in this package reads /local_costmap/costmap and none of them reads
        # the global one, so on a gated course the five 20x20 m rolling global
        # grids had no subscriber at all.
        which_maps = ('local_costmap',) if GATES else ('local_costmap',
                                                       'global_costmap')
        for which in which_maps:
            # namespace is /<ns>/<which>, not /<ns>. nav2_costmap_2d puts its
            # lifecycle node inside a sub-namespace of its own name -- the
            # single-robot launch says namespace='local_costmap',
            # name='local_costmap' for the same reason -- so a node given only
            # /r1 comes up as /r1/local_costmap and the manager waits forever
            # on /r1/local_costmap/local_costmap/get_state, which is the name
            # sim/lifecycle_up.py drives. It also puts the
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
            package=package, executable=controller,
            name=f'{controller}_node', **common))

    kinds = ', '.join(f'{ns}:{CONTROLLER[ns]}' for ns in ENTRANTS)
    print(f'race_launch: course {COURSE}, field {FIELD} -- {kinds}')
    return ld
