print(''.join(chr(x-7) for x in [104,105,107,124,115,39,121,104,111,116,104,117]))
"""
Launch file for reactive_autonomous_nav.

Usage:
  # Real robot (default, A* planner)
  ros2 launch reactive_autonomous_nav nav_launch.py

  # Simulation with A* planner
  ros2 launch reactive_autonomous_nav nav_launch.py use_sim_time:=true

  # Simulation with SMAC global + Pure Pursuit local
  ros2 launch reactive_autonomous_nav nav_launch.py use_sim_time:=true planner:=smac controller:=pure_pursuit

  # Simulation with RRT global + TEB local
  ros2 launch reactive_autonomous_nav nav_launch.py use_sim_time:=true planner:=rrt controller:=teb
  
  # Simulation with Theta* global + MPPI local (information-theoretic optimal control)
  ros2 launch reactive_autonomous_nav nav_launch.py use_sim_time:=true planner:=theta_star controller:=mppi
  
  # Simulation with RRT-SMAC Hybrid global + DWA local (best of both worlds)
  ros2 launch reactive_autonomous_nav nav_launch.py use_sim_time:=true planner:=rrt_smac_hybrid controller:=dwa
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def _launch_nav_nodes(context):
    """OpaqueFunction: Resolve planner & controller and return both nodes."""
    planner    = context.launch_configurations['planner']
    controller = context.launch_configurations['controller']
    use_sim    = context.launch_configurations['use_sim_time']

    # 1. Resolve Global Planner
    if planner == 'theta_star':
        p_exec = 'theta_star_planner'
        p_name = 'theta_star_planner_node'
    elif planner == 'smac':
        p_exec = 'smac_planner'
        p_name = 'smac_planner_node'
    elif planner == 'rrt':
        p_exec = 'rrt_planner'
        p_name = 'rrt_planner_node'
    elif planner == 'rrt_smac_hybrid':
        p_exec = 'rrt_smac_hybrid_planner'
        p_name = 'rrt_smac_hybrid_planner_node'
    else: # default: astar
        p_exec = 'astar_planner'
        p_name = 'astar_planner_node'

    # 2. Resolve Local Controller
    if controller == 'pure_pursuit':
        c_exec = 'pure_pursuit_controller'
        c_name = 'pure_pursuit_controller_node'
    elif controller == 'stanley':
        c_exec = 'stanley_controller'
        c_name = 'stanley_controller_node'
    elif controller == 'teb':
        c_exec = 'teb_controller'
        c_name = 'teb_controller_node'
    elif controller == 'mppi':
        c_exec = 'mppi_controller'
        c_name = 'mppi_controller_node'
    else: # default: dwa
        c_exec = 'dwa_controller'
        c_name = 'dwa_controller_node'

    return [
        Node(
            package    = 'reactive_autonomous_nav',
            executable = p_exec,
            name       = p_name,
            output     = 'screen',
            parameters = [{'use_sim_time': use_sim == 'true'}],
        ),
        Node(
            package    = 'reactive_autonomous_nav',
            executable = c_exec,
            name       = c_name,
            output     = 'screen',
            parameters = [{'use_sim_time': use_sim == 'true'}],
        ),
    ]


def generate_launch_description():
    pkg = get_package_share_directory('reactive_autonomous_nav')

    # ── launch arguments ─────────────────────────────────────────────
    sim_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Set to true when running in simulation',
    )
    planner_arg = DeclareLaunchArgument(
        'planner',
        default_value='astar',
        description="Global planner: 'astar', 'theta_star', 'smac', 'rrt', or 'rrt_smac_hybrid'",
    )
    controller_arg = DeclareLaunchArgument(
        'controller',
        default_value='dwa',
        description="Local controller: 'dwa', 'pure_pursuit', 'stanley', 'teb', or 'mppi'",
    )
    use_sim = LaunchConfiguration('use_sim_time')

    # ── config paths ─────────────────────────────────────────────────
    cm_cfg = os.path.join(pkg, 'config', 'costmap_params.yaml')

    # ── nav2 costmap lifecycle nodes ─────────────────────────────────
    costmaps = [
        Node(
            package    = 'nav2_costmap_2d',
            executable = 'nav2_costmap_2d',
            name       = 'local_costmap',
            namespace  = 'local_costmap',
            output     = 'screen',
            parameters = [cm_cfg, {'use_sim_time': use_sim}],
        ),
        Node(
            package    = 'nav2_costmap_2d',
            executable = 'nav2_costmap_2d',
            name       = 'global_costmap',
            namespace  = 'global_costmap',
            output     = 'screen',
            parameters = [cm_cfg, {'use_sim_time': use_sim}],
        ),
        Node(
            package    = 'nav2_lifecycle_manager',
            executable = 'lifecycle_manager',
            name       = 'lifecycle_manager_costmap',
            output     = 'screen',
            parameters = [{
                'use_sim_time': use_sim,
                'autostart':    True,
                'bond_timeout': 0.0,
                'node_names':   ['local_costmap/local_costmap',
                                 'global_costmap/global_costmap'],
            }],
        ),
    ]

    # ── custom planner + controller (delayed 5 s) ────────────────────
    nav_launch = OpaqueFunction(function=_launch_nav_nodes)

    return LaunchDescription([
        sim_arg,
        planner_arg,
        controller_arg,
        *costmaps,
        TimerAction(period=5.0, actions=[nav_launch]),
    ])
