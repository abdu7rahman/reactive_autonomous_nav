from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'reactive_autonomous_nav'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='abdul rahman',
    maintainer_email='mohammedabdulr.1@northeastern.edu',
    description='Reactive autonomous navigation: A* + vectorized DWA for TurtleBot4',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'astar_planner           = reactive_autonomous_nav.astar_planner:main',
            'theta_star_planner      = reactive_autonomous_nav.theta_star_planner:main',
            'smac_planner            = reactive_autonomous_nav.smac_planner:main',
            'rrt_planner             = reactive_autonomous_nav.rrt_planner:main',
            'rrt_smac_hybrid_planner = reactive_autonomous_nav.rrt_smac_hybrid_planner:main',
            'dwa_controller          = reactive_autonomous_nav.dwa_controller:main',
            'pure_pursuit_controller = reactive_autonomous_nav.pure_pursuit_controller:main',
            'stanley_controller      = reactive_autonomous_nav.stanley_controller:main',
            'teb_controller          = reactive_autonomous_nav.teb_controller:main',
            'mppi_controller         = reactive_autonomous_nav.mppi_controller:main',
            'costmap_manager         = reactive_autonomous_nav.costmap_manager:main',
        ],
    },
)
