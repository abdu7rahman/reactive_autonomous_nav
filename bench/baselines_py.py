"""Fetch the two reference Python DWAs, with only their plotting stripped.

Split out of dwa_compare.py because three things now load them and one of
them is a live ROS 2 node: reactive_autonomous_nav/baseline_controller.py
races these implementations in Gazebo, and importing dwa_compare to get at
its loaders would drag bench/rig.py -- whose whole job is stubbing rclpy --
into a process that has the real one.

Both are fetched at run time rather than vendored, cached beside this file,
and the only edits are to their plotting. The planner functions are theirs.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

PR_URL = ("https://raw.githubusercontent.com/AtsushiSakai/PythonRobotics/"
          "master/PathPlanning/DynamicWindowApproach/dynamic_window_approach.py")
PR_CACHE = os.path.join(HERE, ".pr_dwa.py")

# kmilo7204/dwa_planner: a standalone Python DWA that keeps its tuning in a
# sibling module, so both files are fetched.
KM_URL = "https://raw.githubusercontent.com/kmilo7204/dwa_planner/master/dwa.py"
KM_CFG_URL = "https://raw.githubusercontent.com/kmilo7204/dwa_planner/master/robot_config.py"
KM_CACHE = os.path.join(HERE, ".km_dwa.py")
KM_CFG_CACHE = os.path.join(HERE, ".km_robot_config.py")


def load_reference():
    """AtsushiSakai/PythonRobotics dynamic_window_approach.py (MIT)."""
    if not os.path.exists(PR_CACHE):
        src = urllib.request.urlopen(PR_URL, timeout=30).read().decode()
        src = src.replace("show_animation = True", "show_animation = False")
        # the reference imports matplotlib at module scope purely for its demo;
        # the planner functions themselves do not need it
        src = src.replace("import matplotlib.pyplot as plt",
                          "plt = None  # plotting stripped: only the planner is timed")
        open(PR_CACHE, "w").write(src)
    import importlib.util
    spec = importlib.util.spec_from_file_location("pr_dwa", PR_CACHE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.show_animation = False
    return m


def load_kmilo():
    """kmilo7204/dwa_planner, fetched with only its plotting stripped."""
    if not os.path.exists(KM_CFG_CACHE):
        open(KM_CFG_CACHE, "w").write(
            urllib.request.urlopen(KM_CFG_URL, timeout=30).read().decode())
    if not os.path.exists(KM_CACHE):
        src = urllib.request.urlopen(KM_URL, timeout=30).read().decode()
        src = src.replace("import matplotlib.pyplot as plt",
                          "plt = None  # plotting stripped: only the planner is timed")
        src = src.replace("import robot_config", "import robot_config_km as robot_config")
        # an unused import that numpy removed the target of; the planner never
        # calls it, so dropping it changes nothing that is timed
        src = src.replace("from numpy.lib.npyio import load", "")
        open(KM_CACHE, "w").write(src)
    import importlib.util
    for name, path in (("robot_config_km", KM_CFG_CACHE), ("km_dwa", KM_CACHE)):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
    return sys.modules["km_dwa"]
