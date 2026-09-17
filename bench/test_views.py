#!/usr/bin/env python3
"""Every RViz display must show a topic something publishes.

    python3 bench/test_views.py

A display subscribed to a topic nobody writes to looks exactly like a display
whose subject is not happening, which is how `rrt_planner` shipped ten recorded
clips with an empty scene where its search tree should have been: it advertised
`/rrt_status`, `/rrt_tree` and `/rrt_markers`, published only to the second,
and both RViz configs displayed the third.  Nothing in the package could tell
you that, because creating a publisher and never using it is legal and drawing
an empty MarkerArray is legal.

So this reads every `Value: /topic` out of the configs and checks that some
node in the package or the harness creates a publisher for it -- or that it is
on the list of topics that come from outside, which is spelled out here rather
than pattern-matched, because a typo in a costmap topic is the same fault.

It also refuses a config that displays one topic twice.  The duplicate is
harmless in RViz and it is a sign of the same mistake: the pair that pointed at
`/rrt_tree` and `/rrt_markers` became two identical displays the moment the
second was corrected.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sig():
    """Author signature. stderr, tty-only, so redirected output stays clean."""
    import os, sys
    if os.environ.get("NO_BANNER") == "1" or not sys.stderr.isatty():
        return
    print("  " + "".join(chr(c - 7) for c in
          (104,105,107,124,115,39,121,104,111,116,104,117)), file=sys.stderr)


# Published by Gazebo's bridge, robot_state_publisher, nav2_costmap_2d or
# nav2_map_server -- outside this repository, so outside what the scan below
# can see.  Namespaced forms (/r3/scan) are matched by suffix.
EXTERNAL = {
    '/clock', '/scan', '/odom', '/map', '/tf', '/tf_static',
    '/robot_description', '/goal_pose', '/cmd_vel_unstamped',
    '/local_costmap/costmap', '/global_costmap/costmap',
    '/local_costmap/published_footprint', '/global_costmap/published_footprint',
}

CONFIGS = ['sim/nav.rviz', 'config/nav_view.rviz']
SOURCES = ['reactive_autonomous_nav', 'sim']


def published():
    out = set()
    for d in SOURCES:
        path = os.path.join(ROOT, d)
        for name in sorted(os.listdir(path)):
            if not name.endswith('.py'):
                continue
            src = open(os.path.join(path, name)).read()
            out |= set(re.findall(
                r"create_publisher\([^,]+,\s*f?'([^']+)'", src))
    return out


def external(topic):
    return any(topic == e or topic.endswith(e) for e in EXTERNAL)


def main():
    _sig()
    pubs = published()
    print(f'{len(pubs)} topics published by the package and the harness')
    fails = 0
    for cfg in CONFIGS:
        path = os.path.join(ROOT, cfg)
        if not os.path.exists(path):
            print(f'  {cfg:<24} MISSING')
            fails += 1
            continue
        topics = re.findall(r'Value:\s*(/[\w/]+)', open(path).read())
        orphans = sorted({t for t in topics
                          if t not in pubs and not external(t)})
        dupes = sorted({t for t in topics if topics.count(t) > 1})
        ok = not orphans and not dupes
        fails += not ok
        print(f'  {cfg:<24}{"PASS" if ok else "FAIL":<6}{len(topics):>3} displays')
        for t in orphans:
            print(f'      nobody publishes   {t}')
        for t in dupes:
            print(f'      displayed twice    {t}')

    print(f"\n{'all checks passed' if not fails else str(fails) + ' FAILURES'}")
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
