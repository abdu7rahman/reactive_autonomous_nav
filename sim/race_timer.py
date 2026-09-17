"""Start the race and time it.

    race_timer.py <sim_seconds> [--no-start]

Releases all five robots from one process and then measures, in the simulator's
own clock, when each crosses the finish line.  One process rather than five
`ros2 topic pub --once` calls because the start has to be simultaneous: a race
where the grid is released over four seconds of a world running at a real-time
factor near 0.11 is thirty-six simulated seconds of head start, which is most
of the race.

Progress is the robot's own odometry x.  Each robot's odom frame is created
where it was spawned with x along the heading it spawned in (see race_up.sh),
so odom x is distance along the course -- no transform, no localiser, and the
same quantity for all five.  It stays the right quantity now the course has a
chicane in it: the weave is across the course, which is odom y, and odom x is
still how far up it the robot has got.

The finish line is 6.0 - 0.15 - 0.05 = 5.80 m for a 6.0 m goal: the goal
distance, less the goal tolerance every controller in this package stops
inside, less one costmap cell.  The cell is not padding.  The controllers
follow the planner's path and measure their arrival against its last waypoint,
and that waypoint is a 0.05 m grid cell centre rather than the goal pose, so
the furthest a controller can be asked to reach is a cell short of the goal.
A line at 6.0 - 0.15 was measured to be 0.02 m too far: pure_pursuit stopped at
5.84 and stanley at 5.83, both correctly at their own goal, and both were
recorded as never having finished.

/goal_pose is not latched and a publisher that exits immediately after writing
loses the message if the subscriber has not finished matching, so this waits
for each planner's subscription to appear before it publishes -- a full
recorded run was once 190 seconds of a robot that had never been told where to
go.

What gets released depends on the course.  On the straight it is five goals,
one per robot, and each robot's own A* plans its way there.  On the chicane it
is five copies of one reference path (race_path.py) straight onto /<ns>/plan,
because the question there is how each controller tracks the same curve and
five separate A* runs on five rolling costmaps are five different curves.  The
planners are not launched at all on the chicane -- see launch/race_launch.py --
so nothing else is writing to /plan and the reference cannot be overwritten
halfway up the course.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path

from race_path import path_for
from reactive_autonomous_nav.race_course import (
    CONTROLLER, COSTMAP_RES, COURSE, GATES, LANES, RACE_LENGTH, START_Y)

# The lane positions, the start line and which controller is in which lane all
# come from race_course.py, which is also what grid_tf.py pins each odom origin
# into `map` from and what the launch file builds the stack from, so a goal
# cannot be placed relative to a lane the robot was not spawned in.
GRID = [(ns, x, CONTROLLER[ns]) for ns, x in LANES.items()]
GOAL_TOL = 0.15                     # every controller in this package
FINISH = RACE_LENGTH - GOAL_TOL - COSTMAP_RES

# A robot that has gained less than a centimetre in this many simulated seconds
# has stopped racing, whether or not it crossed.  Without it the first timed
# race spent 160 simulated seconds watching two stationary robots and was
# killed before it could print its table.
STALL = 25.0
MOVED = 0.01


class Timer(Node):

    def __init__(self) -> None:
        super().__init__('race_timer')
        self.t0: float | None = None
        self.progress = {ns: 0.0 for ns, _x, _c in GRID}
        self.moved = {ns: 0.0 for ns, _x, _c in GRID}
        self.finished: dict[str, float] = {}
        for ns, _x, _c in GRID:
            self.create_subscription(
                Odometry, f'/{ns}/odom',
                lambda m, ns=ns: self._odom(ns, m), qos_profile_sensor_data)
        # One or the other, never both: on a course with gates the reference
        # path goes straight to the controllers, on one without it the goal
        # goes to the planners.
        if GATES:
            self.release_pubs = {
                ns: self.create_publisher(Path, f'/{ns}/plan', 10)
                for ns, _x, _c in GRID}
        else:
            self.release_pubs = {
                ns: self.create_publisher(PoseStamped, f'/{ns}/goal_pose', 10)
                for ns, _x, _c in GRID}

    @property
    def sim(self) -> float:
        """The simulator's clock, through the node's own.

        Not a subscription to /clock.  The obvious spelling of that --
        create_subscription(Clock, '/clock', self._clock, 10) -- collides with
        rclpy.node.Node's own `_clock` attribute, so `self._clock` resolves to
        the node's Clock object rather than to the method, and rclpy rejects it
        as "callback should be either be callable with one argument". The node
        clock is what use_sim_time already wires to /clock anyway, so there is
        nothing to subscribe to.
        """
        return self.get_clock().now().nanoseconds * 1e-9

    def _odom(self, ns: str, msg: Odometry) -> None:
        x = msg.pose.pose.position.x
        if x > self.progress[ns] + MOVED:
            self.moved[ns] = self.sim
        if x > self.progress[ns]:
            self.progress[ns] = x
        if ns not in self.finished and x >= FINISH and self.t0 is not None:
            self.finished[ns] = self.sim - self.t0
            self.get_logger().info(
                f'{ns} crossed at {self.finished[ns]:.1f} s '
                f'({len(self.finished)}/{len(GRID)})')

    def wait_for_listeners(self, secs: float) -> int:
        end = time.time() + secs
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.2)
            ready = sum(1 for p in self.release_pubs.values()
                        if p.get_subscription_count() >= 1)
            if ready == len(GRID):
                return ready
        return sum(1 for p in self.release_pubs.values()
                   if p.get_subscription_count() >= 1)

    def _goal(self, x: float) -> PoseStamped:
        m = PoseStamped()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.position.x = x
        m.pose.position.y = START_Y + RACE_LENGTH
        m.pose.orientation.w = 1.0
        return m

    def _plan(self, ns: str) -> Path:
        m = Path()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        for x, y in path_for(ns):
            p = PoseStamped()
            p.header = m.header
            p.pose.position.x = x
            p.pose.position.y = y
            p.pose.orientation.w = 1.0
            m.poses.append(p)
        return m

    def release(self) -> None:
        """All five at once: the goal, or the reference path."""
        for ns, x, _c in GRID:
            self.release_pubs[ns].publish(
                self._plan(ns) if GATES else self._goal(x))
        self.t0 = self.sim
        self.moved = {ns: self.t0 for ns, _x, _c in GRID}
        what = (f'{len(self._plan(GRID[0][0]).poses)}-point reference paths'
                if GATES else f'goals {RACE_LENGTH:.1f} m ahead')
        print(f'grid released at sim {self.t0:.1f}s on the {COURSE}: '
              f'{what}, finish line {FINISH:.2f} m', flush=True)


def main() -> int:
    budget = float(sys.argv[1])
    start = '--no-start' not in sys.argv[2:]

    rclpy.init()
    t = Timer()

    # Clock first: t0 has to be a real simulator reading, not a zero.
    end = time.time() + 60
    while time.time() < end and t.sim == 0.0:
        rclpy.spin_once(t, timeout_sec=0.2)
    if t.sim == 0.0:
        print('no /clock -- the simulator is not running', flush=True)
        rclpy.shutdown()
        return 1

    if start:
        ready = t.wait_for_listeners(120)
        topic = '/plan' if GATES else '/goal_pose'
        print(f'listening on {topic}: {ready}/{len(GRID)}', flush=True)
        t.release()
    else:
        # --no-start is the start-line check race.sh runs before the nav stack
        # exists, so there is nothing to wait for and waiting out the full 120 s
        # for subscribers that have not been launched yet is the whole budget.
        for _ in range(40):
            rclpy.spin_once(t, timeout_sec=0.2)
        t.t0 = t.sim

    last = -1.0
    reason = 'budget'
    while t.sim - t.t0 < budget:
        if not rclpy.ok():
            reason = 'shutdown'
            break
        rclpy.spin_once(t, timeout_sec=0.2)
        if len(t.finished) == len(GRID):
            reason = 'all finished'
            break
        if all(ns in t.finished or t.sim - t.moved[ns] > STALL
               for ns, _x, _c in GRID):
            reason = f'no robot has moved for {STALL:.0f}s'
            break
        if t.sim - last >= 20.0:
            last = t.sim
            row = '  '.join(f'{ns} {t.progress[ns]:4.1f}' for ns, _x, _c in GRID)
            print(f'  sim {t.sim - t.t0:6.1f}s  {row}', flush=True)

    print('', flush=True)
    print(f'race over: {reason}', flush=True)
    print('finish order (simulated seconds from the grid release):', flush=True)
    order = sorted(GRID, key=lambda g: (t.finished.get(g[0], float('inf')),
                                        -t.progress[g[0]]))
    for place, (ns, _x, ctrl) in enumerate(order, 1):
        if ns in t.finished:
            print(f'  {place}. {ns:3s} {ctrl:13s} {t.finished[ns]:7.1f} s  '
                  f'({t.progress[ns]:.2f} m)', flush=True)
        else:
            print(f'  -. {ns:3s} {ctrl:13s}       -  '
                  f'reached {t.progress[ns]:.2f} m of {FINISH:.2f}', flush=True)
    if rclpy.ok():
        rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
