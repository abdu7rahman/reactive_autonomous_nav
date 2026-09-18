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

Neither /goal_pose nor /plan is latched, and a publisher that writes once
loses the message if matching has not finished, so this confirms rather than
counts.  Counting failed twice.  A 190-second recorded run was once a robot
that had never been told where to go, before any wait existed; and the first
chicane race waited for `/rN/plan` to have a subscriber, found five of five,
published, and watched all five robots sit still for twenty-five simulated
seconds -- because the subscriber it found was RViz's Path display, which
subscribes to exactly those five topics to draw the reference.  So the wait is
for a subscription belonging to a node whose name ends in `_controller_node`,
and after the release each robot's own odometry has to show it moving; one
that does not gets its path again, and the resend is reported, because a robot
that started two seconds late did not run the same race as the other four.

What gets released depends on the course and on who is racing.  On the
straight it is five goals, one per robot, and each robot's own A* plans its way
there.  On the chicane it is five copies of one reference path (race_path.py),
because the question there is how each controller tracks the same curve and
five separate A* runs on five rolling costmaps are five different curves.  The
planners are not launched at all on the chicane -- see launch/race_launch.py --
so nothing else is writing to /plan and the reference cannot be overwritten
halfway up the course.

How that path reaches a lane depends on what is in it.  This package's
controllers take it on /<ns>/plan.  A nav2 controller plugin takes it as a
FollowPath action goal on /<ns>/follow_path, which is the only way to hand a
nav2 controller_server a path at all, and the goal carries the same path.  Both
go out in the same loop, so the two kinds of lane start together.
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
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from nav2_msgs.action import FollowPath

from race_path import path_for
from reactive_autonomous_nav.race_course import (
    CONTROLLER, COSTMAP_RES, COURSE, ENTRANTS, FIELD, GATES, LANES,
    RACE_LENGTH, START_Y)

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

# How long to give a robot to start moving before assuming it never got its
# path, in simulated seconds.  The controllers run at 10 Hz and every one of
# them commands a non-zero velocity within a tick or two of receiving a plan --
# measured in bench/test_chicane.py, where all five are moving by step 3 -- so
# 3 s is thirty ticks of margin rather than a guess.
CONFIRM = 3.0
RESENDS = 3


class Timer(Node):

    def __init__(self) -> None:
        super().__init__('race_timer')
        self.t0: float | None = None
        self.lane_x = {ns: x for ns, x, _c in GRID}
        self.progress = {ns: 0.0 for ns, _x, _c in GRID}
        self.moved = {ns: 0.0 for ns, _x, _c in GRID}
        # Where each robot was when its `moved` stamp was last refreshed.
        # Separate from progress[] on purpose -- see _odom.
        self.anchor = {ns: 0.0 for ns, _x, _c in GRID}
        self.finished: dict[str, float] = {}
        for ns, _x, _c in GRID:
            self.create_subscription(
                Odometry, f'/{ns}/odom',
                lambda m, ns=ns: self._odom(ns, m), qos_profile_sensor_data)
        # One or the other, never both: on a course with gates the reference
        # path goes straight to the controllers, on one without it the goal
        # goes to the planners.
        self.release_pubs = {}
        self.follow = {}
        for ns, _x, _c in GRID:
            if ENTRANTS[ns][0] == 'nav2':
                self.follow[ns] = ActionClient(
                    self, FollowPath, f'/{ns}/follow_path')
            elif GATES:
                self.release_pubs[ns] = self.create_publisher(
                    Path, f'/{ns}/plan', 10)
            else:
                self.release_pubs[ns] = self.create_publisher(
                    PoseStamped, f'/{ns}/goal_pose', 10)

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
        # Against the anchor -- where it was when `moved` was last stamped --
        # and not against progress, which is bumped to x a line further down
        # on every improvement.  Compared against progress this asked whether
        # a *single odometry message* had advanced more than a centimetre,
        # which at 0.29 m/s and 50 Hz of odometry is 0.006 m and never true.
        # So a robot driving steadily was recorded as not having moved since
        # the grid released, and once the other four crossed, the stall rule
        # below ended the race at exactly t0 + STALL.
        #
        # It cost the chicane's fifth lane a finish: MPPI drove the course in
        # 24.8 s on a host where that fitted inside the 25 s guillotine by two
        # tenths, and on a slower one it was cut off at 4.39 m of 5.80 while
        # its own log showed it 0.71 m from the goal and still closing.  The
        # comment on STALL says "has gained less than a centimetre in this
        # many simulated seconds", which is what this now measures.
        if x > self.anchor[ns] + MOVED:
            self.moved[ns] = self.sim
            self.anchor[ns] = x
        if x > self.progress[ns]:
            self.progress[ns] = x
        if ns not in self.finished and x >= FINISH and self.t0 is not None:
            self.finished[ns] = self.sim - self.t0
            self.get_logger().info(
                f'{ns} crossed at {self.finished[ns]:.1f} s '
                f'({len(self.finished)}/{len(GRID)})')

    def _listeners(self) -> int:
        """Lanes that can actually receive what we are about to send.

        By node name, not by count.  On a gated course the release goes to
        /<ns>/plan, and so does RViz: five Path displays, one per lane, which
        made a count of subscribers read 5/5 with no controller listening at
        all.  get_subscriptions_info_by_topic carries the node name, so the
        question can be asked properly.  A nav2 lane has no subscription to
        count either way -- it is an action server, and whether it is up is
        exactly what wait_for_server answers.
        """
        ready = 0
        for ns in self.release_pubs:
            topic = f'/{ns}/plan' if GATES else f'/{ns}/goal_pose'
            want = '_controller_node' if GATES else '_planner_node'
            ready += any(e.node_name.endswith(want)
                         for e in self.get_subscriptions_info_by_topic(topic))
        for ns, client in self.follow.items():
            ready += bool(client.server_is_ready())
        return ready

    def wait_for_listeners(self, secs: float) -> int:
        end = time.time() + secs
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self._listeners() == len(GRID):
                # Discovery knowing about a subscription and this publisher
                # having matched it are not the same instant.  Two seconds is
                # far more than the gap measured here and costs nothing: the
                # clock starts after it.
                end2 = time.time() + 2.0
                while time.time() < end2:
                    rclpy.spin_once(self, timeout_sec=0.2)
                return len(GRID)
        return self._listeners()

    def confirm_moving(self) -> list[str]:
        """Release, then make sure every robot actually took it.

        Returns the robots that needed the path sent more than once, which is
        a fairness problem rather than a failure -- a robot given its path
        three seconds late is three seconds behind for a reason that has
        nothing to do with its controller -- so the caller reports it and the
        race is worth re-running.
        """
        late: list[str] = []
        for attempt in range(RESENDS):
            missing = [ns for ns, _x, _c in GRID if self.progress[ns] < MOVED]
            if not missing:
                break
            if attempt:
                late += [ns for ns in missing if ns not in late]
                for ns in missing:
                    self._send(ns, self.lane_x[ns])
                print(f'  resent to {" ".join(missing)} '
                      f'(attempt {attempt + 1})', flush=True)
            deadline = self.sim + CONFIRM
            while self.sim < deadline and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.2)
        return late

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

    def _send(self, ns: str, x: float) -> None:
        """Hand one lane its path, the way that lane can take it."""
        if ns in self.follow:
            goal = FollowPath.Goal()
            goal.path = self._plan(ns)
            goal.controller_id = 'FollowPath'
            goal.goal_checker_id = 'general_goal_checker'
            self.follow[ns].send_goal_async(goal)
        else:
            self.release_pubs[ns].publish(
                self._plan(ns) if GATES else self._goal(x))

    def release(self) -> None:
        """All five at once: the goal, or the reference path."""
        for ns, x, _c in GRID:
            self._send(ns, x)
        self.t0 = self.sim
        self.moved = {ns: self.t0 for ns, _x, _c in GRID}
        self.anchor = {ns: self.progress[ns] for ns, _x, _c in GRID}
        what = (f'{len(self._plan(GRID[0][0]).poses)}-point reference paths'
                if GATES else f'goals {RACE_LENGTH:.1f} m ahead')
        how = (f' ({len(self.follow)} of them as FollowPath goals)'
               if self.follow else '')
        print(f'grid released at sim {self.t0:.1f}s on the {COURSE}, '
              f'field {FIELD}: {what}{how}, finish line {FINISH:.2f} m',
              flush=True)


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
        print(f'controllers listening on {topic}: {ready}/{len(GRID)}'
              if GATES else
              f'planners listening on {topic}: {ready}/{len(GRID)}', flush=True)
        t.release()
        late = t.confirm_moving()
        moving = sum(1 for ns, _x, _c in GRID if t.progress[ns] >= MOVED)
        print(f'moving after the release: {moving}/{len(GRID)}'
              + (f' -- {" ".join(late)} needed the path sent again, so this '
                 f'race is not a fair one' if late else ''), flush=True)
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
    print(f'race over: {reason} at {t.sim - t.t0:.1f}s', flush=True)
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
