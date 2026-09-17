"""Configure and activate lifecycle nodes, one at a time, and say which took.

    lifecycle_up.py [seconds_per_node] [node ...]

With no node names it does whatever this race's lanes bring up: ten costmaps
on the straight, five on a course with gates (no planner runs there, and
nothing subscribes to a global costmap), and a controller_server in place of a
costmap for any lane racing a nav2 plugin, which owns its costmap itself.  Any
names given replace that list, which is how sim_up.sh brings the map server up
and how drive.sh repairs a single robot's pair.

`ros2 lifecycle set` does this job on a command line and cannot be relied on
here at all: the CLI gives discovery about a second, and on this machine with
forty-nine participants under load that is not enough, so it answers "Node not
found" for a node whose process is alive and logging. `ros2 node list` returned
0 nodes in the same graph where a 30-second rclpy subscription found /scan,
/odom and /clock delivering. sim_up.sh drove the map server's two transitions
that way with the output sent to /dev/null, printed "map served", and left the
global costmap's static layer with no map -- for as long as nobody checked.

nav2_lifecycle_manager does the same job for a list, and for the race it does
it badly.  It
brings its whole list up at once and treats any single failure as final: it
logs "Failed to bring up all requested nodes. Aborting bringup" and never
retries, so one robot losing a service call costs the entire race.  With five
managers and ten costmaps in a graph of about fifty participants that happened
on most attempts, and the failure was not a timeout waiting for anything -- it
was

    Failed to change state for node: /r5/local_costmap/local_costmap.
    Exception: .../get_state service client: async_send_request failed.

the request never leaving the client.  Retries on other attempts reported two,
then four, then four of five pairs active; three full bringups in a row were
thrown away that way.

So: one node at a time, each transition retried, and a table at the end saying
which of them are active.  Nothing here monitors the nodes afterwards,
which the manager would -- but its bond monitoring was already switched off
for this race, because with twenty-five nodes on four cores a missed heartbeat
is a scheduling delay and the manager's answer to one is to tear the costmaps
down mid-race.
"""
from __future__ import annotations

__author__ = "".join(
    chr(c - 7) for c in (104, 105, 107, 124, 115, 39, 121, 104, 111, 116, 104, 117)
)

import sys
import time

import rclpy
from rclpy.node import Node
from lifecycle_msgs.msg import Transition, State
from lifecycle_msgs.srv import ChangeState, GetState

from reactive_autonomous_nav.race_course import ENTRANTS, GATES, LANES


def _race_nodes():
    """The lifecycle nodes this race's lanes bring up, in lane order.

    A lane running one of this package's controllers has a nav2_costmap_2d
    node of its own; a lane running a nav2 controller plugin has a
    controller_server, which owns its costmap internally and is the lifecycle
    node for both.

    Only the local costmap on a gated course.  The controllers all read
    /local_costmap/costmap and nothing reads the global one there -- no planner
    is launched -- so five 20x20 m rolling grids with an obstacle layer and an
    inflation layer each were being maintained for no subscriber, on four cores
    already carrying five physics bodies and five raycast sensors.
    """
    out = []
    for ns in LANES:
        kind, _spec = ENTRANTS[ns]
        if kind == 'nav2':
            out.append(f'/{ns}/controller_server')
        else:
            for which in (('local_costmap',) if GATES
                          else ('local_costmap', 'global_costmap')):
                out.append(f'/{ns}/{which}/{which}')
    return out


RACE_COSTMAPS = _race_nodes()

WANT = [(Transition.TRANSITION_CONFIGURE, State.PRIMARY_STATE_INACTIVE, 'configure'),
        (Transition.TRANSITION_ACTIVATE,  State.PRIMARY_STATE_ACTIVE,   'activate')]
TRIES = 4


class Bringup(Node):

    def __init__(self) -> None:
        super().__init__('lifecycle_up')

    def state(self, node: str, budget: float) -> int:
        cli = self.create_client(GetState, f'{node}/get_state')
        try:
            if not cli.wait_for_service(timeout_sec=budget):
                return -1
            fut = cli.call_async(GetState.Request())
            rclpy.spin_until_future_complete(self, fut, timeout_sec=budget)
            return fut.result().current_state.id if fut.done() else -1
        except Exception:
            return -1
        finally:
            self.destroy_client(cli)

    def transition(self, node: str, tid: int, budget: float) -> bool:
        cli = self.create_client(ChangeState, f'{node}/change_state')
        try:
            if not cli.wait_for_service(timeout_sec=budget):
                return False
            req = ChangeState.Request()
            req.transition.id = tid
            fut = cli.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=budget)
            return bool(fut.done() and fut.result().success)
        except Exception:
            # async_send_request failing is exactly the case this retries.
            return False
        finally:
            self.destroy_client(cli)

    def bring_up(self, node: str, budget: float) -> bool:
        # Already active is already up.  This used to walk the transitions
        # regardless, so a node that had reached ACTIVE on a previous attempt
        # was sent a configure it could not accept -- "Unable to start
        # transition 1 from current state active: Transition is not
        # registered" -- four times, and was then reported FAILED.  Two nav2
        # lanes of a race were lost that way while both were running
        # perfectly, and the relaunch that followed made it worse.
        if self.state(node, budget) == State.PRIMARY_STATE_ACTIVE:
            return True
        for tid, want, label in WANT:
            for attempt in range(1, TRIES + 1):
                st = self.state(node, budget)
                # ACTIVE satisfies the configure step too: it is past it.
                if st == want or st == State.PRIMARY_STATE_ACTIVE:
                    break
                self.transition(node, tid, budget)
                if self.state(node, budget) in (want,
                                                State.PRIMARY_STATE_ACTIVE):
                    break
                self.get_logger().warn(
                    f'{node}: {label} attempt {attempt}/{TRIES} did not take')
                time.sleep(2.0)
            else:
                return False
        return True


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    budget = float(args[0]) if args else 30.0
    nodes = args[1:] or RACE_COSTMAPS
    rclpy.init()
    n = Bringup()
    ok = []
    for node in nodes:
        good = n.bring_up(node, budget)
        ok.append(good)
        print(f'  {"active " if good else "FAILED "} {node}', flush=True)
    print(f'active: {sum(ok)}/{len(nodes)}', flush=True)
    rclpy.shutdown()
    return 0 if all(ok) else 1


if __name__ == '__main__':
    raise SystemExit(main())
