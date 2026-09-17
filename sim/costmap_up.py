"""Configure and activate the race's ten costmaps, one at a time.

    costmap_up.py [seconds_per_node]

nav2_lifecycle_manager does this job, and for the race it does it badly.  It
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
which of the ten are active.  Nothing here monitors the nodes afterwards,
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

from grid_tf import LANES

NODES = [f'/{ns}/{which}/{which}'
         for ns in LANES
         for which in ('local_costmap', 'global_costmap')]

WANT = [(Transition.TRANSITION_CONFIGURE, State.PRIMARY_STATE_INACTIVE, 'configure'),
        (Transition.TRANSITION_ACTIVATE,  State.PRIMARY_STATE_ACTIVE,   'activate')]
TRIES = 4


class Bringup(Node):

    def __init__(self) -> None:
        super().__init__('costmap_up')

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
        for tid, want, label in WANT:
            for attempt in range(1, TRIES + 1):
                if self.state(node, budget) == want:
                    break
                self.transition(node, tid, budget)
                if self.state(node, budget) == want:
                    break
                self.get_logger().warn(
                    f'{node}: {label} attempt {attempt}/{TRIES} did not take')
                time.sleep(2.0)
            else:
                return False
        return True


def main() -> int:
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    rclpy.init()
    n = Bringup()
    ok = []
    for node in NODES:
        good = n.bring_up(node, budget)
        ok.append(good)
        print(f'  {"active " if good else "FAILED "} {node}', flush=True)
    print(f'costmaps active: {sum(ok)}/{len(NODES)}', flush=True)
    rclpy.shutdown()
    return 0 if all(ok) else 1


if __name__ == '__main__':
    raise SystemExit(main())
