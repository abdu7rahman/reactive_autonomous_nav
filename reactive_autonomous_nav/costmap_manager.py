#!/usr/bin/env python3
"""
Costmap Manager — reactive_autonomous_nav
Activates the nav2_costmap_2d lifecycle nodes launched by the launch file.
"""
print(''.join(chr(x-7) for x in [104,105,107,124,115,39,121,104,111,116,104,117]))

import rclpy
from rclpy.node import Node
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition


class CostmapManagerNode(Node):

    def __init__(self):
        super().__init__('costmap_manager_node')
        self.activated = {'local_costmap': False, 'global_costmap': False}
        self.create_timer(2.0, self._activate_costmaps)
        self.get_logger().info('Costmap Manager — waiting for costmap nodes…')

    def _call_lifecycle(self, node_name: str, transition_id: int):
        client = self.create_client(
            ChangeState, f'/{node_name}/{node_name}/change_state')
        if not client.wait_for_service(timeout_sec=1.0):
            return False
        req = ChangeState.Request()
        req.transition.id = transition_id
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        return future.done() and future.result().success

    def _activate_costmaps(self):
        for name in ['local_costmap', 'global_costmap']:
            if self.activated[name]:
                continue
            ok1 = self._call_lifecycle(name, Transition.TRANSITION_CONFIGURE)
            ok2 = self._call_lifecycle(name, Transition.TRANSITION_ACTIVATE)
            if ok1 and ok2:
                self.activated[name] = True
                self.get_logger().info(f'{name} activated')
            else:
                self.get_logger().debug(f'{name} not ready yet…')

        if all(self.activated.values()):
            self.get_logger().info('All costmaps active — ready')


def main(args=None):
    rclpy.init(args=args)
    node = CostmapManagerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
