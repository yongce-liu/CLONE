import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

import zmq
import pickle
import numpy as np
from multiprocessing import Process, Queue


class PositionNode(Node):
    def __init__(self, queue: Queue):
        super().__init__("localization_listener")
        self.queue = queue
        self.subscription = self.create_subscription(
            Odometry, "/localization", self.callback, 10
        )

    def callback(self, data: Odometry):
        position = np.array(
            [
                data.pose.pose.position.x,
                data.pose.pose.position.y,
                data.pose.pose.position.z,
            ],
            dtype=np.float32,
        )
        quat = np.array(
            [
                data.pose.pose.orientation.x,
                data.pose.pose.orientation.y,
                data.pose.pose.orientation.z,
                data.pose.pose.orientation.w,
            ],
            dtype=np.float32,
        )
        self.get_logger().info(f"Position: {position}, Quaternion: {quat}")
        try:
            self.queue.put_nowait((position, quat))
        except Exception as e:
            self.get_logger().warn(f"Queue put failed: {e}")

    @staticmethod
    def start_main_loop(queue: Queue):
        rclpy.init()
        node = PositionNode(queue)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()


class PositionServer:
    def __init__(self, config: dict):
        self.port = config["port"]
        self.server_ip = config["server_ip"]

        self.queue = Queue(maxsize=1)

        # Setup ZeroMQ
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")

        print(
            f"[Position Server] Started on {self.server_ip}:{self.port}, waiting for data..."
        )

    def _close(self):
        self.socket.close()
        self.context.term()
        print("[Position Server] Closed.")

    def send_process(self):
        process = Process(target=PositionNode.start_main_loop, args=(self.queue,))
        process.start()
        try:
            while True:
                position, quat = self.queue.get()
                localization_message = pickle.dumps((position, quat))
                self.socket.send(localization_message)
        except KeyboardInterrupt:
            print("[Position Server] Interrupted by user.")
        finally:
            self._close()
            process.terminate()
            process.join()


if __name__ == "__main__":
    config = {
        "port": 60060,
        "server_ip": "192.168.123.164",
    }
    pos_server = PositionServer(config)
    pos_server.send_process()
