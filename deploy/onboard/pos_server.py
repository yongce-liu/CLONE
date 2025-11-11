import zmq
from collections import deque
import pickle
import numpy as np

from multiprocessing import Process, Queue

import rospy
from nav_msgs.msg import Odometry


class Position_Node:
    def __init__(self, queue):
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.queue = queue

    def callback(self, data):
        rospy.loginfo(data.pose.pose.position)
        rospy.loginfo(data.pose.pose.orientation)
        self.position = np.array(
            [
                data.pose.pose.position.x,
                data.pose.pose.position.y,
                data.pose.pose.position.z,
            ],
            dtype=np.float32,
        )
        self.quat = np.array(
            [
                data.pose.pose.orientation.x,
                data.pose.pose.orientation.y,
                data.pose.pose.orientation.z,
                data.pose.pose.orientation.w,
            ],
            dtype=np.float32,
        )
        try:
            self.queue.put_nowait((self.position, self.quat))
        except:
            import traceback

            traceback.print_exc()
            pass

    def localization_listener(self):
        rospy.init_node("localization_listener", anonymous=True)

        rospy.Subscriber("/localization", Odometry, self.callback)

        rospy.spin()

    def main_loop(self):
        self.localization_listener()

    @staticmethod
    def start_main_loop(queue: Queue):
        node = Position_Node(queue)
        node.main_loop()


class Position_Server:
    def __init__(self, config: dict = None, Unit_Test=False):
        self.port = config["port"]
        self.server_ip = config["server_ip"]
        # self.Unit_Test = Unit_Test

        self.queue = Queue(maxsize=1)
        # self.Pos_node = Position_Node()

        # Set ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")

        # if self.Unit_Test:
        #     self._init_performance_metrics()

        print(
            "[Position Server] Position server has started, waiting for client connections..."
        )

    def _close(self):
        self.socket.close()
        self.context.term()
        # self.sock.close()
        print("[Position Server] The server has been closed.")

    def send_process(self):
        # process = Process(target=self.Pos_node.main_loop)
        process = Process(target=Position_Node.start_main_loop, args=(self.queue,))
        process.start()
        try:
            while True:
                # position = self.Pos_node.position
                position, quat = self.queue.get()
                localization_message = pickle.dumps((position, quat))
                self.socket.send(localization_message)
        except KeyboardInterrupt:
            print("[Position Server] Interrupted by user.")
        finally:
            self._close()
        process.close()


if __name__ == "__main__":
    config = {
        "port": 60060,
        "server_ip": "192.168.123.164",
    }
    pos_server = Position_Server(config)
    pos_server.send_process()
