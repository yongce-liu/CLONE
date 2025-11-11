import zmq
from collections import deque  # Not used in the ROS 2 version, but kept for context
import pickle
import numpy as np
import traceback

from multiprocessing import Process, Queue

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class PositionNode(Node):
    """
    ROS 2 节点：订阅 /localization 话题，并将位置和姿态数据放入一个多进程队列。
    """

    def __init__(self, queue: Queue):
        # 初始化 Node，节点名称为 'localization_listener'
        super().__init__("localization_listener")

        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self.queue = queue

        # 创建订阅器，话题名为 '/localization'，消息类型为 Odometry，QoS 深度为 10
        self.subscription = self.create_subscription(
            Odometry, "/localization", self.callback, 10
        )
        self.get_logger().info("Localization Listener Node has started.")

    def callback(self, msg: Odometry):
        """
        订阅回调函数：提取位置和姿态，并将它们放入队列。
        """

        # 提取位置
        self.position[0] = msg.pose.pose.position.x
        self.position[1] = msg.pose.pose.position.y
        self.position[2] = msg.pose.pose.position.z

        # 提取姿态 (四元数)
        self.quat[0] = msg.pose.pose.orientation.x
        self.quat[1] = msg.pose.pose.orientation.y
        self.quat[2] = msg.pose.pose.orientation.z
        self.quat[3] = msg.pose.pose.orientation.w

        # self.get_logger().info(f"Received Pos: {self.position[0]:.2f}, {self.position[1]:.2f}") # 用于调试

        # 将数据放入多进程队列。使用 put_nowait 避免阻塞。
        # 如果队列已满，则跳过此次更新，保持最新的数据。
        try:
            # 清空队列中可能存在的旧数据，确保发送的是最新的数据
            while not self.queue.empty():
                self.queue.get_nowait()

            self.queue.put_nowait((self.position.copy(), self.quat.copy()))
        except:
            # 捕获异常，如队列已满
            # traceback.print_exc() # 调试时可打开
            pass

    @staticmethod
    def start_main_loop(queue: Queue):
        """
        这是一个静态方法，用于在独立的进程中运行 ROS 2 节点。
        注意：rclpy.init() 和 rclpy.shutdown() 必须在进程内部调用。
        """
        # 必须在进程内部初始化 ROS 2
        rclpy.init(args=None)
        node = PositionNode(queue)

        # 启动 ROS 2 消息循环
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("Localization Listener Node shutting down.")
        finally:
            node.destroy_node()
            # 必须在进程内部关闭 ROS 2
            rclpy.shutdown()


class PositionServer:
    """
    ZeroMQ 服务器：从 ROS 2 进程获取数据，并以 PUB/SUB 模式发送。
    """

    def __init__(self, config: dict):
        self.port = config["port"]
        self.server_ip = config[
            "server_ip"
        ]  # Server IP is not strictly used for binding, but kept for context

        # 队列的最大容量设为 1，确保只传输最新的位置数据
        self.queue = Queue(maxsize=1)
        self.ros_process = None  # 存储 ROS 2 进程的引用

        # Set ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        # 绑定到所有接口的指定端口
        self.socket.bind(f"tcp://*:{self.port}")

        print(
            f"[Position Server] Position server has started on port {self.port}, waiting for client connections..."
        )

    def _close(self):
        """关闭 ZeroMQ 连接和 ROS 2 进程。"""
        # 终止 ROS 2 进程
        if self.ros_process and self.ros_process.is_alive():
            self.ros_process.terminate()
            self.ros_process.join()
            print("[Position Server] ROS 2 Listener process terminated.")

        # 关闭 ZeroMQ
        self.socket.close()
        self.context.term()
        print("[Position Server] ZeroMQ server has been closed.")

    def send_process(self):
        """
        启动 ROS 2 监听进程，并在主进程中循环获取数据并通过 ZeroMQ 发送。
        """
        # 启动 ROS 2 节点进程
        self.ros_process = Process(
            target=PositionNode.start_main_loop, args=(self.queue,)
        )
        self.ros_process.start()

        try:
            while True:
                # 阻塞式获取队列中的最新数据 (position, quat)
                # 这会等待直到 ROS 2 进程放入新数据
                position, quat = self.queue.get()

                # 序列化数据
                localization_message = pickle.dumps((position, quat))

                # 通过 ZeroMQ 发布数据
                self.socket.send(localization_message)

        except KeyboardInterrupt:
            print("\n[Position Server] Interrupted by user. Shutting down...")
        finally:
            self._close()


if __name__ == "__main__":
    # 在主进程中，不应该调用 rclpy.init() 或 rclpy.shutdown()
    # 因为 ROS 2 的初始化和关闭都移动到了子进程 PositionNode.start_main_loop 中。

    config = {
        "port": 6006,
        "server_ip": "192.168.123.164",
    }

    pos_server = PositionServer(config)
    pos_server.send_process()
