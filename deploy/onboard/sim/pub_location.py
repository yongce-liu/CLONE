import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion
from std_msgs.msg import Header
import time
import math


class LocalizationPublisher(Node):
    """
    ROS 2 节点：循环发布 nav_msgs/msg/Odometry 类型的 /localization 话题。
    """

    def __init__(self):
        # 初始化 Node，节点名称为 'localization_publisher_node'
        super().__init__("localization_publisher_node")

        # 创建发布器，话题名为 '/localization'，消息类型为 Odometry，QoS 深度为 10
        self.publisher_ = self.create_publisher(Odometry, "/localization", 10)

        # 定义发布频率（Hz）
        timer_period = 0.01  # 2 Hz

        # 创建定时器，到期后调用 timer_callback 方法
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # 初始化计数器和模拟的位置数据
        self.i = 0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0

        self.get_logger().info(
            "Localization Publisher Node has started and is publishing at 2 Hz."
        )

    def timer_callback(self):
        """
        定时器回调函数，用于构建并发布 Odometry 消息。
        模拟机器人在一个圆周上移动。
        """

        # 1. 创建 Odometry 消息对象
        odom_msg = Odometry()

        # 2. 填充 Header（包含时间戳和坐标系）
        odom_msg.header = Header()
        odom_msg.header.stamp = self.get_clock().now().to_msg()  # 当前时间戳
        odom_msg.header.frame_id = "odom"  # 里程计坐标系

        # 3. 填充 child_frame_id（子坐标系/机器人基座）
        odom_msg.child_frame_id = "base_link"  # 机器人基座坐标系

        # 4. 模拟位置 (Pose.Pose.position)
        # 模拟在一个半径为 R 的圆周上移动
        R = 5.0
        angle = self.i * 0.01 * math.pi  # 角度随时间递增

        self.x = R * math.cos(angle)
        self.y = R * math.sin(angle)
        self.z = 0.0  # 保持 z 轴为 0

        odom_msg.pose.pose.position = Point(x=self.x, y=self.y, z=self.z)

        # 5. 模拟姿态 (Pose.Pose.orientation)
        # 这里的姿态为了简单，我们保持一个默认值（四元数 (0, 0, 0, 1) 表示无旋转）
        # 实际的定位通常需要计算航向角并转换为四元数
        odom_msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        # 6. 模拟线速度和角速度 (Twist.Twist)
        odom_msg.twist.twist.linear.x = 0.5  # 模拟恒定的线速度
        odom_msg.twist.twist.angular.z = 0.1  # 模拟恒定的角速度

        # 7. 发布消息
        self.publisher_.publish(odom_msg)

        # # 8. 打印日志和更新计数器
        # self.get_logger().info(
        #     f'Publishing localization: (x={self.x:.2f}, y={self.y:.2f})'
        # )

        self.i += 1

        # 防止计数器溢出，实际上不必要，但为了演示
        if self.i > 10000:
            self.i = 0


def main(args=None):
    # 初始化 ROS 2 Python 客户端库
    rclpy.init(args=args)

    # 实例化节点
    localization_publisher = LocalizationPublisher()

    # 循环运行节点，接收回调
    rclpy.spin(localization_publisher)

    # 程序被中断 (Ctrl+C) 时，销毁节点
    localization_publisher.destroy_node()

    # 关闭 ROS 2 Python 客户端库
    rclpy.shutdown()


if __name__ == "__main__":
    main()
