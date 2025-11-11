#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Twist
from std_msgs.msg import Header
import random


class LocalizationPublisher:
    def __init__(self):
        # 初始化 ROS 节点
        rospy.init_node("localization_publisher", anonymous=True)

        # 创建发布者，发布到 /localization 话题，消息类型是 Odometry
        self.publisher = rospy.Publisher("/localization", Odometry, queue_size=10)

        # 设置循环频率
        self.rate = rospy.Rate(10)  # 10Hz

    def create_odometry_message(self):
        # 创建一个 Odometry 消息
        odom_msg = Odometry()

        # 设置头部信息
        odom_msg.header = Header()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "odom"

        # 设置位置 (x, y, z)
        odom_msg.pose.pose.position = Point(
            random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0), 0.0
        )

        # 设置方向 (四元数表示)
        odom_msg.pose.pose.orientation = Quaternion(
            0.0, 0.0, random.uniform(-1.0, 1.0), 1.0
        )

        # 设置线性速度（x, y, z）
        odom_msg.twist.twist.linear.x = random.uniform(0.0, 1.0)
        odom_msg.twist.twist.linear.y = random.uniform(0.0, 1.0)
        odom_msg.twist.twist.linear.z = 0.0

        # 设置角速度（x, y, z）
        odom_msg.twist.twist.angular.x = random.uniform(-0.5, 0.5)
        odom_msg.twist.twist.angular.y = random.uniform(-0.5, 0.5)
        odom_msg.twist.twist.angular.z = random.uniform(-0.5, 0.5)

        return odom_msg

    def publish_odometry(self):
        # 发布 Odometry 消息
        while not rospy.is_shutdown():
            odom_msg = self.create_odometry_message()
            rospy.loginfo(f"Publishing Odometry message: {odom_msg}")
            self.publisher.publish(odom_msg)
            self.rate.sleep()  # 控制发布频率


if __name__ == "__main__":
    try:
        # 创建发布者对象并开始发布消息
        localization_publisher = LocalizationPublisher()
        localization_publisher.publish_odometry()  # 开始发布
    except rospy.ROSInterruptException:
        pass
