import math
import os
import random
import subprocess
import time
from os import path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

# 定义环境常量
GOAL_REACHED_DIST = 0.3  # 到达目标点的距离阈值
COLLISION_DIST = 0.35    # 碰撞检测的距离阈值
TIME_DELTA = 0.1         # 时间步长


# 检查随机目标位置是否位于障碍物上，如果是则不接受该位置
def check_pos(x, y):
    """
    检查给定坐标是否在有效区域内（不在障碍物上）
    参数:
        x, y: 要检查的坐标
    返回:
        goal_ok: 布尔值，True表示位置有效，False表示位置无效
    """
    goal_ok = True

    # 定义多个障碍物区域，如果目标在这些区域内则标记为无效
    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False

    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False

    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False

    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False

    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False

    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False

    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False

    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False

    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False

    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False

    # 检查是否超出环境边界
    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

    return goal_ok


class GazeboEnv:
    """Gazebo环境的基类，用于机器人导航强化学习"""

    def __init__(self, launchfile, environment_dim):
        """
        初始化Gazebo环境
        参数:
            launchfile: Gazebo启动文件名
            environment_dim: 激光雷达数据维度
        """
        self.environment_dim = environment_dim
        self.odom_x = 0  # 机器人x坐标
        self.odom_y = 0  # 机器人y坐标

        self.goal_x = 1  # 目标点x坐标
        self.goal_y = 0.0  # 目标点y坐标

        self.upper = 5.0  # 目标生成的上界
        self.lower = -5.0  # 目标生成的下界
        self.velodyne_data = np.ones(self.environment_dim) * 10  # 激光雷达数据，初始化为10
        self.last_odom = None  # 最新的里程计数据

        # 设置机器人状态
        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"  # 机器人模型名称
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        # 定义激光雷达的角度范围，将360度分成environment_dim个区间
        # 把激光雷达扫描的一个角度范围分成 environment_dim 个等宽的小区间，方便后续对每个区间提取特征（比如最小距离、均值等）
        # 激光雷达的扫描范围是[-np.pi / 2, np.pi / 2]，分成environment_dim个区间，每个区间的宽度是np.pi / self.environment_dim
        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]] # gaps 用来存储每个区间
        # self.gaps 最终是一个二维列表（list of lists）：
        # 外层列表：长度 = environment_dim，每个元素是一个区间
        # 内层列表：长度固定为 2，分别是区间起点角度、区间终点角度（单位：弧度）
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03 # 最后一个区间的终点角度增加0.03弧度，确保区间覆盖完整

        # 启动ROS核心
        port = "11311"
        subprocess.Popen(["roscore", "-p", port])
        print("Roscore launched!")

        # 使用给定的启动文件启动仿真
        rospy.init_node("gym", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")

        # 设置ROS发布者和订阅者
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)  # 速度控制发布者
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )  # 模型状态设置发布者
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)  # 恢复物理仿真服务
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)  # 暂停物理仿真服务
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)  # 重置世界服务
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)  # 目标点可视化发布者
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)  # 线速度可视化发布者
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)  # 角速度可视化发布者
        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        )  # 激光雷达数据订阅者
        self.odom = rospy.Subscriber(
            "/r1/odom", Odometry, self.odom_callback, queue_size=1
        )  # 里程计数据订阅者

    # 读取激光雷达点云数据，将其转换为距离数据，然后为每个角度范围选择最小值作为状态表示
    def velodyne_callback(self, v):
        """
        激光雷达回调函数，处理点云数据
        参数:
            v: 激光雷达点云消息
        """
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10  # 重置激光雷达数据
        
        for i in range(len(data)):
            if data[i][2] > -0.2:  # 只处理高度大于-0.2的点（过滤地面）
                # 计算点相对于机器人的角度
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                # 将距离数据分配到对应的角度区间
                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    def odom_callback(self, od_data):
        """
        里程计回调函数，更新机器人位置信息
        参数:
            od_data: 里程计消息
        """
        self.last_odom = od_data

    # 执行动作并读取新状态
    def step(self, action):
        """
        执行一步动作
        参数:
            action: 动作数组 [线速度, 角速度]
        返回:
            state: 当前状态
            reward: 奖励值
            done: 是否结束
            target: 是否到达目标
        """
        target = False

        # 发布机器人动作
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]  # 线速度
        vel_cmd.angular.z = action[1]  # 角速度
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)  # 发布可视化标记

        # 恢复物理仿真
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # 让仿真运行TIME_DELTA秒
        time.sleep(TIME_DELTA)

        # 暂停物理仿真
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # 读取激光雷达状态
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        # 从里程计数据计算机器人朝向
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)  # 获取yaw角

        # 计算机器人到目标的距离
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # 计算机器人朝向与目标方向之间的相对角度
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        # 【update: 2025-08-07】将角度限制在[-π, π]范围内
        theta = (theta + np.pi) % (2 * np.pi) - np.pi

        # 检测是否到达目标
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True

        # 构建状态向量：[激光雷达数据, 距离, 角度差, 线速度, 角速度]
        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        # target: 是否到达目标;collision: 是否碰撞;action: 当前动作;min_laser: 最小激光距离
        reward = self.get_reward(target, collision, action, min_laser)
        return state, reward, done, target

    def reset(self):
        """
        重置环境状态并返回初始观察
        返回:
            state: 初始状态
        """
        # 重置仿真世界
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()
        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        # 随机设置机器人初始位置和朝向
        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        # 在有效区域内随机选择机器人位置
        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # 在环境中设置随机目标点
        self.change_goal()
        # 随机放置障碍物盒子
        self.random_box()
        self.publish_markers([0.0, 0.0])

        # 恢复物理仿真
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        # 暂停物理仿真
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        
        # 构建初始状态
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        # 计算初始距离和角度
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        return state

    def change_goal(self):
        """
        改变目标点位置，确保新目标不在障碍物上
        """
        # 逐渐扩大目标生成范围，增加训练难度
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False

        # 在有效区域内随机生成新目标
        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x, self.goal_y)

    def random_box(self):
        """
        随机改变环境中盒子的位置，增加训练环境的随机性
        """
        for i in range(4):
            name = "cardboard_box_" + str(i)

            # 为每个盒子找到合适的位置
            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                # 确保盒子不会太靠近机器人或目标
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            
            # 设置盒子状态
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)

    def publish_markers(self, action):
        """
        在RViz中发布可视化数据
        参数:
            action: 当前动作
        """
        # 发布目标点标记
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)
        self.publisher.publish(markerArray)

        # 发布线速度标记
        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        # 发布角速度标记
        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        """
        从激光数据检测碰撞
        参数:
            laser_data: 激光雷达数据
        返回:
            done: 是否结束
            collision: 是否碰撞
            min_laser: 最小激光距离
        """
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        """
        计算奖励函数
        参数:
            target: 是否到达目标
            collision: 是否碰撞
            action: 当前动作
            min_laser: 最小激光距离
        返回:
            reward: 奖励值
        """
        if target:
            return 100.0  # 到达目标给予高奖励
        elif collision:
            return -100.0  # 碰撞给予高惩罚
        else:
            # 奖励函数：鼓励前进，惩罚转向，惩罚接近障碍物
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
