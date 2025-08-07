import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from velodyne_env import GazeboEnv


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        策略网络（Actor），用于输出动作
        参数：
            state_dim: 状态空间维度
            action_dim: 动作空间维度
        """
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)  # 第一层，全连接
        self.layer_2 = nn.Linear(800, 600)        # 第二层，全连接
        self.layer_3 = nn.Linear(600, action_dim) # 输出层
        self.tanh = nn.Tanh()                     # 输出动作范围归一化

    def forward(self, s):
        """
        前向传播，输入状态s，输出动作a
        """
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


# TD3网络封装
class TD3(object):
    def __init__(self, state_dim, action_dim):
        """
        初始化TD3，主要用于加载和推理Actor网络
        """
        # 初始化Actor网络
        self.actor = Actor(state_dim, action_dim).to(device)

    def get_action(self, state):
        """
        根据当前状态获取动作
        参数：
            state: 当前状态（numpy数组）
        返回：
            动作（numpy数组）
        """
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def load(self, filename, directory):
        """
        加载已训练好的Actor网络参数
        参数：
            filename: 文件名
            directory: 文件夹路径
        """
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )


# 设置实验参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择cuda或cpu
seed = 0  # 随机种子
max_ep = 500  # 每个episode的最大步数
file_name = "TD3_velodyne"  # 策略模型文件名

# 创建测试环境
environment_dim = 20  # 激光雷达维度
robot_dim = 4         # 机器人自身状态维度
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)  # 启动仿真环境

# 等待环境初始化
time.sleep(5)

# 设置随机种子，保证实验可复现
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim  # 状态空间总维度
action_dim = 2                          # 动作空间维度

# 创建TD3网络实例
network = TD3(state_dim, action_dim)
try:
    network.load(file_name, "./pytorch_models")  # 加载训练好的模型参数
except:
    raise ValueError("Could not load the stored model parameters")

done = False
episode_timesteps = 0
state = env.reset()  # 环境重置，获取初始状态

# 开始测试循环
while True:
    action = network.get_action(np.array(state))  # 根据当前状态获取动作

    # 线速度归一化到[0,1]，角速度保持[-1,1]
    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)  # 执行动作，获得新状态和奖励
    done = 1 if episode_timesteps + 1 == max_ep else int(done)  # 达到最大步数也视为done

    # 回合结束，重置环境
    if done:
        state = env.reset()
        done = False
        episode_timesteps = 0
    else:
        state = next_state
        episode_timesteps += 1
