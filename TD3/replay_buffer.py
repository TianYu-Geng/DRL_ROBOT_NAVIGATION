"""
Data structure for implementing experience replay
Author: Patrick Emami
"""
import random
from collections import deque

import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        初始化经验回放缓冲区
        参数：
            buffer_size: 缓冲区最大容量
            random_seed: 随机种子，保证实验可复现
        说明：
            deque右侧为最新的经验
        """
        self.buffer_size = buffer_size  # 缓冲区最大容量
        self.count = 0                 # 当前缓冲区中经验数量
        self.buffer = deque()          # 使用双端队列存储经验
        random.seed(random_seed)       # 设置随机种子

    def add(self, s, a, r, t, s2):
        """
        向缓冲区添加一条新经验
        参数：
            s: 当前状态
            a: 执行动作
            r: 奖励
            t: 是否终止（done）
            s2: 下一个状态
        """
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)  # 缓冲区未满，直接添加
            self.count += 1
        else:
            self.buffer.popleft()           # 缓冲区已满，移除最旧的经验
            self.buffer.append(experience)  # 添加最新经验

    def size(self):
        """
        返回当前缓冲区中经验数量
        """
        return self.count

    def sample_batch(self, batch_size):
        """
        随机采样一批经验用于训练
        参数：
            batch_size: 采样的批量大小
        返回：
            s_batch, a_batch, r_batch, t_batch, s2_batch: 分别为状态、动作、奖励、终止标志、下一个状态的批量数据
        """
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)  # 缓冲区不足batch_size时，全部采样
        else:
            batch = random.sample(self.buffer, batch_size)  # 随机采样batch_size条经验

        # 分别提取每个字段，转换为numpy数组
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        """
        清空缓冲区
        """
        self.buffer.clear()
        self.count = 0
