import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv

'''
evaluate() 是一个策略评估函数，它不会更新网络，
只用当前训练好的 network 跑若干回合（eval_episodes），计算平均奖励和平均碰撞率，用来衡量策略性能。
    network: 当前训练好的Actor网络
    epoch: 当前训练的轮数
    eval_episodes: 评估的回合数
'''
def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        count = 0
        state = env.reset()
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, _ = env.step(a_in)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    print("..............................................")
    return avg_reward


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        策略网络（Actor），输入状态，输出动作
        """
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)  # 第一层
        self.layer_2 = nn.Linear(800, 600)        # 第二层
        self.layer_3 = nn.Linear(600, action_dim) # 输出层
        self.tanh = nn.Tanh()                     # 输出归一化

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        价值网络（Critic），输入状态和动作，输出Q值
        TD3有两个独立的Critic分支
        """
        super(Critic, self).__init__()

        # 第一个Critic分支
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        # 第二个Critic分支
        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        # 第一个Critic分支
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        # 第二个Critic分支
        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2


# TD3网络封装
class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        """
        初始化TD3，包含Actor和两个Critic网络及其目标网络
        """
        # 初始化Actor网络和目标网络
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # 初始化Critic网络和目标网络
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        self.writer = SummaryWriter()  # tensorboard日志
        self.iter_count = 0

    def get_action(self, state):
        """
        根据当前状态获取动作
        """
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    # 训练循环
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=1,
        tau=0.005,
        policy_noise=0.2,  # discount=0.99
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        for it in range(iterations):
            # 从经验回放池采样一批数据
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)

            # 用actor_target预测下一个动作
            next_action = self.actor_target(next_state)

            # 给动作加噪声（策略平滑）
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # 用critic_target计算下一个状态-动作的Q值
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # 取两个Q值的较小者，防止过估计
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Bellman方程计算目标Q值
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # 用当前critic计算当前Q值
            current_Q1, current_Q2 = self.critic(state, action)

            # 计算损失
            loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            # Critic网络反向传播
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            # 每隔policy_freq步更新一次Actor和目标网络
            if it % policy_freq == 0:
                # Actor损失为-Q，梯度上升
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # 软更新Actor目标网络
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # 软更新Critic目标网络
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )

            av_loss += loss
        self.iter_count += 1
        # 写入tensorboard
        self.writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )


# 设置训练参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择cuda或cpu
seed = 0  # 随机种子
eval_freq = 5e3  # 每隔多少步评估一次
max_ep = 500  # 每个episode最大步数
eval_ep = 10  # 每次评估的回合数
max_timesteps = 5e6  # 总训练步数
expl_noise = 1  # 初始探索噪声
expl_decay_steps = (
    500000  # 探索噪声衰减步数
)
expl_min = 0.1  # 最小探索噪声
batch_size = 40  # mini-batch大小
discount = 0.99999  # 折扣因子
tau = 0.005  # 软更新系数
policy_noise = 0.2  # 策略噪声
noise_clip = 0.5  # 噪声裁剪
policy_freq = 2  # Actor更新频率
buffer_size = 1e6  # 经验池最大容量
file_name = "TD3_velodyne"  # 模型文件名
save_model = True  # 是否保存模型
load_model = False  # 是否加载已有模型
random_near_obstacle = True  # 是否在障碍物附近随机动作

# 创建结果和模型保存文件夹
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")

# 创建训练环境
environment_dim = 20  # 激光雷达维度
robot_dim = 4         # 机器人自身状态维度
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)
torch.manual_seed(seed)
np.random.seed(seed)
state_dim = environment_dim + robot_dim  # 状态空间总维度
action_dim = 2                          # 动作空间维度
max_action = 1

# 创建TD3网络和经验回放池
network = TD3(state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer(buffer_size, seed)
if load_model:
    try:
        network.load(file_name, "./pytorch_models")
    except:
        print(
            "Could not load the stored model parameters, initializing training with random parameters"
        )

# 创建评估结果存储
evaluations = []

timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1

count_rand_actions = 0
random_action = []

# 开始训练主循环
while timestep < max_timesteps:

    # 回合结束，训练网络
    if done:
        if timestep != 0:
            network.train(
                replay_buffer,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
                policy_freq,
            )

        # 定期评估并保存模型
        if timesteps_since_eval >= eval_freq:
            print("Validating")
            timesteps_since_eval %= eval_freq
            evaluations.append(
                evaluate(network=network, epoch=epoch, eval_episodes=eval_ep)
            )
            network.save(file_name, directory="./pytorch_models")
            np.save("./results/%s" % (file_name), evaluations)
            epoch += 1

        state = env.reset()
        done = False

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # 探索噪声衰减
    if expl_noise > expl_min:
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

    # 选取动作并加噪声
    action = network.get_action(np.array(state))
    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )

    # 如果靠近障碍物，随机采取一段时间的随机动作，增强探索
    if random_near_obstacle:
        if (
            np.random.uniform(0, 1) > 0.85
            and min(state[4:-8]) < 0.6
            and count_rand_actions < 1
        ):
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.random.uniform(-1, 1, 2)

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action
            action[0] = -1

    # 线速度归一化到[0,1]，角速度保持[-1,1]
    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target = env.step(a_in)
    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    episode_reward += reward

    # 保存经验到回放池
    replay_buffer.add(state, action, reward, done_bool, next_state)

    # 更新计数器和状态
    state = next_state
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# 训练结束后，评估并保存模型
evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    network.save("%s" % file_name, directory="./pytorch_models")
np.save("./results/%s" % file_name, evaluations)
