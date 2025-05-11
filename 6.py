import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import time

# 构建模型环境
class CliffWalkingEnv:
    def __init__(self, c, r):
        self.ncol = c
        self.nrow = r
        self.x = 0
        self.y = r - 1
    def get_state(self):
        return self.x + self.y * self.ncol
    def step(self, act):
        change = [[0,-1],[0,1],[-1,0],[1,0]]
        self.x = max(0, min(self.ncol - 1, self.x + change[act][0]))
        self.y = max(0, min(self.nrow - 1, self.y + change[act][1]))
        reward = -1
        done = False
        if self.y == self.nrow - 1 and self.x > 0:
            if self.x != self.ncol - 1:
                reward = -100
            done = True
        return self.get_state(), reward, done
    def reset(self):
        self.x = 0
        self.y = self.nrow - 1
        return self.get_state()

# 实现DynaQ算法
class DynaQ:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_planning, n_action, mode):
        self.Q_table = np.zeros([nrow * ncol, n_action])
        self.model = dict()  # 创建了一个空字典
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.n_planing = n_planning
        self.n_action = n_action
        if mode == 1:
            self.temp_n_planning = max(int(n_planning / 2), n_planning - 10)
        else:
            self.temp_n_planning = n_planning 
    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action
    def q_learning(self, s0, a0, r, s1):
        td_error = r + self.gamma * (self.Q_table[s1].max() - self.Q_table[s0][a0])
        self.Q_table[s0, a0] += self.alpha * td_error
    def update(self, s0, a0, r, s1):
        self.q_learning(s0, a0, r, s1)
        self.model[(s0,a0)] = r, s1
        for _ in range(self.temp_n_planning):
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_)
        self.temp_n_planning = min(self.temp_n_planning + 1, self.n_planing)


# 训练
def DynaQ_CliffWalking(n_planning, n_action, mode):
    ncol = 12
    nrow = 4
    env = CliffWalkingEnv(ncol, nrow)
    epsilon = 0.01
    alpha = 0.1
    gamma = 0.9
    agent = DynaQ(ncol, nrow, epsilon, alpha, gamma, n_planning, n_action, mode)
    num_episodes = 300  
    return_list = []  # 记录每一条序列的回报
    for i in range(10):  # 显示10个进度条
        # tqdm的进度条功能
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):  # 每个进度条的序列数
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    episode_return += reward  # 这里回报的计算不进行折扣因子衰减
                    agent.update(state, action, reward, next_state)
                    state = next_state
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list

np.random.seed(0)
random.seed(0)
# n_planning_list = [0, 2, 10, 20,100]
n_planning_list = [50,100]
n_action = 4
for n_planning in n_planning_list:
    print('Q-planning steaps is %d' % n_planning)
    return_list = DynaQ_CliffWalking(n_planning, n_action,0)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list,
             return_list,
             label=str(n_planning) + ' planning steps  mode 0')
for n_planning in n_planning_list:
    print('Q-planning steaps is %d' % n_planning)
    return_list = DynaQ_CliffWalking(n_planning, n_action,1)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list,
             return_list,
             label=str(n_planning) + ' planning steps  mode 1')
plt.legend()
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Dyna-Q on {}'.format('Cliff Walking'))
plt.show()