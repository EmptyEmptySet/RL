import numpy 
import copy

import copy


class CliffWalkingEnv:
    """ 悬崖漫步环境"""
    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义网格世界的列
        self.nrow = nrow  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()

    def createP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]
        # 4种动作, change[0]:上,change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.nrow - 1 and j > 0:
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0,
                                                    True)]
                        continue
                    # 其他位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.nrow - 1 and next_x > 0:
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P

class PolicyIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.v = [0] * self.env.ncol * self.env.nrow
        self.pi = [[0.25, 0.25, 0.25, 0.25]] * self.env.ncol * self.env.nrow # 均等概率
    def make_qsa_list(self, s, mode):
        qsa_list = []
        for a in range(4):
            qsa = 0
            for res in self.env.P[s][a]:
                p, next_state, r, done = res
                qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                if mode == 1:
                    qsa_list.append(self.pi[s][a] * qsa)
                else:
                    qsa_list.append(qsa)
        return qsa_list
    def policy_evaluation(self):
        cnt = 1
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = self.make_qsa_list(s, 1)
                new_v[s] = sum(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            else:
                cnt = cnt + 1
        print(f"do {cnt} times policy_evaluation")
    def policy_improvement(self): 
        for s in range(self.env.ncol * self.env.nrow):
            qsa_list = self.make_qsa_list(s, 0)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)
            temppi = []
            for a in range(4):
                if qsa_list[a] == maxq:
                    temppi.append(1/cntq)
                else:
                    temppi.append(0)
            self.pi[s] = temppi
        print("finish policy improvement")
        return self.pi
    def policy_iteration(self): 
        while(1):
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)  
            new_pi = self.policy_improvement()
            if old_pi == new_pi: break

class ValueIteration:
    def __init__(self, env, theta, gamma):
        self.env = env
        self.theta = theta
        self.gamma = gamma
        self.v = [0] * self.env.ncol * self.env.nrow
        self.pi = [[0.25, 0.25, 0.25, 0.25]] * self.env.ncol * self.env.nrow # 均等概率
    def make_qsa_list(self, s, mode):
        qsa_list = []
        for a in range(4):
            qsa = 0
            for res in self.env.P[s][a]:
                p, next_state, r, done = res
                qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                if mode == 1:
                    qsa_list.append(self.pi[s][a] * qsa)
                else:
                    qsa_list.append(qsa)
        return qsa_list
    def value_iteration(self):
        cnt = 1
        while 1:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = self.make_qsa_list(s, 0)
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            else:
                cnt = cnt + 1
        print(f"do {cnt} times value_iteration")
        self.get_policy()
    def get_policy(self): 
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = self.make_qsa_list(s, 0)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)  
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]

def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("v is")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.ncol + j]), end=' ')
        print()

    print("pi is")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.ncol + j) in end:  # 目标状态
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()




env = CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>']
theta = 0.001
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47]) 
env = CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>']
theta = 0.001
gamma = 0.9
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])