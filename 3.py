import numpy as np
import numpy.random as random

# MRP的模拟

MRP1 =  [
    [0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
    [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
MRP2 =  [
    [0.0, 0.2, 0.2, 0.2, 0.2, 0.2],
    [0.2, 0.0, 0.2, 0.2, 0.2, 0.2],
    [0.2, 0.2, 0.0, 0.2, 0.2, 0.2],
    [0.2, 0.2, 0.2, 0.0, 0.2, 0.2],
    [0.2, 0.2, 0.2, 0.2, 0.0, 0.2],
    [0.2, 0.2, 0.2, 0.2, 0.2, 0.0],
]

# 从start随机生成长len的采样
def do_sampling_mrp(MRP, start, len): 
    rows, columns = MRP.shape
    chain = [start]
    for _ in range(len - 1):
        p = random.random()
        for i in range(columns):
            if p <= MRP[start][i]:
                start = i
                break
            else :
                p = p - MRP[start][i]
        chain.append(start)
    return chain

def compute_return(chain, gamma, rewards):
    G = 0
    temp = 1
    for i in range(len(chain)):
        G = G + temp * rewards[chain[i]]
        temp = temp * gamma
    return G

# 价值函数
def compute_value(MRP, gamma, rewards): # gamma!=1
    rows, columns = MRP.shape
    I = np.eye(columns)
    rewards = np.array(rewards).reshape((-1, 1))  
    value =  np.dot(np.linalg.inv(I - gamma*MRP) , rewards)
    return value


# test

MRP1 = np.array(MRP1)
MRP2 = np.array(MRP2)
rewards = [-1, -2, -2, 10, 1, 0] 
gamma = 0.5

sam1 = do_sampling_mrp(MRP1, 0, 10)
print(sam1)
G = compute_return(sam1, gamma, rewards)
print(G)
V = compute_value(MRP1, gamma, rewards)
print(V)


# MDP的模拟
# MDP = (S, A, P, R, gamma)
def join(s1, s2):
    return str(s1) + '-' + s2 
# 给定MDP和策略pi，将其转化为MRP,最一般的方法
def MDP_to_MRP(MDP, pi):
    S, A, P, R, gamma = MDP 
    MRP = np.zeros((len(S),len(S)))
    rewards = []
    for i in range(len(S)):
        for j in range(len(S)):
            p = 0
            for k in range(len(A)):
                a = join(S[i],A[k])
                s = join(a,S[j])
                if s in P:
                    p = p + P[s] * pi[a]
            MRP[i][j] = p
        r = 0
        for j in range(len(A)):
            a = join(S[i],A[j])
            if a in R:
                r = r + R[a] * pi[a]    
        rewards.append(r)
    return MRP, rewards

# 蒙特卡罗方法计算MRP
# 对于MDP的取样  
# 因为可能遇到无法终止的情况，所以这里开始和讲义有些区别，即考虑4->4->4
# 在计算状态价值等的时候没啥区别，但多了几次动作，导致占用度量不完全一样
def do_sampling_mdp(MDP, pi, l):
    # start是随机的
    S, A, P, R, gamma = MDP 
    chain = []
    start = str(random.randint(0,len(S)))
    for _ in range(l):
        p = random.random()
        for i in range(len(A)):
            a = join(start, A[i])
            if a in pi:
                if p <= pi[a]:
                    r = R[a]
                    temp_a = A[i]
                    break
                else:
                    p = p - pi[a]
        p = random.random()
        for i in range(len(S)):
            s = join(a, S[i])
            if s in P:
                if p < P[s]:
                    nxt = S[i]
                    break
                else:
                    p = p - P[s]
        chain.append((start,temp_a,nxt,r))
        start = nxt
    return chain

def do_MC(MDP, chains, gamma):
    S, A, P, R, gamma = MDP 
    N = {i: 0 for i in S}
    V = {i: 0 for i in S}
    for chain in chains:
        G = 0
        for i in range(len(chain) - 1, -1, -1):
            (s, a, s_next, r) = chain[i]
            G = r + gamma * G
            N[s] = N[s] + 1
            V[s] = V[s] + (G - V[s]) / N[s]
    return V

# 在状态s，进行动作a的频率->占用度量
def occupancy(chains, t_s, t_a, gamma):
    l = len(chains[0])
    total_times = np.zeros(l)
    occur_times = np.zeros(l)
    for chain in chains:
        for i in range(len(chain)):
            (s, a, nxt, r) = chain[i]
            total_times[i] = total_times[i] + 1
            if t_s == s and t_a == a:
                occur_times[i] = occur_times[i] + 1
                
    rho = 0
    for i in range(l):
        rho = rho + gamma ** i *occur_times[i]/total_times[i]
    return rho * (1-gamma)
# test

S = ["0","1","2","3","4"]
A = ["stay_0", "goto_0", "goto_1", "goto_2", "goto_3", "goto_4", "pos_goto","stay_4"]  # 动作集合
P = {
    "0-stay_0-0": 1.0,
    "0-goto_1-1": 1.0,
    "1-goto_0-0": 1.0,
    "1-goto_2-2": 1.0,
    "2-goto_3-3": 1.0,
    "2-goto_4-4": 1.0,
    "3-goto_4-4": 1.0,
    "3-pos_goto-1": 0.2,
    "3-pos_goto-2": 0.4,
    "3-pos_goto-3": 0.4,
    "4-stay_4-4": 1,
}
R = {
    "0-stay_0": -1,
    "0-goto_1": 0,
    "1-goto_0": -1,
    "1-goto_2": -2,
    "2-goto_3": -2,
    "2-goto_4": 0,
    "3-goto_4": 10,
    "3-pos_goto": 1,
    "4-stay_4": 0,
}

Pi_1 = {
    "0-stay_0": 0.5,
    "0-goto_1": 0.5,
    "1-goto_0": 0.5,
    "1-goto_2": 0.5,
    "2-goto_3": 0.5,
    "2-goto_4": 0.5,
    "3-goto_4": 0.5,
    "3-pos_goto": 0.5,
    "4-stay_4": 1,
}

Pi_2 = {
    "0-stay_0": 0.6,
    "0-goto_1": 0.4,
    "1-goto_0": 0.3,
    "1-goto_2": 0.7,
    "2-goto_3": 0.5,
    "2-goto_4": 0.5,
    "3-goto_4": 0.1,
    "3-pos_goto": 0.9,
    "4-stay_4": 1,
}
gamma = 0.5 
MDP = (S, A, P, R, gamma)
MRP, rewards = MDP_to_MRP(MDP, Pi_1)
print(MRP)
print(rewards)
V = compute_value(MRP, gamma, rewards)
print(V)
chains_pi1 = []
chains_pi2 = []
for i in range(100):
    chains_pi1.append(do_sampling_mdp(MDP, Pi_1, 10))
    chains_pi2.append(do_sampling_mdp(MDP, Pi_2, 10))

V = do_MC(MDP, chains_pi1, gamma)
print(V)
V = do_MC(MDP, chains_pi2, gamma)
print(V)

print(occupancy(chains_pi1, "2", "goto_4", gamma))
print(occupancy(chains_pi2, "2", "goto_4", gamma))
print(occupancy(chains_pi1, "3", "pos_goto", gamma))
print(occupancy(chains_pi2, "3", "pos_goto", gamma))
print(occupancy(chains_pi1, "4", "stay_4", gamma))
print(occupancy(chains_pi2, "4", "stay_4", gamma))