import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    def __init__(self, K):
        self.size = K
        self.prob = np.random.uniform(low = 0.0, high = 1.0, size = K)
        self.best_index = np.argmax(self.prob)
        self.best_prob = self.prob[self.best_index]
    def step(self, k):
        if np.random.rand() < self.prob[k]:
            return 1
        else:
            return 0

def log_bandit(bandit):
    print(f"bandit size is {bandit.size}")
    print(f"the best arm is {bandit.best_index}, and it's probability is {bandit.best_prob}")

class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.regret = 0
        self.counts = np.zeros(bandit.size)
        self.regret_list = []
        self.action_list = []
    def update_regret(self, k):
        self.regret -= self.bandit.prob[k] - self.bandit.best_prob
        self.regret_list.append(self.regret)
    def run_one_step(self):
        raise NotImplementedError
    def run_steps(self, step_len):
        for _ in range(step_len):
            k = self.run_one_step()
            self.counts[k] += 1
            self.action_list.append(k)
            self.update_regret(k)

def log_solver(solver, solver_name):
    time_list = range(len(solver.regret_list))
    plt.plot(time_list, solver.regret_list)
    plt.xlabel("Time steps")
    plt.ylabel("Cumulative regrets")
    plt.title(solver_name)
    plt.show()

# 各种算法,继承自solver
class EpsilonGreedy(Solver):
    def __init__(self, bandit, epsilon, rate):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.rate = rate
        self.estimates = np.array([1.0] * self.bandit.size)
    def run_one_step(self):
        if np.random.rand() < self.epsilon:
            k = np.random.randint(0, self.bandit.size)
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        self.epsilon *= self.rate
        return k
    
class UCB(Solver):
    def __init__(self, bandit, coef):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.coef = coef
        self.estimates = np.array([1.0] * self.bandit.size)
    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(np.log(self.total_count)/(2 * (self.counts + 1)))
        k = np.argmax(ucb)
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r - self.estimates[k])
        return k

class ThompsonSampling(Solver):
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self.a_num = np.ones(bandit.size)
        self.b_num = np.ones(bandit.size)
    def run_one_step(self):
        samples = np.random.beta(self.a_num, self.b_num)
        k = np.argmax(samples)
        r = self.bandit.step(k)
        if r == 1:
            self.a_num[k] += 1
        else:
            self.b_num[k] += 1
        return k

# np.random.seed(20)
bandit = Bandit(20)
log_bandit(bandit)
epsilongreedy = EpsilonGreedy(bandit, 0.1, 0.999)
epsilongreedy.run_steps(10000)
log_solver(epsilongreedy, "epsilongreedy")
ucb = UCB(bandit, 0.5)
ucb.run_steps(10000)
log_solver(ucb, "UCB")
thompsonsampling = ThompsonSampling(bandit)
thompsonsampling.run_steps(10000)
log_solver(thompsonsampling, "thompsonsampling")

bandit = Bandit(100)
log_bandit(bandit)
epsilongreedy = EpsilonGreedy(bandit, 0.1, 0.99)
epsilongreedy.run_steps(10000)
log_solver(epsilongreedy, "epsilongreedy")
ucb = UCB(bandit, 0.5)
ucb.run_steps(10000)
log_solver(ucb, "UCB")
thompsonsampling = ThompsonSampling(bandit)
thompsonsampling.run_steps(10000)
log_solver(thompsonsampling, "thompsonsampling")
