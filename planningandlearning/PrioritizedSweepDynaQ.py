import heapq
import numpy as np
import gym.spaces
from planningandlearning.DynaQ import TrivialModel, choose_action, score, normal_run
from visdom import Visdom
import matplotlib.pyplot as plt
import itertools
env = gym.make("Taxi-v2")
state_space = 500
action_space = 6

gamma = 0.95
epsilon = 0.1
alpha = 0.1

viz = Visdom(env="DynaQ")


class PriorityQueue:
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = '<removed-task>'
        self.counter = itertools.count()

    def add_item(self, item, priority=0):
        if item in self.entry_finder:
            self.remove_item(item)
        count = next(self.counter)
        entry = [priority, count, item]

        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)

    def remove_item(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED

    def pop_task(self):
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return item, priority
        raise KeyError('pop from an empty priority queue')

    def empty(self):
        return not self.entry_finder


class PriorityModel(TrivialModel):
    def __init__(self):
        TrivialModel.__init__(self)
        self.priorityQueue = PriorityQueue()
        self.predecessors = dict()

    def insert(self, priority, state, action):
        self.priorityQueue.add_item((state, action), -priority)

    def empty(self):
        return self.priorityQueue.empty()

    def sample(self):
        (state, action), priority = self.priorityQueue.pop_task()
        new_state, reward = self.model[state][action]
        return -priority, state, action, new_state, reward

    def feed(self, current_state, action, new_state, reward):
        TrivialModel.feed(self, current_state, action, new_state, reward)
        if new_state not in self.predecessors.keys():
            self.predecessors[new_state] = set()
        self.predecessors[new_state].add((current_state, action))

    def predecessor(self, state):
        if state not in self.predecessors.keys():
            return []
        predecessors = []
        for statePre, actionPre in list(self.predecessors[state]):
            predecessors.append([statePre, actionPre, self.model[statePre][actionPre][1]])
        return predecessors


def priority_dyna_q(model, planning_steps, q):
    s = env.reset()
    while True:
        a = choose_action(s, epsilon, q)
        s_, r, done, _ = env.step(a)
        model.feed(s, a, s_, r)
        q[s, a] += alpha * (r + gamma*np.max(q[s_]) - q[s, a])
        p = np.abs(r+gamma*np.max(q[s_])-q[s][a])
        if p > 0.001:
            model.insert(p, s, a)
        step = 0
        while step < planning_steps and not model.empty():
            step += 1
            _, e_s, e_a, e_s_, e_r = model.sample()
            q[e_s][e_a] += alpha * (e_r + gamma * np.max(q[e_s_]) - q[e_s][e_a])
            for pre_s, pre_a, pre_r in model.predecessor(e_s):
                p = np.abs(pre_r + gamma * np.max(q[e_s]) - q[pre_s][pre_a])
                if p > 0.001:
                    model.insert(p, pre_s, pre_a)
        s = s_
        if done:
            break


def priority_run(steps, q, num_episode):
    all_r = []
    model = PriorityModel()
    for i_episode in range(1, num_episode):
        priority_dyna_q(model, steps, q)
        if i_episode % 10 == 0:
            mean_reward = score(q=q)
            all_r.append(mean_reward)
            print("i_episode:", i_episode, "mean_reward:", mean_reward)
    return all_r


if __name__ == '__main__':
    q_value = np.zeros((state_space, action_space))
    a_r = priority_run(8, q_value, 1000)
    plt.plot(a_r, label="Prioritized")
    plt.legend()

    q_value = np.zeros((state_space, action_space))
    a_r = normal_run(8, q_value, 1000)
    plt.plot(a_r, label="Normal")
    plt.legend()
    viz.matplot(plt)

