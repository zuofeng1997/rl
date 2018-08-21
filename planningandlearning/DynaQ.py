import numpy as np
import gym.spaces
from visdom import Visdom
import matplotlib.pyplot as plt
env = gym.make("Taxi-v2")
state_space = 500
action_space = 6

gamma = 0.95
epsilon = 0.1
alpha = 0.1

viz = Visdom(env="DynaQ")


def choose_action(state, eps, q):
    if np.random.binomial(1, eps) == 1:
        return np.random.randint(action_space)
    else:
        return np.argmax(q[state])


def score(q, n_samples=20):
    rewards = []
    for _ in range(n_samples):
        observation = env.reset()
        cum_rewards = 0
        while True:
            action = choose_action(observation, 0, q)
            observation, reward, d, _ = env.step(action)
            cum_rewards += reward
            if d:
                rewards.append(cum_rewards)
                break
    return np.mean(rewards)


class TrivialModel:
    def __init__(self):
        self.model = dict()

    def feed(self, s, a, s_, r):
        if s not in self.model.keys():
            self.model[s] = dict()
        self.model[s][a] = [s_, r]

    def sample(self):
        s = np.random.choice(list(self.model.keys()))
        a = np.random.choice(list(self.model[s].keys()))
        s_, r = self.model[s][a]
        return s, a, s_, r


def dyna_q(model, planning_steps, q):
    s = env.reset()
    while True:
        a = choose_action(s, epsilon, q)
        s_, r, done, _ = env.step(a)
        q[s][a] += alpha*(r+gamma*np.max(q[s_])-q[s][a])
        model.feed(s, a, s_, r)
        for i in range(planning_steps):
            e_s, e_a, e_s_, e_r = model.sample()
            q[e_s][e_a] += alpha * (e_r + gamma * np.max(q[e_s_]) - q[e_s][e_a])
        s = s_
        if done:
            break


def normal_run(steps, q, num_episode):
    all_r = []
    model = TrivialModel()
    for i_episode in range(1, num_episode):
        dyna_q(model, steps, q)
        if i_episode % 10 == 0:
            mean_reward = score(q=q)
            all_r.append(mean_reward)
            print("i_episode:", i_episode, "mean_reward:", mean_reward)
    return all_r


if __name__ == '__main__':
    all_steps = [32, 16, 8, 4, 2, 0]
    for step in all_steps:
        q_value = np.zeros((state_space, action_space))
        a_r = normal_run(step, q_value, 1000)
        plt.plot(a_r, label=step)
        plt.legend()
    viz.matplot(plt)

