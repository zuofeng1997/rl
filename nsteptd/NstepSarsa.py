import numpy as np
import gym.spaces
import matplotlib.pyplot as plt
from visdom import Visdom

viz = Visdom(env="mean_reward")

env = gym.make("Taxi-v2")
state_space = 500
action_space = 6
epsilon = 0.1
q = np.zeros((state_space, action_space))
num_episode = 20000
n = 8


def choose_action(state, eps):
    if np.random.rand() > eps:
        return np.argmax(q[state])
    else:
        return np.random.randint(action_space)


def score(n_samples=100):
    rs = []
    for _ in range(n_samples):
        observation = env.reset()
        cum_rewards = 0
        while True:
            action = choose_action(observation, 0)
            observation, reward, d, _ = env.step(action)
            cum_rewards += reward
            if d:
                rs.append(cum_rewards)
                break
    return np.mean(rs)


states = []
actions = []
rewards = []


for i_episode in range(1, num_episode):
    del states[:]
    del actions[:]
    del rewards[:]
    s = env.reset()
    a = choose_action(s, epsilon)
    states.append(s)
    actions.append(a)
    rewards.append(0)
    t = 0
    T = float('inf')
    while True:
        t += 1
        if t < T:
            s, r, done, _ = env.step(a)
            states.append(s)
            rewards.append(r)
            if done:
                T = t
            else:
                a = choose_action(s, epsilon)
                actions.append(a)
        tao = t-n
        G = 0
        p = 1
        if tao >= 0:
            for j in range(tao+1, min((tao+n, T))+1):
                G += rewards[j]
            if tao+n < T:
                G += q[states[tao+n]][actions[tao+n]]

            q[states[tao]][actions[tao]] += 0.01*p*(G-q[states[tao]][actions[tao]])
        if tao == T-1:
            break

    if i_episode % 100 == 0:
        mean_reward = score()
        print("i_episode:", i_episode, "mean_reward:", mean_reward, "epsilon:", epsilon)




