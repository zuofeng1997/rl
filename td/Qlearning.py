import numpy as np
import gym.spaces
from visdom import Visdom
env = gym.make("Taxi-v2")
state_space = 500
action_space = 6
epsilon = 0.1
q = np.zeros((state_space, action_space))
num_episode = 30000

viz = Visdom(env="mean_reward")


def choose_action(state, eps):
    if np.random.rand() > eps:
        return np.argmax(q[state])
    else:
        return np.random.randint(action_space)


def score(n_samples=100):
    rewards = []
    for _ in range(n_samples):
        observation = env.reset()
        cum_rewards = 0
        while True:
            action = choose_action(observation, 0)
            observation, reward, d, _ = env.step(action)
            cum_rewards += reward
            if d:
                rewards.append(cum_rewards)
                break
    return np.mean(rewards)


for i_episode in range(1, num_episode):
    s = env.reset()
    # if i_episode<0.8*num_episode:
    #     epsilon = 1-0.9*(i_episode/(0.8*num_episode))
    # else:
    #     epsilon = 0.1
    while True:
        a = choose_action(s, epsilon)
        s_, r, done, _ = env.step(a)

        target = r+np.max(q[s_])
        error = target-q[s][a]
        q[s][a] += 0.01*error
        if done:
            break
        s = s_

    if i_episode % 100 == 0:
        mean_reward = score()
        # viz.line(
        #     Y=np.array([mean_reward]),
        #     X=np.array([i_episode]),
        #     win="reward",
        #     opts=dict(
        #       title="qlearning"
        #     ),
        #     update=None if i_episode == 100 else 'append'
        # )
        print("i_episode:", i_episode, "mean_reward:", mean_reward, "epsilon:", epsilon)


