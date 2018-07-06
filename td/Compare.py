import numpy as np
import gym
import matplotlib.pyplot as plt
env = gym.make("Taxi-v2")
state_space = 500
action_space = 6
epsilon = 0.1
q = np.zeros((state_space,action_space))
num_episode = 50000
r_qlearning = []
r_sarsa = []
def choose_action(state,epsilon):
    if np.random.rand()>epsilon:
        return np.argmax(q[state])
    else:
        return np.random.randint(action_space)


def score(n_samples=100):
    rewards = []
    for _ in range(n_samples):
        observation = env.reset()
        cum_rewards = 0
        while True:
            action = choose_action(observation,0)
            observation, reward, done, _ = env.step(action)
            cum_rewards += reward
            if done:
                rewards.append(cum_rewards)
                break
    return np.mean(rewards)




for i_episode in range(1,num_episode):
    s = env.reset()
    # if i_episode<0.8*num_episode:
    #     epsilon = 1-0.9*(i_episode/(0.8*num_episode))
    # else:
    #     epsilon = 0.1
    while True:
        a = choose_action(s,epsilon)
        s_,r,done,_ = env.step(a)
        if done:
            target = r
        else:
            target = r+np.max(q[s_])
        error = target-q[s][a]
        q[s][a] += 0.01*error
        if done:
            break
        s = s_

    if i_episode%100==0:
        mean_reward = score()
        r_qlearning.append(mean_reward)
        # print("i_episode:",i_episode,"mean_reward:",mean_reward,"epsilon:",epsilon)

q = np.zeros((state_space,action_space))
for i_episode in range(1,num_episode):
    # if i_episode<0.8*num_episode:
    #     epsilon = 1-0.9*(i_episode/(0.8*num_episode))
    # else:
    #     epsilon = 0.1
    s = env.reset()
    a = choose_action(s,epsilon)

    while True:
        s_,r,done,_ = env.step(a)
        a_ = choose_action(s_,epsilon)
        if done:
            target = r
        else:
            target = r+q[s_][a_]
        error = target-q[s][a]
        q[s][a] += 0.01*error
        if done:
            break
        s = s_
        a = a_

    if i_episode%100==0:
        mean_reward = score()
        r_sarsa.append(mean_reward)
        # print("i_episode:",i_episode,"mean_reward:",mean_reward,"epsilon:",epsilon)
plt.plot(r_qlearning,label="qlearning")
plt.legend()
plt.plot(r_sarsa,label="sarsa")
plt.legend()
plt.show()

