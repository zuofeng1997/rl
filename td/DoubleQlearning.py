import gym
import numpy as np
import matplotlib.pyplot as plt
env = gym.make("Taxi-v2")
state_space = 500
action_space = 6
epsilon = 0.1
num_episode = 30000
q = np.zeros((state_space,action_space))
q1 = np.zeros((state_space,action_space))
q2 = np.zeros((state_space,action_space))
r_normal = []
r_double = []
def choose_action(state,epsilon,flag=True):
    if flag:
        if np.random.rand()>epsilon:
            action = np.argmax(q[state])
        else:
            action = np.random.randint(action_space)
        return action
    else:
        if np.random.rand()>epsilon:
            action = np.argmax((q1[state]+q2[state])/2)
        else:
            action = np.random.randint(action_space)
        return action
def score(n_samples=100,flag=True):
    rewards = []
    for _ in range(n_samples):
        observation = env.reset()
        cum_rewards = 0
        while True:
            action = choose_action(observation,0,flag)
            observation, reward, done, _ = env.step(action)
            cum_rewards += reward
            if done:
                rewards.append(cum_rewards)
                break
    return np.mean(rewards)
#q learning
flag = True
for i_episode in range(num_episode):
    s = env.reset()
    while True:
        a = choose_action(s,epsilon,flag=True)
        s_,r,done,_ = env.step(a)
        if done:
            target = r
        else:
            target = r + np.max(q[s_])
        td_error = target - q[s][a]
        q[s][a] += 0.01 * td_error
        s = s_
        if done:
            break

    if i_episode%100==0:
        mean_reward = score(flag=True)
        r_normal.append(mean_reward)
        print("i_episode:",i_episode,"mean_reward:",mean_reward,"epsilon:",epsilon)
#double qlearning
num_episode = 60000
flag = False
for i_episode in range(num_episode):
    s= env.reset()
    while True:
        a = choose_action(s,epsilon,flag=False)
        s_,r,done,_ = env.step(a)
        if done:
            target_q1 = r
            target_q2 = r
        else:
            target_q1 = r + q2[s_][np.argmax(q1[s_])]
            target_q2 = r + q1[s_][np.argmax(q2[s_])]
        if i_episode % 2==0:
            q1[s][a] += 0.01*(target_q1-q1[s][a])
        else:
            q2[s][a] += 0.01*(target_q2-q2[s][a])
        if done:
            break
        s = s_
    if i_episode % 200 == 0:
        mean_reward = score(flag=False)
        r_double.append(mean_reward)
        print("i_episode:", i_episode, "mean_reward:", mean_reward, "epsilon:", epsilon)

plt.plot(r_normal,label="Qlearning")
plt.legend()
plt.plot(r_double,label="doubleQlearning")
plt.legend()
plt.show()