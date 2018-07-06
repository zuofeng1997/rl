import numpy as np
import gym
env = gym.make("Taxi-v2")
state_space = 500
action_space = 6
epsilon = 0.1
q = np.zeros((state_space,action_space))
num_episode = 50000
n = 2

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

G = 0
for i_episode in range(1,num_episode):
    s = env.reset()
    a = choose_action(s,epsilon)
    nstep = []
    nstep.append([0,s,a])
    t = 0
    T = 1000000
    while True:

        if t<T:
            s_,r,done,_ = env.step(a)

            nstep.append([r,s_])
            if done:
                T = t+1
            else:
                a_ = choose_action(s_,epsilon)
                nstep[t+1].append(a_)


        tao = t-n+1
        G = 0
        if tao>=0:
            minT = np.min((tao+n,T))
            for j in range(tao+1,minT+1):
                G+=nstep[j][0]
            if tao+n<T:
                G+=G+q[nstep[tao+n][1]][nstep[tao+n][2]]

            q[nstep[tao][1]][nstep[tao][2]] +=0.01*(G-q[nstep[tao][1]][nstep[tao][2]])
        if tao==T-1:
            break
        s = s_
        a = a_

        t+=1
    if i_episode%100==0:
        mean_reward = score()
        print("i_episode:",i_episode,"mean_reward:",mean_reward,"epsilon:",epsilon)


