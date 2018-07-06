from mc.OnPolicyMc import FiniteMcModel as MC
from mc.OffPolicyMc import FiniteMcModel as OffMc
import gym

env = gym.make("Taxi-v2")
num_episode = 1000000
action_space = 6
state_space = 500
mc = OffMc(action_space=action_space,state_space=state_space)
def run():
    for i_episode in range(1,num_episode):
        state = env.reset()
        ep = []
        while True:
            action = mc.choose_action(mc.behave,state)
            next_state,reward,done,_ = env.step(action)
            ep.append((state,action,reward))
            state = next_state
            if done:
                break

        mc.update_Q(ep)
        # if i_episode<0.8*num_episode:
        #     mc.epsilon = 1-0.9*(i_episode/(0.8*num_episode))
        # else:
        #     mc.epsilon = 0.1
        if i_episode%100==0:
            mean_reward = mc.score(env,mc.policy)
            print("i_episode:",i_episode,"mean_reward:",mean_reward,"epsilon:",mc.epsilon)
if __name__ == '__main__':
    run()
    # print(env.action_space,env.observation_space)