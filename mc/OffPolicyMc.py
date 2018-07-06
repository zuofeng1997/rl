from copy import deepcopy
import numpy as np

class FiniteMcModel:
    def __init__(self,state_space,action_space,gamma=0.99,epsilon=0.3):
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = None
        if isinstance(action_space,int):
            self.action_space = np.arange(action_space)
            actions = [0]*action_space
            self._act_rep = "list"
        else:
            self.action_space = action_space
            actions = {k:0 for k in action_space}
            self._act_rep = "dict"
        if isinstance(state_space,int):
            self.state_space = np.arange(state_space)
            self.Q = [deepcopy(actions) for _ in range(state_space)]
        else:
            self.state_space = state_space
            self.Q = {k:deepcopy(actions) for k in state_space}
        self.count = deepcopy(self.Q)
        self.C = deepcopy(self.Q)
    def policy(self,action,state):  #target policy
        if self._act_rep == "list":
            if action == np.argmax(self.Q[state]):
                return 1
            return 0
        elif self._act_rep == "dict":
            if action == max(self.Q[state], key=self.Q[state].get):
                return 1
            return 0
    def behave(self,action,state):   # behave policy
        return self.epsilon / len(self.action_space) + (1 - self.epsilon) * self.policy(action, state)

    def generate_returns(self,ep):
        G = {}  # return on state
        cumC = 0  # cumulative reward
        W = {}
        w = 1
        for tpl in reversed(ep):
            observation, action, reward = tpl
            G[(observation, action)] = cumC = reward + self.gamma * cumC
            self.C[observation][action] += w
            W[(observation,action)] = w
            if self.policy(action,observation) == 0:
                break
            w = w*(1/self.behave(action,observation))
        return G,W

    def choose_action(self, policy, state):
        probs = [policy(a, state) for a in self.action_space]
        return np.random.choice(self.action_space, p=probs)
    def update_Q(self,ep):
        G,W = self.generate_returns(ep)
        for s in G:
            state,action = s
            self.count[state][action] += 1
            self.Q[state][action] += (W[(state,action)]/self.C[state][action])*(G[s]-self.Q[state][action])

    def score(self,env,policy,n_samples=100):
        rewards = []
        for _ in range(n_samples):
            observation = env.reset()
            cum_rewards = 0
            while True:
                action = self.choose_action(policy, observation)
                observation, reward, done, _ = env.step(action)
                cum_rewards += reward
                if done:
                    rewards.append(cum_rewards)
                    break
        return np.mean(rewards)



