import numpy as np
import random


LEARNING_RATE = 0.01
E_GREEDY = 0.1
size = 3
table_prob = dict()
class Env(object):
    def __init__(self):
        self.board = np.zeros((size, size))

    def step(self, action):
        self.board[action[0]][action[1]] = 1  # ai step
        blank_ioc = np.where(self.board == 0)
        if blank_ioc[0].shape==(0,):
            done=True
            reword=0
            return self.board,done,reword
        elif blank_ioc[0].shape==(1,):
            oppoment_step = (blank_ioc[0][0],blank_ioc[1][0])
        else:
            oppoment_step = random.sample([*zip(*blank_ioc)], 1)[0]
        self.board[oppoment_step[0]][oppoment_step[1]] = -1   # env step

        results = []
        #chech rows
        for i in range(size):
            results.append(np.sum(self.board[i,:]))

        #chech columns
        for i in range(size):
            results.append(np.sum(self.board[:,i]))

        results.append(0)
        for i in range(size):
            results[-1]+=self.board[i,i]
        results.append(0)
        for i in range(size):
            results[-1]+=self.board[i,2-i]
        done = False
        reword=0
        for result in results:
            if result==size:
                done = True
                reword=1
            if result==-size:
                done = True
                reword=0
        sum=np.sum(np.abs(self.board))
        if sum==9:
            done=True
            reword=0
        return self.board,done,reword

    def render(self):
        print(self.board)

    def reset(self):
        self.board[:, :] = 0
        return self.board

    def getHash(self):
        Hash = 0
        for i in self.board.reshape(size*size):
            Hash = Hash * size + i
        return Hash
# env = Env()
# for i in range(100):
#     env.reset()
#     while True:
#         state,done,reword = env.step([1,2])
#         if state not in table_prob.keys():
#             table_prob[state] = 0.5
#         if done:
#             break
def select_action(state):
    image_blank_ioc = np.where(state == 0)
    if image_blank_ioc[0].shape == (1,):
        all_actions = [(image_blank_ioc[0][0], image_blank_ioc[1][0])]
    else:
        all_actions = [*zip(*image_blank_ioc)]

    if np.random.rand()>E_GREEDY:
        is_learn = True
        image_next_state_prob = []
        for action_image in all_actions:
            env_image = Env()
            env_image.board = state.copy()
            env_image.step(action_image)
            hash_image = env_image.getHash()
            if hash_image not in table_prob.keys():
                table_prob[hash_image] = 0.5
            image_next_state_prob.append(table_prob[hash_image])
        max_ids = []
        max_prob = np.max(image_next_state_prob)
        for i in range(len(image_next_state_prob)):
            if image_next_state_prob[i]==max_prob:
                max_ids.append(i)

        best_id = random.sample(max_ids,1)[0]

        action = all_actions[best_id]
        return is_learn,action
    else:
        is_learn = False
        action = random.sample(all_actions,1)[0]

        return is_learn,action

reword_count = 0
for i in range(100000):
    E_GREEDY = E_GREEDY-(0.3/100000)*i
    env = Env()
    state = env.reset()
    while True:
        # print(hash)
        # print(env.board)
        is_learn ,action = select_action(state)
        v_before = env.getHash()
        if v_before not in table_prob.keys():
            table_prob[v_before] = 0.5
        state,done,reword = env.step(action)
        reword_count+=reword
        v_after = env.getHash()
        if v_after not in table_prob.keys():
            table_prob[v_after] = 0.5
        if is_learn:
            table_prob[v_before] = table_prob[v_before]+LEARNING_RATE*(table_prob[v_after]-table_prob[v_before])

        if done:
            if reword ==1:
                table_prob[env.getHash()] = 1
            break

    if i%1000 == 0:
        print("1000次中获胜次数",reword_count)
        reword_count=0


