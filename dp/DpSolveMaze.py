from Maze_env import Maze
import numpy as np

env = Maze()
value = np.zeros((4, 4))  # init state value
A = [(-1, 0), (1, 0), (0, 1), (0, -1)]   # action set
MAP = [[0, 0, 0, 0],    # maze map,2 is goal,1 is hole,env will reset when you approach 1
       [0, 0, 1, 0],
       [0, 1, 2, 0],
       [0, 0, 0, 0]]


def move_help(n, minn, maxn):
    return max(min(maxn - 1, n), minn)


def move(s,a):   #move function
    if MAP[s[0]][s[1]] ==1 or MAP[s[0]][s[1]] ==2:
        return(0,0)
    return (move_help(s[0] + a[0], 0, 4), move_help(s[1] + a[1], 0, 4))


def reward(s):  #reward function
    return MAP[s[0]][s[1]] == 2


#value iter
while True:
    error = 0
    for i in range(4):
        for j in range(4):
            old_value = value[i,j]
            lookhead_values = []
            for action in range(4):
                new_state = move((i,j),A[action])
                evaluate_value = 0.99*value[new_state[0]][new_state[1]] + reward((i,j))
                lookhead_values.append(evaluate_value)
            value[i][j] = np.max(lookhead_values)
            error = np.max([error, np.sum(np.abs(old_value - value[i][j]))])
    if error < 1e-4:
        break

#policy iter

#1 policy evaluation
# while True:
#     policy = dict()
#     for i in range(4):
#         for j in range(4):
#             next_values = []
#             for action in range(4):
#                 new_state = move((i, j), A[action])
#                 next_values.append(value[new_state[0]][new_state[1]])
#             policy[(i,j)] = np.argmax(next_values)
#     while True:
#         error = 0
#         for i in range(4):
#             for j in range(4):
#                 old_value = value[i, j]
#                 selected_action = policy[(i,j)]
#                 next_state = move((i,j),A[selected_action])
#                 value[i][j] = 0.9*value[next_state[0],next_state[1]]+reward((i,j))
#                 error = np.max([error, np.sum(np.abs(old_value - value[i][j]))])
#         if error < 1e-4:
#             break
#
#     # 2 policy improvement
#     policy_stable = True
#     for i in range(4):
#         for j in range(4):
#             next_values_for_new = []
#             for action in range(4):
#                 new_state = move((i, j), A[action])
#                 next_values_for_new.append(reward((i,j))+0.9*value[new_state[0],new_state[1]])
#             old_action = policy[(i,j)]   #select action fom old policy
#             greedy_action = np.argmax(next_values_for_new)
#             if greedy_action != old_action:
#                 policy_stable = False
#
#     if policy_stable == True:
#         break




print("find optimal policy")
optimal_greedy_policy = dict()

for i in range(4):
    for j in range(4):
        next_values = []
        for action in range(4):
            new_state = move((i, j), A[action])
            next_values.append(value[new_state[0]][new_state[1]])
        optimal_greedy_policy[(i, j)] = np.argmax(next_values)


# vadidate the optimal value policy
for _ in range(10):
    state = env.reset()
    while True:
        env.render()
        i,j= int((state[3]+5.0)/40)-1,int((state[2]+5.0)/40)-1
        optimal_action = optimal_greedy_policy[(i,j)]
        state,_,done = env.step(optimal_action)
        if done:
            break
