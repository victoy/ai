import gym
from gym.envs.registration import register
import numpy as np
import matplotlib as plt

'''
Q-Learning with e-greedy

'''
register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4', 'is_slippery': False}
)

env = gym.make('FrozenLake-v3')

# Initialize
Q = np.zeros([env.observation_space.n, env.action_space.n])

# set parameters
dis = .99
num_episodes = 2000

# create lists
rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    # e-Greedy
    e = 1. / ((i // 100) + 1)  # // : divide with integral result (discard remainder)

    # The Q-Table learning Algorithm
    while not done:
        # choose an action by e greedy
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        new_state, reward, done, _ = env.step(action)

        Q[state, action] = reward + dis * np.max(Q[new_state, :])

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()