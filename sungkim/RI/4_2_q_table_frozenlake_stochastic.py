import gym
import numpy as np
import matplotlib.pyplot as plt

'''
Q-Learning in Non-deterministic environment
use one more Q and learning rate.  
'''

env = gym.make('FrozenLake-v0') # slippery is true
env.reset()

# Initialize
Q = np.zeros([env.observation_space.n, env.action_space.n])

# set parameters
learning_rate = 0.5
dis = .98
num_episodes = 2000

# create lists
rList = []
for i in range(num_episodes):
    state = env.reset()
    rAll = 0
    done = False

    while not done:
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n)/(i+1))

        new_state, reward, done, info = env.step(action)

        # Previous Q
        # Q[state, action] = reward + np.max(Q[new_state, :])
        # use one more Q
        Q[state, action] = (1-learning_rate) * Q[state, action] + learning_rate * (reward + np.max(Q[new_state, :]))
        state = new_state

        rAll += reward

    rList.append(rAll)

print("Score over time: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")
plt.show()