import gym
import numpy as np
import tensorflow as tf

'''
Q-Network for cartpole game
poor result
'''

env = gym.make('CartPole-v0')

# Parameters
learning_rate = 1e-1  # e means exponential. 1*pow(10, -1)
input_size = env.observation_space.shape[0]  # size : 4
output_size = env.action_space.n           # size : 2
num_episodes = 5000
dis = 0.9

# Bias
x = tf.placeholder(tf.float32, [None, input_size], name="input_x")
# Weight
w1 = tf.get_variable("W1", shape=[input_size, output_size], initializer=tf.contrib.layers.xavier_initializer())

Qpred = tf.matmul(x, w1)

y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)

# Cost Function
loss = tf.reduce_sum(tf.square(y-Qpred))

# Minimize
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# create lists
step_history = []

# init session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(num_episodes):
    e = 1 / ((i / 10) + 1)  # for decaying e-greedy
    state = env.reset()
    step_count = 0
    done = False

    while not done:
        step_count += 1
        x1 = np.reshape(state, [1, input_size])

        Q = sess.run(Qpred, feed_dict={x: x1})
        # e-greedy
        if np.random.randn(1) < e:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q)

        next_state, reward, done, info = env.step(action)

        # Network
        if done:    # done in cartpole game means fail.  ( Terminal )
            Q[0, action] = -100     # return bad reward.
        else:       # Non-Terminal
            x_next = np.reshape(next_state, [1, input_size])
            # Obtain the Q' values by feeding the new state through our network
            Q_next = sess.run(Qpred, feed_dict={x: x_next})
            Q[0, action] = reward + dis * np.max(Q_next)

        sess.run(optimizer, feed_dict={x: x1, y: Q})
        state = next_state

    step_history.append(step_count)
    print("Episod: ", i, "steps:", step_count)
    # If last 10's avg steps are 500, it's good enough and break.
    if len(step_history) > 10 and np.mean(step_history[-10:]) >500:
        break

# Test above train
observation = env.reset()
reward_sum = 0
while True:
    env.render()

    x1 = np.reshape(observation, [1, input_size])
    Q = sess.run(Qpred, feed_dict={x: x1})
    action = np.argmax(Q)   # choose the best action

    observation, reward, done, info = env.step(action)
    reward_sum += reward
    if done:
        print("Total score", reward_sum)
        break