import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

'''
Q-Network
=> Q-Learning + Deep Learning 
'''

env = gym.make('FrozenLake-v0')

# set parameters
input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1
dis = 0.99
num_episodes = 2000

x = tf.placeholder(shape=[1, input_size], dtype=tf.float32)     # state input (1x16)
w = tf.Variable(tf.random_uniform([input_size, output_size], 0, 0.01))  # weight (16x4)

Qpred = tf.matmul(x,w)  # out Q prediction
y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)    # Y Label

# Cost Function
loss = tf.reduce_sum(tf.square(y-Qpred))
# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

def one_hot(x):
    return np.identity(16)[x:x+1]     # one hot (0~15 => 16)

# create list
rList = []

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        state = env.reset()
        e = 1. / ((i /50) + 10)
        rAll = 0
        done = False
        local_loss = []

        # Q-Networking Training
        while not done:
            Qs = sess.run(Qpred, feed_dict={x: one_hot(state)})
            # e-greedy
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(Qs)

            new_state, reward, done, info = env.step(action)
            if done:
                Qs[0, action] = reward
            else:
                Qs1 = sess.run(Qpred, feed_dict={x: one_hot(state)})
                Qs[0, action] = reward + dis * np.max(Qs1)

            sess.run(optimizer, feed_dict={x: one_hot(state), y: Qs})

            rAll += reward
            state = new_state

        rList.append(rAll)

print("Percent of successful episodes :" + str(sum(rList) / num_episodes) + "%")
plt.bar(range(len(rList)), rList, color="blue")
plt.show()