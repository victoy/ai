#!/usr/bin/python

"""
Let's assume the file name is "test-data.csv" and file format is csv file.
It could look like below.
===== File :  test-data.csv =====
#EXAME1,EXAME2,EXAME3,FINAL
73,80,75,152
93,88,93,185
89,91,90,180
96,98,100,196
73,66,70,142
53,46,55,101
"""

import tensorflow as tf
import numpy as np

tf.set_random_seed(777)  # for reproducibility

# Load the file with numpy
xy = np.loadtxt('csv/data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Make sure the shape and data are OK
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

# placeholders for a tensor that will be always fed.
x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.placeholder(tf.float32, shape=[None,1])

w = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1], name='bias'))

# Hypothesis
hypothesis = tf.matmul(x, w) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y))

# Mnimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch
sess = tf.Session()

# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={x: x_data, y:y_data}
    )
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\n Prediction :\n", hy_val)

# Ask my score
print("Your score will be ", sess.run(hypothesis, feed_dict={x: [[100, 70, 101]]}))
print("Orher score will be ", sess.run(hypothesis, feed_dict={x: [[60, 70, 110], [90, 100, 80]]}))

