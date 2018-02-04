import tensorflow as tf
import numpy as np

tf.set_random_seed(777)
learning_rate = 0.1

x_data = [[0, 0],
          [0, 1],
          [1, 0],
          [1, 1]]
y_data = [[0],
          [1],
          [1],
          [0]]

x_data = np.array(x_data, dtype=np.float32)
y_data = np.array(y_data, dtype=np.float32)

x = tf.placeholder(tf.float32, [None, 2])
y = tf.placeholder(tf.float32, [None, 1])

w1 = tf.Variable(tf.random_normal([2, 2]), name='weight1')
b1 = tf.Variable(tf.random_normal([2]), name='bias1')
layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal([2, 1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1, w2) + b2)

# cost / loss function
cost = -tf.reduce_mean(y * tf.log(hypothesis) + (1-y) * tf.log(1 - hypothesis))

# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        sess.run(optimizer, feed_dict={x: x_data, y: y_data})
        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={x: x_data, y: y_data}), sess.run([w1, w2]))

    # Accuracy
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x: x_data, y: y_data})
    print("\n Hypothesis: ", h, "\n Correct: ", c, "\n Accuracy: ", a)
