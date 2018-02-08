import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# x, y
x = tf.placeholder(tf.float32, [None, 784])
x_img = tf.reshape(x, [-1, 28, 28, 1])    # reshape image 28x28 color 1 (black&white) => -1 means None or N
y = tf.placeholder(tf.float32, [None, 10])

# L1 : weight -> convolution -> relu -> max pool
# image shape -> (?, 28, 28, 1)
w1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
# Conv  -> (?, 28, 28, 32)
# Pool  -> (?, 14, 14, 32)
l1 = tf.nn.conv2d(x_img, w1, strides=[1,1,1,1], padding='SAME')
l1 = tf.nn.relu(l1)
l1 = tf.nn.max_pool(l1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# L2 : weight -> convolution -> relu -> max pool
w2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
# Conv -> (?, 14, 14, 64)
# pool -> (?, 7, 7, 64)
l2 = tf.nn.conv2d(l1, w2, strides=[1,1,1,1], padding='SAME')
l2 = tf.nn.relu(l2)
l2 = tf.nn.max_pool(l2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
l2_flat = tf.reshape(l2, [-1, 7*7*64])

# output weight & bias & hypothesis
w3 = tf.get_variable("w3", shape=[7*7*64,10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(l2_flat, w3) + b

# cost/loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y))

# minimize
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# session initizlize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train model
print("Learning started. It takes sometime.")
for epoch in range(training_epochs):
    avg_cost = 0
    iteration = int(mnist.train.num_examples / batch_size)

    for i in range(iteration):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {x: batch_xs, y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / iteration

    print('Epoch: ', '%04d' % (epoch + 1), 'cost =', '{:.94}'.format(avg_cost))

print('Learning finished.')

# test model
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean((tf.cast(correct_prediction, tf.float32)))
print('Accuracy:', sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

# get one and predict
r = random.randint(0, mnist.test.num_examples -1 )
print('Label: ', sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
print('Prediction: ', sess.run(tf.argmax(hypothesis, 1), feed_dict={x: mnist.test.images[r:r+1]}))