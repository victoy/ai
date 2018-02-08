import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
# dropout (keep_prob) rate 0.7~0.5 on training, but should be q for testing
keep_prob = tf.placeholder(tf.float32)

# x, y
x = tf.placeholder(tf.float32, [None, 784])
x_img = tf.reshape(x, [-1, 28, 28, 1])  # 28x28 image (black & white)
y = tf.placeholder(tf.float32, [None, 10])

# l1 convolutions
# input image shape = (?, 28, 28, 1)
w1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
l1 = tf.nn.conv2d(x_img, w1, strides=[1,1,1,1], padding='SAME')
l1 = tf.nn.relu(l1)
l1 = tf.nn.max_pool(l1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
l1 = tf.nn.dropout(l1, keep_prob=keep_prob)

# l2 convolutions shape=(?, 14, 14, 32)
w2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
l2 = tf.nn.conv2d(l1, w2, strides=[1,1,1,1], padding='SAME')
l2 = tf.nn.relu(l2)
l2 = tf.nn.max_pool(l2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
l2 = tf.nn.dropout(l2, keep_prob=keep_prob)

# l3 convolutions  hape=(?, 7, 7, 64)
w3 = tf.Variable(tf.random_normal([3,3,64,128]))
#    Conv      ->(?, 7, 7, 128)
#    Pool      ->(?, 4, 4, 128)
#    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
l3 = tf.nn.conv2d(l2, w3, strides=[1,1,1,1], padding='SAME')
l3 = tf.nn.relu(l3)
l3 = tf.nn.max_pool(l3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
l3 = tf.nn.dropout(l3, keep_prob=keep_prob)
l3_flat = tf.reshape(l3, [-1, 128*4*4])

# l4 Fully connected 4x4x128 inputs -> 625 outputs
w4 = tf.get_variable("w4", shape=[128*4*4, 625], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
l4 = tf.nn.relu(tf.matmul(l3_flat, w4 + b4))
l4 = tf.nn.dropout(l4, keep_prob=keep_prob)

# l5 Final FC 625 inputs -> 10 outputs
w5 = tf.get_variable("w5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(l4, w5) + b5

# cost/loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

# minimize
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    iteration = int(mnist.train.num_examples / batch_size)

    for i in range(iteration):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {x: batch_xs, y: batch_ys, keep_prob: 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / iteration

    print('Epoch: ', '%04d' % (epoch + 1), 'cost: ', '{:.9f}'.format(avg_cost))

print('Learning finished.')

# test
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy: ', sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1}))

# get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={x: mnist.test.images[r:r+1], keep_prob: 1}))