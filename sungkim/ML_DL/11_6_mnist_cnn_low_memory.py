import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters
learning_rate = 0.01
training_epochs = 15
batch_size = 100

# dropout (keep_prob) rate 0.7~0.5 for training but should be 1 for testing.
keep_prob = tf.placeholder(tf.float32)

# x, y
x = tf.placeholder(tf.float32, [None, 784])
x_img = tf.reshape(x, [-1, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape=(?, 28, 28, 1)
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
w1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
l1 = tf.nn.conv2d(x_img, w1, strides=[1,1,1,1], padding='SAME')
l1 = tf.nn.relu(l1)
l1 = tf.nn.max_pool(l1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
l1 = tf.nn.dropout(l1, keep_prob=keep_prob)
'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)
Tensor("dropout/mul:0", shape=(?, 14, 14, 32), dtype=float32)
'''

# L2 ImgIn shape=(?, 14, 14, 32)
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
w2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
l2 = tf.nn.conv2d(l1, w2, strides=[1,1,1,1], padding='SAME')
l2 = tf.nn.relu(l2)
l2 = tf.nn.max_pool(l2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
l2 = tf.nn.dropout(l2, keep_prob=keep_prob)
'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("dropout_1/mul:0", shape=(?, 7, 7, 64), dtype=float32)
'''

# L3 ImgIn shape=(?, 7, 7, 64)
#    Conv      ->(?, 7, 7, 128)
#    Pool      ->(?, 4, 4, 128)
#    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
w3 = tf.Variable(tf.random_normal([3,3,64,128], stddev=0.01))
l3 = tf.nn.conv2d(l2, w3, strides=[1,1,1,1], padding='SAME')
l3 = tf.nn.relu(l3)
l3 = tf.nn.max_pool(l3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
l3 = tf.nn.dropout(l3, keep_prob=keep_prob)
l3 = tf.reshape(l3, [-1, 128*4*4])
'''
Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
'''

# L4 FC 4x4x128 inputs -> 625 outputs
w4 = tf.get_variable("w4", shape=[128*4*4, 625], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)
l4 = tf.nn.dropout(l4, keep_prob=keep_prob)
'''
Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
'''

# L5 Final FC 625 inputs -> 10 outputs
w5 = tf.get_variable("w5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(l4, w5) + b5
'''
Tensor("add_1:0", shape=(?, 10), dtype=float32)
'''

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y))

# minimize
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
print('Learning started.')
for epoch in range(training_epochs):
    avg_cost = 0
    iteration = int(mnist.train.num_examples / batch_size)

    for i in range(iteration):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        feed_dict = {x: batch_x, y: batch_y, keep_prob: 0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / iteration

    print('Epoch: ', '%04d'%(epoch + 1), 'Cost: ', '{:.9f}'.format(avg_cost))

print('Learning finished.')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(x_sample, y_sample, batch_size=512):
    """Run a minibatch accuracy op"""

    n = x_sample.shape[0]
    correct_sample = 0

    for i in range(0, n, batch_size):
        x_batch = x_sample[i: i + batch_size]
        y_batch = y_sample[i: i + batch_size]
        n_batch = x_batch.shape[0]

        feed = {
            x: x_batch,
            y: y_batch,
            keep_prob: 1
        }

        correct_sample += sess.run(accuracy, feed_dict=feed) * n_batch

    return correct_sample / n

print("\nAccuracy Evaluates")
print("-------------------------------")
print('Train Accuracy:', evaluate(mnist.train.images, mnist.train.labels))
print('Test Accuracy:', evaluate(mnist.test.images, mnist.test.labels))


# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), {x: mnist.test.images[r:r+1], keep_prob: 1}))