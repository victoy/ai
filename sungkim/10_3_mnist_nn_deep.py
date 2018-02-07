import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data
'''
Make it deeper. (= adding more layers)
We could see overfitting when we add too many layers. 
'''
tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters
learning_rate = 0.001
training_epoch = 15
batch_size = 100

# input placeholders
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# weight and bias
w1 = tf.get_variable("w1", shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
l1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.get_variable("w2", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

w3 = tf.get_variable("w3", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
l3 = tf.nn.relu(tf.matmul(l2, w3) + b3)

w4 = tf.get_variable("w4", shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
l4 = tf.nn.relu(tf.matmul(l3, w4) + b4)

w5 = tf.get_variable("w5", shape=[512, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(l4, w5) + b5

# cost/loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=y))

# minimize
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initializer
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train
for epoch in range(training_epoch):
    avg_cost = 0
    iteration = int(mnist.train.num_examples / batch_size)

    for i in range(iteration):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {x: batch_xs, y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / iteration

    print('Epoch: ', '%04d' % (epoch + 1), 'cost: ', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy: ', sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

# get one and prection
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={x: mnist.test.images[r:r+1]}))