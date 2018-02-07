import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

'''
To avoid of overfitting issue, we use dropout which is one of regularization.
'''
tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters
learning_rate = 0.001
training_epoch = 15
batch_size = 100

# placeholders
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# dropout (keep_prob) rate  0.7 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# wegith & bias layers using loop
layer = x
w_shapes = [[784,512], [512,512], [512,512], [512,512], [512,10]]
b_shapes = [[512], [512], [512], [512], [10]]
layer_length = len(w_shapes);
for i in range(0, layer_length):
    w_shape = w_shapes[i]
    b_shape = b_shapes[i]
    w = tf.get_variable("w"+`i`, shape=w_shape, initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal(b_shape))
    if i == layer_length - 1:
        layer = tf.matmul(layer, w) + b
    else:
        layer = tf.nn.relu(tf.matmul(layer, w) + b)
        layer = tf.nn.dropout(layer, keep_prob=keep_prob)

# cost/loss function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer, labels=y))

# minimize
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train
for epoch in range(training_epoch):
    avg_cost = 0
    iteration = int(mnist.train.num_examples / batch_size)

    for i in range(iteration):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {x: batch_xs, y: batch_ys, keep_prob: 0.7}  # dropout (keep_prob) must be less than 1 for training
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / iteration

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# test model and get accuracy
correct_prediction = tf.equal(tf.argmax(layer, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1}))  # dropout (keep_prob) must be 1 for test

# get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
print("Prediction: ", sess.run(tf.argmax(layer, 1), feed_dict={x: mnist.test.images[r:r + 1], keep_prob: 1}))
