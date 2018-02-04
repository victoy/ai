import tensorflow as tf
import random
import matplotlib.pyplot as plt

tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10  # 0~9 numbers

# MNIST data image of shape 28 x 28 = 784
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, nb_classes])

w = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# Hypothesis (using softmax)
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

# cost /loss function
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

# minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

traing_epoch = 15
batch_size = 100

with tf.Session() as sess:
    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())

    # Training Cycle
    for epoch in range(traing_epoch):
        avg_cost = 0
        iteration = int(mnist.train.num_examples / batch_size)

        for i in range(iteration):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += c / iteration

        print('Epoch:', '%04d' % (epoch +1), 'cost = ', '{:.9f}'.format(avg_cost))

    print("Learning finished")

    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

    # get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1, 1])))
    print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={x: mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()