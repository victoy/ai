import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# parameters
learning_rate = 0.01
training_epochs = 15
batch_size = 100

class Model:

    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):
        with tf.variable_scope(self.name):
            # dropout (keep_prob) rate 0.7~0.5 for training but 1 for test.
            self.keep_prob = tf.placeholder(tf.float32)

            # input placeholders
            self.x = tf.placeholder(tf.float32, [None, 784])   # image 28x28x1 (black/white)
            x_img = tf.reshape(self.x, [-1, 28, 28, 1])
            self.y = tf.placeholder(tf.float32, [None, 10])

            # L1 ImgIn shape = (?, 28, 28, 1)
            #    Conv     -> (?, 28, 28, 32)
            #    Pool     -> (?, 14, 14, 32)
            w1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
            l1 = tf.nn.conv2d(x_img, w1, strides=[1,1,1,1], padding='SAME')
            l1 = tf.nn.relu(l1)
            l1 = tf.nn.max_pool(l1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            l1 = tf.nn.dropout(l1, keep_prob=self.keep_prob)

            # L2 ImgIn shape=(?, 14, 14, 32)
            #    Conv      ->(?, 14, 14, 64)
            #    Pool      ->(?, 7, 7, 64)
            w2 = tf.Variable(tf.random_normal([3,3,32,64], stddev=0.01))
            l2 = tf.nn.conv2d(l1, w2, strides=[1,1,1,1], padding='SAME')
            l2 = tf.nn.relu(l2)
            l2 = tf.nn.max_pool(l2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            l2 = tf.nn.dropout(l2, keep_prob=self.keep_prob)

            # L3 ImgIn shape=(?, 7, 7, 64)
            #    Conv      ->(?, 7, 7, 128)
            #    Pool      ->(?, 4, 4, 128)
            #    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
            w3 = tf.Variable(tf.random_normal([3,3,64,128], stddev=0.01))
            l3 = tf.nn.conv2d(l2, w3, strides=[1,1,1,1], padding='SAME')
            l3 = tf.nn.relu(l3)
            l3 = tf.nn.max_pool(l3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
            l3 = tf.nn.dropout(l3, keep_prob=self.keep_prob)
            l3_flat = tf.reshape(l3, [-1, 128*4*4])

            # L4 FC 4x4x128 inputs -> 625 outputs
            w4 = tf.get_variable("w4", shape=[128*4*4, 625], initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([625]))
            l4 = tf.nn.relu(tf.matmul(l3_flat, w4) + b4)
            l4 = tf.nn.dropout(l4, keep_prob=self.keep_prob)

            # L5 Final FC 625 inputs -> 10 outputs
            w5 = tf.get_variable("w5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([10]))
            self.logits = tf.matmul(l4, w5) + b5

        # cost/loss
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))

        # minimize
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # Test
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, keep_prob=1.0):
        return self.sess.run(self.logits, feed_dict= {self.x: x_test, self.keep_prob: keep_prob})

    def get_accuracy(self, x_test, y_test, keep_prob=1.0):
        return self.sess.run(self.accuracy, feed_dict={self.x: x_test, self.y: y_test, self.keep_prob: keep_prob})

    def train(self, x_data, y_data, keep_prob=0.7):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.x: x_data, self.y: y_data, self.keep_prob: keep_prob})


# initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())

print('Learning started!')

# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    iteration = int(mnist.train.num_examples / batch_size)

    for i in range(iteration):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / iteration

    print('Epoch: ', '%04d' % (epoch + 1), 'cost: ', '{:.9f}'.format(avg_cost))

# Test model and check accuracy
print('Accuracy: ', m1.get_accuracy(mnist.test.images, mnist.test.labels))