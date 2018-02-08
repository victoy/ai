import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

'''
hmm.... this performance and accuracy are not good. 
'''
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
        self._build_net(name)

    def _build_net(self, name):
        with tf.variable_scope(name):
            self.training = tf.placeholder(tf.bool)

            # placeholders
            self.x = tf.placeholder(tf.float32, [None, 784])
            x_img = tf.reshape(self.x, [-1, 28, 28, 1])
            self.y = tf.placeholder(tf.float32, [None, 10])

            # Layer1 : use "tf.layers.conv2d()" instead of "tf.nn.conv2d()"
            l1 = tf.layers.conv2d(inputs=x_img, filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
            l1 = tf.layers.max_pooling2d(inputs=l1, pool_size=[2,2], strides=2, padding='SAME')
            l1 = tf.layers.dropout(inputs=l1, rate=0.7, training=self.training)

            # Layer2
            l2 = tf.layers.conv2d(inputs=l1, filters=64, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)
            l2 = tf.layers.max_pooling2d(inputs=l2, pool_size=[2,2], strides=2, padding='SAME')
            l2 = tf.layers.dropout(inputs=l2, rate=0.7, training=self.training)

            # Layer3
            l3 = tf.layers.conv2d(inputs=l2, filters=128, kernel_size=[3,3], padding='SAME', activation=tf.nn.relu)
            l3 = tf.layers.max_pooling2d(inputs=l3, pool_size=[2,2], strides=2, padding='SAME')
            l3 = tf.layers.dropout(inputs=l3, rate=0.7, training=self.training)

            # Dense layer with relu
            l4 = tf.reshape(l3, [-1, 128*4*4])
            l4 = tf.layers.dense(inputs=l4, units=625, activation=tf.nn.relu)  # 625 output
            l4 = tf.layers.dropout(inputs=l4, rate=0.5, training=self.training)

            # Logits (no activation) Layer: L5 Final FC 625 inputs -> 10 outputs
            self.logits = tf.layers.dense(inputs=l4, units=10)

        # cost/loss
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))

        # minimize
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # accuracy
        correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def predict(self, x_test, training=False):
        return self.sess.run(self.logits, feed_dict={self.x: x_test, self.training: training})

    def get_accuracy(self, x_test, y_test, training=False):
        return self.sess.run(self.accuracy, feed_dict={self.x: x_test, self.y: y_test, self.training: training})

    def train(self, x_data, y_data, training=True):
        return self.sess.run([self.cost, self.optimizer], feed_dict={self.x: x_data, self.y: y_data, self.training: training})

# Initialize
sess = tf.Session()
m1 = Model(sess, "m1")

sess.run(tf.global_variables_initializer())

print("Learning started.")

# train
for epoch in range(training_epochs):
    avg_cost = 0
    iteration = int(mnist.train.num_examples / batch_size)

    for i in range(iteration):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c, _ = m1.train(batch_xs, batch_ys)
        avg_cost += c / iteration

    print('Epoch: ', '%04d' % (epoch + 1), 'cost: ', '{:.9f}'.format(avg_cost))

print('Learning finished!')

# test and accuracy
print('Accuracy: ', m1.get_accuracy(mnist.test.images, mnist.test.labels))