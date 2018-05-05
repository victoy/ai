import tensorflow as tf

'''
TODO 

didn't understand yet. need to see this again. 
'''
tf.set_random_seed(777)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# wight and bias
w1 = tf.Variable(tf.truncated_normal([784, 30]))
b1 = tf.Variable(tf.truncated_normal([1, 30]))
w2 = tf.Variable(tf.truncated_normal([30, 10]))
b2 = tf.Variable(tf.truncated_normal([1, 10]))

def sigma(p):
    return tf.div(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(-p)))

def sigma_prime(p):
    return sigma(p) * (1 - sigma(p))

# Forward prop
l1 = tf.add(tf.matmul(x, w1), b1)
a1 = sigma(l1)
l2 = tf.add(tf.matmul(a1, w2), b2)
y_pred = sigma(l2)

# diff
assert y_pred.shape.as_list() == y.shape.as_list()
diff = (y_pred - y)

# Back prop (chain rule)
d_l2 = diff * sigma_prime(l2)
d_b2 = d_l2
d_w2 = tf.matmul(tf.transpose(a1), d_l2)

d_a1 = tf.matmul(d_l2, tf.transpose(w2))
d_l1 = d_a1 * sigma_prime(l1)
d_b1 = d_l1
d_w1 = tf.matmul(tf.transpose(x), d_l1)

# updating network using gradients
learning_rate = 0.5
step = [
    tf.assign(w1, w1-learning_rate*d_w1),
    tf.assign(b1, b1-learning_rate*tf.reduce_mean(d_b1, reduction_indices=[0])),
    tf.assign(w2, w2-learning_rate*d_w2),
    tf.assign(b2, b2-learning_rate*tf.reduce_mean(d_b2, reduction_indices=[0]))
]

# Run and test the training process
acct_mat = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y,1))
acct_res = tf.reduce_mean(tf.cast(acct_mat, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(10)
    sess.run(step, feed_dict={x: batch_xs, y: batch_ys})
    if i % 1000 == 0:
        res = sess.run(acct_res, feed_dict={x: mnist.test.images[:1000], y: mnist.test.labels[:1000]})
        print(res)

cost = diff * diff
step = tf.train.GradientDescentOptimizer(0.1).minimize(cost)