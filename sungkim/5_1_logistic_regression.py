import tensorflow as tf
import numpy as np

'''
About "set_random_seed". 
https://www.tensorflow.org/api_docs/python/tf/set_random_seed
When we repeat the session and use same sequeces of values. Otherwise, don't use this. 
'''
tf.set_random_seed(777)   # for reproducibility

xy = np.loadtxt('csv/data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

# Placeholders for a tensor that will be always fed.
x = tf.placeholder(tf.float32, shape=[None, 8])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(x,w))
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

# Cost / Loss function
cost = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis > 0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={x: x_data, y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy Report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={x: x_data, y: y_data})

    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
