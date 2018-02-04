import tensorflow as tf
import numpy as np

tf.set_random_seed(777)

# predictrin animal type based on various features
xy = np.loadtxt('csv/data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

nb_classes = 7  # 0~6

x = tf.placeholder(tf.float32, [None, 16])
y = tf.placeholder(tf.int32, [None, 1])
y_one_hot = tf.one_hot(y, nb_classes)
y_one_hot = tf.reshape(y_one_hot, [-1, nb_classes])

w = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes], name='bias'))

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(x,w) + b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)   # one-hot encoding using argmax
correct_prediction = tf.equal(prediction, tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        sess.run(optimizer, feed_dict={x: x_data, y: y_data})
        if step % 100 == 0:
            # loss, acc = sess.run([cost, accuracy], feed_dict={x: x_data, y: y_data})
            # print("Step: {:5}\t Loss: {:.3f}\t Acc: {:.2%}".format(step, loss, acc))
            loss, acc = sess.run([cost, accuracy], feed_dict={x: x_data, y: y_data})
            print("Step: ", step,  "Loss: ", loss, "Acc: ", acc)

    # check if we can predict
    pred = sess.run(prediction, feed_dict={x: x_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))