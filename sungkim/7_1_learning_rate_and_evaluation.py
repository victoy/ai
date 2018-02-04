import tensorflow as tf

tf.set_random_seed(777)

'''
We weill separate training and test dataset from now on. 
And this example will show you how learning rate impact your result.
'''

x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# Evaluation our model using this test dataset
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]


x = tf.placeholder("float", [None, 3])
y = tf.placeholder("float", [None, 3])

w = tf.Variable(tf.random_normal([3, 3]))
b = tf.Variable(tf.random_normal([3]))

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(x,w) + b)

# cost function
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis=1))

# minimize
'''
1) when lr = 1.5
2) When lr = 1e-10
3) When lr = 0.1
'''
learning_rate = 0.1   #1.5  # 1e-10   # 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Correct prediction Test model
prediction = tf.argmax(hypothesis, 1)
is_correct = tf.equal(prediction, tf.argmax(prediction))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize Tensorflow variables
    sess.run(tf.global_variables_initializer())

    for step in range(201):
        cost_val, w_val, _ = sess.run([cost, w, optimizer], feed_dict={x: x_data, y: y_data})
        print(step, cost_val, w_val)

    # prediction
    print("prediction : ", sess.run([prediction], feed_dict={x:x_data, y:y_data}))
    # accuracy
    print("accuracy : ", sess.run([accuracy], feed_dict={x: x_data, y: y_data}))