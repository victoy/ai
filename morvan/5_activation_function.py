#!/usr/bin/python

/**
https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/activation_functions_

- Neural Network 
	- Activation Functions
		+ tf.nn.relu(features, name=Nome)
		+ tf.nn.relu6(features, name=None)
		+ tf.nn.elu(features, name=None)
		+ tf.nn.softplus(features, name=None)
		+ tf.nn.softsign(features, name=None)
		+ tf.nn.dropout(x, keep_prob, noise_shape=No, seed=None, name=None)
*/


def add_layer(inputs, in_size, out_size, activation_function=None):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	Wx_plus_b = tf.matmul(inputs, Weights) + biases
	if activation_function is None:
		outputs = Wx_plus_b
	else:
		outputs = activation_function(Wx_plus_b)
	return outputs