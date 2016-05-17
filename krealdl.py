# Sungjin Kim, 2016-5-7
# Python 3

from importlib import reload
import tensorflow as tf
import pandas as pd

# Import MINST data
import input_data

def multilayer_perceptron(_X, _weights, _biases):
	#Hidden layer with RELU activation
	layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) 
	#Hidden layer with RELU activation
	layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) 
	return tf.matmul(layer_2, _weights['out']) + _biases['out']

def conv_dl_all( h12_array, Nit):
	"""
	h12_array = [ (64,64), (32,32)]
	"""
	df_l = []
	for h1, h2 in h12_array:
		df = conv_dl_it( h1, h2, Nit)
		df_l.append( df)

	return pd.concat( df_l, ignore_index = True)

def conv_dl_it( h1, h2, Nit):
	"""
	h1: n_hidden_1
	h2: n_hidden_2
	Nit: N iterations for ensembling
	"""
	df_l = []
	for n in range(Nit):
		acc = conv_dl( h1, h2)
		df = pd.DataFrame([[n, h1, h2, acc]], columns = ['Unit', 'n_hidden_1','n_hidden_2','accuracy'])
		print( df)
		df_l.append( df)
	return pd.concat( df_l, ignore_index = True)

def conv_dl( n_hidden_1 = 256, n_hidden_2 = 256):
	mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

	# Parameters
	learning_rate = 0.001
	training_epochs = 15
	batch_size = 100
	display_step = 1

	# Network Parameters
	#n_hidden_1 = 64 # 1st layer num features
	#n_hidden_2 = 64 # 2nd layer num features
	n_input = 784 # MNIST data input (img shape: 28*28)
	n_classes = 10 # MNIST total classes (0-9 digits)

	# tf Graph input
	x = tf.placeholder("float", [None, n_input])
	y = tf.placeholder("float", [None, n_classes])

	# Create model

	# Store layers weight & bias
	weights = {
		'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
		'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
		'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
	}
	biases = {
		'b1': tf.Variable(tf.random_normal([n_hidden_1])),
		'b2': tf.Variable(tf.random_normal([n_hidden_2])),
		'out': tf.Variable(tf.random_normal([n_classes]))
	}

	# Construct model
	pred = multilayer_perceptron(x, weights, biases)

	# Define loss and optimizer
	# Softmax loss
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) 
	# Adam Optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 

	# Initializing the variables
	init = tf.initialize_all_variables()

	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)

		# Training cycle
		for epoch in range(training_epochs):
			avg_cost = 0.
			total_batch = int(mnist.train.num_examples/batch_size)
			# Loop over all batches
			for i in range(total_batch):
				batch_xs, batch_ys = mnist.train.next_batch(batch_size)
				# Fit training using batch data
				sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
				# Compute average loss
				avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
			# Display logs per epoch step
			if epoch % display_step == 0:
				print( "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

		print( "Optimization Finished!")

		# Test model
		correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
		# Calculate accuracy
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		acc = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
		print( "Accuracy:", acc)

	return acc

import input_data_r01
class RealDL():
	def __init__(self, 
		labels_transform = "half&half_with_one_hot"):

		#mnist = input_data_r01.read_data_sets_gray("/tmp/data/", labels_transform = labels_transform)
		#self.mnist = mnist
		self.labels_transform = labels_transform

	def run_all( self, h12_array, Nit):
		"""
		h12_array = [ (64,64), (32,32)]
		"""
		df_l = []
		for h1, h2 in h12_array:
			df = self.run_it( h1, h2, Nit)
			df_l.append( df)

		return pd.concat( df_l, ignore_index = True)

	def run_it( self, h1, h2, Nit):
		"""
		h1: n_hidden_1
		h2: n_hidden_2
		Nit: N iterations for ensembling
		"""
		df_l = []
		for n in range(Nit):
			acc = self.run( h1, h2)
			df = pd.DataFrame([[n, h1, h2, acc]], columns = ['Unit', 'n_hidden_1','n_hidden_2','accuracy'])
			print( df)
			df_l.append( df)
		return pd.concat( df_l, ignore_index = True)	

	def run( self, h1 = 256, h2 = 256):

		#mnist = self.mnist
		labels_transform = self.labels_transform
		mnist = input_data_r01.read_data_sets_gray("/tmp/data/", labels_transform = labels_transform)

		# Parameters
		learning_rate = 0.001
		training_epochs = 15
		batch_size = 100
		display_step = 1

		# Network Parameters
		n_hidden_1 = h1 # 1st layer num features
		n_hidden_2 = h2 # 2nd layer num features
		n_input = 784 # MNIST data input (img shape: 28*28)
		if labels_transform == "half&half_with_one_hot":
			n_classes = 2 # Transformed into 0, 1 classes (0..4 --> 0, 5..9 --> 1)
		else:
			n_classes = 10 # MNIST total classes (0-9 digits)

		# tf Graph input
		x = tf.placeholder("float", [None, n_input])
		y = tf.placeholder("float", [None, n_classes])

		# Create model
		def multilayer_perceptron(_X, _weights, _biases):
			#Hidden layer with RELU activation
			layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) 
			#Hidden layer with RELU activation
			layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) 
			return tf.matmul(layer_2, weights['out']) + biases['out']

		# Store layers weight & bias
		weights = {
			'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
			'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
			'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
		}
		biases = {
			'b1': tf.Variable(tf.random_normal([n_hidden_1])),
			'b2': tf.Variable(tf.random_normal([n_hidden_2])),
			'out': tf.Variable(tf.random_normal([n_classes]))
		}

		# Construct model
		pred = multilayer_perceptron(x, weights, biases)

		# Define loss and optimizer
		# Softmax loss
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) 
		# Adam Optimizer
		optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) 

		# Initializing the variables
		init = tf.initialize_all_variables()

		# Launch the graph
		with tf.Session() as sess:
			sess.run(init)

			# Training cycle
			for epoch in range(training_epochs):
				avg_cost = 0.
				total_batch = int(mnist.train.num_examples/batch_size)
				# Loop over all batches
				for i in range(total_batch):
					batch_xs, batch_ys = mnist.train.next_batch(batch_size)
					# Fit training using batch data
					sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
					# Compute average loss
					avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
				# Display logs per epoch step
				if epoch % display_step == 0:
					print( "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

			print( "Optimization Finished!")

			# Test model
			correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
			# Calculate accuracy
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			acc = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
			print( "Accuracy:", acc)

		return acc