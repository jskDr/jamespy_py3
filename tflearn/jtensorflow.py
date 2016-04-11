"""
This is a set of codes to use tensorlow for molecular machine learning. 
"""

import os
import pickle

import tensorflow as tf
import numpy as np

from tflearn import jinput_data

def already_pikle(filename, work_directory):
	"""Download the data from Yann's website, unless it's already here."""
	filepath = os.path.join(work_directory, filename)
	if os.path.exists(filepath):
		with open(filepath, "rb") as f:
			print('Successfully loaded:', filepath)
			return pickle.load(f)
	return None

def save_pickle(filename, work_directory, data_list):
	"""
	Save all data as a pickle form.
	If save is success, it returns True
	Otherwise, it returns False.
	"""
	if not os.path.exists(work_directory):
		os.mkdir(work_directory)
		filepath = os.path.join(work_directory, filename)
		with open(filepath, "wb") as f:
			pickle.dump(data_list, f)
			print('Successfully saved:', filepath)
			return True

	return False

# Create model
def multilayer_perceptron(_X, _weights, _biases):
	layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with RELU activation
	layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) #Hidden layer with RELU activation
	return tf.matmul(layer_2, _weights['out']) + _biases['out']

def define_multilayer(n_input, n_classes, n_hidden_list):
		# Store layers weight & bias
		n_hidden_1, n_hidden_2 = n_hidden_list
		
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
		
		return weights, biases

def ANN(mol_data, 
		n_hidden_list = [256, 256], learning_rate = 0.001,
		training_epochs = 15, batch_size = 100, display_step = 1):
		"""
		mol_data are data for learning, validation and testing.  
		"""
		n_input = mol_data.train.images.shape[1]
		n_classes = mol_data.train.labels.shape[1]
		
		print "n_input, n_classes =", n_input, n_classes 
		
		# tf Graph input
		x = tf.placeholder("float", [None, n_input])
		y = tf.placeholder("float", [None, n_classes])

		weights, biases = define_multilayer(n_input, n_classes, n_hidden_list)
		
		# Construct model
		pred = multilayer_perceptron(x, weights, biases)

		# Define loss and optimizer
		# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
		# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
		
		# For regression, other loss definintion and optimizatino tool are selected. 
		loss = tf.reduce_mean(tf.square(y - pred))
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
		train = optimizer.minimize(loss)
		
		# train = optimizer.minimize(loss)    
		
		# Initializing the variables
		init = tf.initialize_all_variables()

		avg_cost_train_list = list()
		avg_cost_val_list = list()
		with tf.Session() as sess:
				sess.run(init)
				
				for epoch in range(training_epochs):
						avg_cost_train, avg_cost_val = 0., 0.
						total_batch = int(mol_data.train.data_size/batch_size)
						
						# Loop over all batches
						for i in range(total_batch):
								batch_xs, batch_ys = mol_data.train.next_batch(batch_size)
								#print batch_xs.shape, batch_ys.shape
								# Fit training using batch data
								sess.run(train, feed_dict={x: batch_xs, y: batch_ys})               

								# Compute average loss
								avg_cost_train += sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})

						avg_cost_train /= total_batch								
						avg_cost_val = sess.run(loss, 
								feed_dict={x: mol_data.validation.images, y: mol_data.validation.labels})
								# Display logs per epoch step
						avg_cost_train_list.append( avg_cost_train)
						avg_cost_val_list.append( avg_cost_val)
						if epoch % display_step == 0:
								print "Epoch:", '%04d' % (epoch+1), "RMSE(train) =", "{:.9f}".format(np.sqrt(avg_cost_train))
								print "Epoch:", '%04d' % (epoch+1), "RMSE(validation) =", "{:.9f}".format(np.sqrt(avg_cost_val))

		avg_cost_list = [avg_cost_train_list, avg_cost_val_list]
		return avg_cost_list

def ANN_ridge(mol_data, 
		n_hidden_list = [256, 256], learning_rate = 0.001,
		training_epochs = 15, batch_size = 100, display_step = 1,
		alpha = 0.5):
		"""
		mol_data are data for learning, validation and testing.  
		"""
		n_input = mol_data.train.images.shape[1]
		n_classes = mol_data.train.labels.shape[1]
		
		print "n_input, n_classes =", n_input, n_classes 
		
		# tf Graph input
		x = tf.placeholder("float", [None, n_input])
		y = tf.placeholder("float", [None, n_classes])

		weights, biases = define_multilayer(n_input, n_classes, n_hidden_list)
		
		# Construct model
		pred = multilayer_perceptron(x, weights, biases)

		# Define loss and optimizer
		# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
		# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
		
		# For regression, other loss definintion and optimizatino tool are selected. 
		# alpha = 0.5 # this is for Ridge regularization 
		loss = tf.reduce_mean(tf.square(y - pred)) + alpha * (tf.reduce_sum( tf.square( weights['h1'])) 
			+ tf.reduce_sum( tf.square( weights['h2']))
			+ tf.reduce_sum( tf.square( weights['out'])))
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
		train = optimizer.minimize(loss)
		
		# train = optimizer.minimize(loss)    
		
		# Initializing the variables
		init = tf.initialize_all_variables()

		avg_cost_train_list = list()
		avg_cost_val_list = list()
		with tf.Session() as sess:
				sess.run(init)
				
				for epoch in range(training_epochs):
						avg_cost_train, avg_cost_val = 0., 0.
						total_batch = int(mol_data.train.data_size/batch_size)
						
						# Loop over all batches
						for i in range(total_batch):
								batch_xs, batch_ys = mol_data.train.next_batch(batch_size)
								#print batch_xs.shape, batch_ys.shape
								# Fit training using batch data
								sess.run(train, feed_dict={x: batch_xs, y: batch_ys})               

								# Compute average loss
								avg_cost_train += sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})
								
						avg_cost_train /= total_batch								
						avg_cost_val = sess.run(loss, 
								feed_dict={x: mol_data.validation.images, y: mol_data.validation.labels})
								# Display logs per epoch step
						avg_cost_train_list.append( avg_cost_train)
						avg_cost_val_list.append( avg_cost_val)
						if epoch % display_step == 0:
								print "Epoch:", '%04d' % (epoch+1), "RMSE(train) =", "{:.9f}".format(np.sqrt(avg_cost_train))
								print "Epoch:", '%04d' % (epoch+1), "RMSE(validation) =", "{:.9f}".format(np.sqrt(avg_cost_val))

		avg_cost_list = [avg_cost_train_list, avg_cost_val_list]
		return avg_cost_list		


def singlelayer_perceptron(_X, _weights, _biases):
	#layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with RELU activation
	#layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) #Hidden layer with RELU activation
	return tf.add( tf.matmul( _X, _weights['out']), _biases['out'])

def define_singlelayer(n_input, n_classes):
		# Store layers weight & bias
		# n_hidden_1, n_hidden_2 = n_hidden_list
		
		weights = {
				# 'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
				# 'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
				'out': tf.Variable(tf.random_normal([n_input, n_classes]))
		}
		biases = {
				# 'b1': tf.Variable(tf.random_normal([n_hidden_1])),
				# 'b2': tf.Variable(tf.random_normal([n_hidden_2])),
				'out': tf.Variable(tf.random_normal([n_classes]))
		}
		
		return weights, biases

def ANN_single(mol_data, 
		learning_rate = 0.001,
		training_epochs = 15, batch_size = 100, display_step = 1):
		"""
		mol_data are data for learning, validation and testing.  
		n_hidden list is removed from the argument list. 
		"""

		n_input = mol_data.train.images.shape[1]
		n_classes = mol_data.train.labels.shape[1]
		
		print "n_input, n_classes =", n_input, n_classes 
		
		# tf Graph input
		x = tf.placeholder("float", [None, n_input])
		y = tf.placeholder("float", [None, n_classes])

		weights, biases = define_singlelayer(n_input, n_classes)
		
		# Construct model
		pred = singlelayer_perceptron(x, weights, biases)

		# Define loss and optimizer
		# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
		# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
		
		# For regression, other loss definintion and optimizatino tool are selected. 

		# Construct a linear model
		#activation = tf.add(tf.mul(X, W), b)

		# Minimize the squared errors
		#cost = tf.reduce_sum(tf.pow(activation-Y, 2))/(2*n_samples) #L2 loss
		#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #Gradient descent

		#loss = tf.reduce_sum(tf.pow(y - pred, 2))
		loss = tf.reduce_mean(tf.pow(y - pred, 2))
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
		train = optimizer.minimize(loss)
		
		# Initializing the variables
		init = tf.initialize_all_variables()

		avg_cost_train_list = list()
		avg_cost_val_list = list()
		with tf.Session() as sess:
				sess.run(init)
				
				for epoch in range(training_epochs):
						avg_cost_train, avg_cost_val = 0., 0.
						total_batch = int(mol_data.train.data_size/batch_size)
						
						# Loop over all batches
						for i in range(total_batch):
								batch_xs, batch_ys = mol_data.train.next_batch(batch_size)
								#print batch_xs.shape, batch_ys.shape
								# Fit training using batch data
								sess.run(train, feed_dict={x: batch_xs, y: batch_ys})               

								# Compute average loss
								avg_cost_train += sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})
						
						avg_cost_train /= total_batch							
						avg_cost_val = sess.run(loss, 
								feed_dict={x: mol_data.validation.images, y: mol_data.validation.labels})
						avg_cost_train_list.append( avg_cost_train)
						avg_cost_val_list.append( avg_cost_val)

						# Display logs per epoch step
						if epoch % display_step == 0:
								print "Epoch:", '%04d' % (epoch+1), "RMSE(train) =", "{:.9e}".format(np.sqrt(avg_cost_train))
								print "Epoch:", '%04d' % (epoch+1), "RMSE(validation) =", "{:.9e}".format(np.sqrt(avg_cost_val))
								print "Weights ->", sess.run( weights['out'])

		avg_cost_list = [avg_cost_train_list, avg_cost_val_list]
		return avg_cost_list

def ANN_single_ridge(mol_data, 
		learning_rate = 0.001,
		training_epochs = 15, batch_size = 100, display_step = 1,
		alpha = 0.5):
		"""
		mol_data are data for learning, validation and testing.  
		n_hidden list is removed from the argument list. 
		"""

		n_input = mol_data.train.images.shape[1]
		n_classes = mol_data.train.labels.shape[1]
		
		print "n_input, n_classes =", n_input, n_classes 
		
		# tf Graph input
		x = tf.placeholder("float", [None, n_input])
		y = tf.placeholder("float", [None, n_classes])

		weights, biases = define_singlelayer(n_input, n_classes)
		
		# Construct model
		pred = singlelayer_perceptron(x, weights, biases)

		# Define loss and optimizer
		# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
		# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer
		
		# For regression, other loss definintion and optimizatino tool are selected. 

		# Construct a linear model
		#activation = tf.add(tf.mul(X, W), b)

		# Minimize the squared errors
		#cost = tf.reduce_sum(tf.pow(activation-Y, 2))/(2*n_samples) #L2 loss
		#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #Gradient descent

		#loss = tf.reduce_sum(tf.pow(y - pred, 2))
		loss = tf.reduce_mean(tf.pow(y - pred, 2)) + alpha * tf.reduce_sum( tf.square( weights['out']))
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
		train = optimizer.minimize(loss)
		
		# Initializing the variables
		init = tf.initialize_all_variables()

		avg_cost_train_list = list()
		avg_cost_val_list = list()
		with tf.Session() as sess:
				sess.run(init)
				
				for epoch in range(training_epochs):
						avg_cost_train, avg_cost_val = 0., 0.
						total_batch = int(mol_data.train.data_size/batch_size)
						
						# Loop over all batches
						for i in range(total_batch):
								batch_xs, batch_ys = mol_data.train.next_batch(batch_size)
								#print batch_xs.shape, batch_ys.shape
								# Fit training using batch data
								sess.run(train, feed_dict={x: batch_xs, y: batch_ys})               

								# Compute average loss
								avg_cost_train += sess.run(loss, feed_dict={x: batch_xs, y: batch_ys})
						
						avg_cost_train /= total_batch							
						avg_cost_val = sess.run(loss, 
								feed_dict={x: mol_data.validation.images, y: mol_data.validation.labels})
						avg_cost_train_list.append( avg_cost_train)
						avg_cost_val_list.append( avg_cost_val)

						# Display logs per epoch step
						if epoch % display_step == 0:
								print "Epoch:", '%04d' % (epoch+1), "RMSE(train) =", "{:.9e}".format(np.sqrt(avg_cost_train))
								print "Epoch:", '%04d' % (epoch+1), "RMSE(validation) =", "{:.9e}".format(np.sqrt(avg_cost_val))
								print "Weights ->", sess.run( weights['out'])

		avg_cost_list = [avg_cost_train_list, avg_cost_val_list]
		return avg_cost_list		

def run( mol_datasets): # this is testing code
		# Parameters are defined here. 
		# n_input = 784
		# n_classes = 1
		n_hidden_list = [256, 256]
		learning_rate = 0.01
		training_epochs = 1000
		batch_size = 100
		display_step = 10

		# Based on the defined parameters, ANN is called. 
		return ANN(mol_datasets, n_hidden_list, learning_rate, training_epochs, batch_size, display_step)
