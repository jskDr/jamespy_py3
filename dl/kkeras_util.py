#from keras.models import Sequential
from keras.layers import Dense, Input
from keras.models import Model
from keras.regularizers import l1

import matplotlib.pyplot as plt

def plot_model_history( history):
	"""
	accuracy and loss are depicted.
	"""
	plt.plot(history.history['acc'])
	#plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	#plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss 
	plt.plot(history.history['loss']) 
	#plt.plot(history.history['val_loss']) 
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	#plt.legend(['train', 'test'], loc='upper left')
	plt.show()

def plot_history( history):
	"""
	accuracy and loss are depicted.
	"""
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	
	# summarize history for loss 
	plt.plot(history.history['loss']) 
	plt.plot(history.history['val_loss']) 
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

class Model_Ordinary( Model):
	"""
	Adaptive linear model based on Keras Model
	"""
	def __init__(self, X_shape_1):
		in_model = Input(shape=(X_shape_1,))
		out_model = Dense(1, activation='linear')(in_model)
		
		super().__init__(input = in_model, output=out_model)
		
		self.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])

class Model_Ordinary_Hidden( Model):
	"""
	Adaptive linear model based on Keras Model
	"""
	def __init__(self, X_shape_1, n_h_nodes):
		in_model = Input(shape=(X_shape_1,))
		hidden_l = Dense(n_h_nodes, activation='relu')(in_model)
		out_model = Dense(1, activation='linear')(hidden_l)
		
		super().__init__(input = in_model, output=out_model)
		
		self.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])


class Model_Lasso( Model):
	"""
	Adaptive linear model based on Keras Model
	"""
	def __init__(self, X_shape_1, alpha):
		in_model = Input(shape=(X_shape_1,))
		out_model = Dense(1, activation='linear', W_regularizer=l1(alpha))(in_model)
		
		super().__init__(input = in_model, output=out_model)
		
		self.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])

class Model_Lasso_Hidden( Model):
	"""
	Adaptive linear model based on Keras Model
	"""
	def __init__(self, X_shape_1, n_h_nodes, alpha):
		in_model = Input(shape=(X_shape_1,))
		hidden_l = Dense(n_h_nodes, activation='relu', W_regularizer=l1(alpha))(in_model)
		out_model = Dense(1, activation='linear', W_regularizer=l1(alpha))(hidden_l)
		
		super().__init__(input = in_model, output=out_model)
		
		self.compile(optimizer='adadelta', loss='mse', metrics=['accuracy'])