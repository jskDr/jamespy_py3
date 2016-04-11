"""
Python code for artificial neural networks

In this file, control codes for fann2 are also included.
"""

import platform
if platform.system() == 'Linux':
	from pyfann import libfann
else: # such as OSX
	from fann2 import libfann

from collections import OrderedDict
import pandas as pd
from sklearn import cross_validation
from pylab import *


import jpandas as jpd
# in jchem, there is a code which generates an input file for fann.
import jchem
# advanced ann code is included.
import jutil


def gen_input_files_valid_overfit( At, yt, Av, yv):
	"""
	Validation is also considerd.
	At and yt are for training while Av, yv are for validation.
	Input files of ann_in.data and ann_run.dat are gerneated.
	The files are used in ann_aq.c (./ann_aq) 
	* Input: At, Av is matrix, yt, yv is vector
	"""

	print "For overfitting testing, a validation file includes desired value."

	const_no_of_output = 1 # Now, only 1 output is considerd.

	# in file
	no_of_set = At.shape[0]
	no_of_input = At.shape[1]
	with open("ann_in.data", "w") as f:
		f.write( "%d %d %d\n" % (no_of_set, no_of_input, const_no_of_output))
		for ix in range( no_of_set):
			for iy in range( no_of_input):
				f.write( "{} ".format(At[ix,iy]))
			f.write( "\n{}\n".format( yt[ix,0]))
		print("ann_in.data with {0} sets, {1} inputs is saved".format( no_of_set, no_of_input))

	# run file 
	no_of_set = Av.shape[0]
	no_of_input = Av.shape[1]
	with open("ann_run.data", "w") as f:
		f.write( "%d %d\n" % (no_of_set, no_of_input))
		for ix in range( no_of_set):
			for iy in range( no_of_input):
				f.write( "{} ".format(Av[ix,iy]))
			f.write( "\n{}\n".format( yv[ix,0]))
		print("ann_run.data with {0} sets, {1} inputs is saved with desired values".format( no_of_set, no_of_input))



def run_fann( num_hidden = 4, fname = "ann_ws496.net", fname_data_prefix = '', n_iter = 100, disp = True, graph = True):
	print "num_hidden =", num_hidden    
	
	fname_data_train = fname_data_prefix + "train_in.data"
	fname_data_test = fname_data_prefix + "test_in.data"

	connection_rate = 1
	learning_rate = 0.7
	num_input = 1024
	#num_hidden = 40
	num_output = 1

	desired_error = 0.0001
	max_iterations = 1
	iterations_between_reports = 1

	ann = libfann.neural_net()
	ann.create_sparse_array(connection_rate, (num_input, num_hidden, num_output))
	ann.set_learning_rate(learning_rate)
	ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC)
	ann.set_activation_function_output(libfann.LINEAR)

	# train_data is loaded
	train_data = libfann.training_data()
	train_data.read_train_from_file( fname_data_train)

	# test_data is loaded
	test_data = libfann.training_data()
	test_data.read_train_from_file( fname_data_test)
	train_mse = list()
	test_mse = list()
	for ii in range(n_iter):
		# Training is performed with training data
		ann.reset_MSE()
		ann.train_on_data(train_data, max_iterations, iterations_between_reports, desired_error)

		# Testing is performed with test data
		ann.reset_MSE()
		ann.test_data(train_data)
		mse_train = ann.get_MSE(); train_mse.append( mse_train)

		# Testing is performed with test data
		ann.reset_MSE()
		ann.test_data(test_data)
		mse_test = ann.get_MSE(); test_mse.append( mse_test)

		if disp: 
			print ii, "MSE of train, test", mse_train, mse_test

	ann.save( fname)

	# We show the results of ANN training with validation. 
	if graph:
		plot( train_mse, label = 'train')
		plot( test_mse, label = 'test')
		legend( loc = 1)
		xlabel('iteration')
		ylabel('MSE')
		grid()
		show()
	
	return train_mse, test_mse

def pd_run_fann( pdw, num_hidden = 4, fname = "ann_ws496.net", fname_data_prefix = '', n_iter = 100,  disp = True, graph = True):
	"""
	Uisng pandas, in order to manage data more professionally. 
	The results will be stacked on a pandas dataframe. 
	n_iter is included so as to control the iteration time which is fixed to be 100 previously.
	"""
	train_mse, test_mse = run_fann( num_hidden = num_hidden, fname = fname, 
		fname_data_prefix = fname_data_prefix, n_iter = n_iter, disp = disp, graph = graph)
	
	#If pdw is empty, we can make an empty dataframe
	# I found that if an empty array is used, the content can not be accesced later on in DataFrame().
	if pdw.shape != (0,0):
		#print pdw.mse
		mse_ext = pdw.mse.tolist()      
		mse_ext.extend( train_mse)
		mse_ext.extend( test_mse)

		no_ext = pdw.no.tolist()
		no_ext.extend( range( len(train_mse)))
		no_ext.extend( range( len(test_mse)))

		phase_ext = pdw.phase.tolist()
		phase_ext.extend( ['train'] * len( train_mse))
		phase_ext.extend( ['test'] * len( test_mse))    

		num_hidden_ext = pdw.num_hidden.tolist()
		num_hidden_ext.extend( [num_hidden] * (len( train_mse) + len( test_mse)))
	else:
		mse_ext = list()      
		mse_ext.extend( train_mse)
		mse_ext.extend( test_mse)

		no_ext = list()
		no_ext.extend( range( len(train_mse)))
		no_ext.extend( range( len(test_mse)))

		phase_ext = list()
		phase_ext.extend( ['train'] * len( train_mse))
		phase_ext.extend( ['test'] * len( test_mse))    

		num_hidden_ext = list()
		num_hidden_ext.extend( [num_hidden] * (len( train_mse) + len( test_mse)))        
	
	new_dw = OrderedDict( [('num_hidden', num_hidden_ext), ('phase', phase_ext), ('no', no_ext), ('mse', mse_ext)])
	new_pdw = pd.DataFrame( new_dw)
	
	return new_pdw

class RunFANN():
	def __init__(self, csv_name = 'sheet/ws496.csv', y_id = 'exp', n_iter = 100, graph = True, offline = False):

		fname_data_prefix = csv_name[:-4]

		if offline:
			self.fname_data_prefix = fname_data_prefix
		else:
			pdr = pd.read_csv( csv_name)

			xM = jpd.pd_get_xM( pdr)
			yV = jpd.pd_get_yV( pdr, y_id = y_id)

			if graph:
				hist( yV)
				show()

			self.kFold( xM, yV, fname_data_prefix = fname_data_prefix)

			self.n_iter = n_iter

	def kFold(self, xM, yV, fname_data_prefix = ''):
		"""
		training ad testing data are saved separately with respect to the associated arrays.
		"""
		kf = cross_validation.KFold( xM.shape[0], n_folds = 5, shuffle=True)
		for tr, te in kf:
				pass

		self.fname_data_prefix = fname_data_prefix
		fname_prefix_train = fname_data_prefix + 'train' + '_in.data'
		fname_prefix_test = fname_data_prefix + 'test' + '_in.data'
		jchem.gen_input_files( xM[ tr, :], yV[ tr, 0], fname_prefix_train[:-8])
		jchem.gen_input_files( xM[ te, :], yV[ te, 0], fname_prefix_test[:-8])

	def pd_run_fann( self, pdw, num_hidden = 4, fname = "ann_ws496.net", disp = True, graph = True):
		"""
		In order to match between the input file names defined in the initialization and used in training and testing phases,
		we use prefix now. 
		"""
		return pd_run_fann( pdw, num_hidden = num_hidden, fname = fname, fname_data_prefix = self.fname_data_prefix, n_iter = self.n_iter, disp = disp, graph = graph)

class RunFANN_Dual( RunFANN):
	def __init__(self, csv_name = 'sheet/ws496.csv', y_id = 'exp', n_iter = 100, graph = True, offline = False):
		"""
		If offline is true, there is no need to write data again. 
		We can use saved data from now and on.	
		Now, I make fname_data_prefix becomes the csv file name except the posfix of '.csv',
		which will be convenient to use later on. Hence, there is no need to define it. 
		The prefix value will be determined automatically. 
		"""

		fname_data_prefix = csv_name[:-4]

		if offline:
			self.fname_data_prefix = fname_data_prefix

		else:
			pdr = pd.read_csv( csv_name)

			xM_r = jpd.pd_get_xM( pdr, smiles_id = 'R-SMILES')
			xM_o = jpd.pd_get_xM( pdr, smiles_id = 'SMILES')
			xM = np.concatenate( [xM_r, xM_o], axis = 1)
			print "Shape of combined xM is", xM.shape

			yV = jpd.pd_get_yV( pdr, y_id = y_id)

			if graph:
				hist( yV)
				show()

			self.kFold( xM, yV, fname_data_prefix = fname_data_prefix)

			self.n_iter = n_iter