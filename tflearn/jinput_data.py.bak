"""
This is a modified version of input_data.py by Google,
for generalization.  
"""
import tensorflow as tf

import numpy
import pandas as pd
import jpandas as jpd
import numpy as np
import random


def placeholder_inputs(batch_size, mnist_IMAGE_PIXELS):
	"""Generate placeholder variables to represent the the input tensors.

	These placeholders are used as inputs by the rest of the model building
	code and will be fed from the downloaded data in the .run() loop, below.

	Args:
		batch_size: The batch size will be baked into both placeholders.

	Returns:
		images_placeholder: Images placeholder.
		labels_placeholder: Labels placeholder.
	"""
	# Note that the shapes of the placeholders match the shapes of the full
	# image and label tensors, except the first dimension is now batch_size
	# rather than the full size of the train or test data sets.
	images_placeholder = tf.placeholder(tf.float32, 
							shape=(batch_size, mnist_IMAGE_PIXELS))
	labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
	return images_placeholder, labels_placeholder


def XY_split( X, Y, rate = 0):
	assert rate >= 0

	total_N = X.shape[0]
	train_N = int( total_N * (1.0 - rate))

	X1, X2 = np.split( X, [train_N])
	Y1, Y2 = np.split( Y, [train_N])

	return X1, Y1, X2, Y2	

def read_data_sets_csv( fname, validation_rate = 0, test_rate = 0, disp = False):
	class DataSets(object):
		pass
	data_sets = DataSets()

	pdr = pd.read_csv( fname)
	xM = jpd.pd_get_xM( pdr)
	yV = jpd.pd_get_yV( pdr, y_id = 'exp')

	X, Y = map( np.array, [xM, yV])
	assert X.shape[0] == Y.shape[0]

	if test_rate > 0:
		X, Y, X_test, Y_test = XY_split( X, Y, test_rate)
		data_sets.test = DataSet_CSV( X_test, Y_test, disp = disp)

	if validation_rate > 0:
		X, Y, X_val, Y_val = XY_split( X, Y, validation_rate)
		data_sets.validation = DataSet_CSV( X_val, Y_val, disp = disp)

	# If test_rate and validation_rate are both zero, 
	# all data is allocated to train dataset. 
	data_sets.train = DataSet_CSV( X, Y, disp = disp)

	return data_sets

def read_data_sets_mol_md( fname, validation_rate = 0, test_rate = 0, disp = False):
	class DataSets(object):
		pass
	data_sets = DataSets()

	pdr = pd.read_csv( fname)
	xM_fp = jpd.pd_get_xM( pdr)
	xM_key = jpd.pd_get_xM_MACCSkeys( pdr)
	xM_molw = jpd.pd_get_xM_molw( pdr)
	xM_molw = np.divide( xM_molw, np.std( xM_molw, axis = 0))

	xM_lasa = jpd.pd_get_xM_lasa( pdr)
	xM_lasa = np.divide( xM_lasa, np.std( xM_lasa, axis = 0))

	xM = np.concatenate( [xM_fp, xM_key, xM_molw, xM_lasa], axis = 1)

	yV = jpd.pd_get_yV( pdr, y_id = 'exp').A1
	yV = [1 if y > 0 else 0 for y in yV] # classification is performed

	X, Y = map( np.array, [xM, yV])
	assert X.shape[0] == Y.shape[0]

	if test_rate > 0:
		X, Y, X_test, Y_test = XY_split( X, Y, test_rate)
		data_sets.test = DataSet_CSV( X_test, Y_test, disp = disp)

	if validation_rate > 0:
		X, Y, X_val, Y_val = XY_split( X, Y, validation_rate)
		data_sets.validation = DataSet_CSV( X_val, Y_val, disp = disp)

	# If test_rate and validation_rate are both zero, 
	# all data is allocated to train dataset. 
	data_sets.train = DataSet_CSV( X, Y, disp = disp)

	data_sets.IMAGE_PIXELS = xM.shape[1]
	return data_sets

def read_data_sets_mol_sd( fname, validation_rate = 0, test_rate = 0, disp = False):
	class DataSets(object):
		pass
	data_sets = DataSets()

	pdr = pd.read_csv( fname)
	xM_fp = jpd.pd_get_xM( pdr)
	#xM_key = jpd.pd_get_xM_MACCSkeys( pdr)
	#xM_molw = jpd.pd_get_xM_molw( pdr)
	#xM_lasa = jpd.pd_get_xM_lasa( pdr)
	#xM = np.concatenate( [xM_fp, xM_key, xM_molw, xM_lasa], axis = 1)
	xM = xM_fp

	yV = jpd.pd_get_yV( pdr, y_id = 'exp').A1
	yV = [1 if y > 0 else 0 for y in yV] # classification is performed

	X, Y = map( np.array, [xM, yV])
	assert X.shape[0] == Y.shape[0]

	if test_rate > 0:
		X, Y, X_test, Y_test = XY_split( X, Y, test_rate)
		data_sets.test = DataSet_CSV( X_test, Y_test, disp = disp)

	if validation_rate > 0:
		X, Y, X_val, Y_val = XY_split( X, Y, validation_rate)
		data_sets.validation = DataSet_CSV( X_val, Y_val, disp = disp)

	# If test_rate and validation_rate are both zero, 
	# all data is allocated to train dataset. 
	data_sets.train = DataSet_CSV( X, Y, disp = disp)

	data_sets.IMAGE_PIXELS = xM.shape[1]
	return data_sets

def read_data_sets_mol_sd_molw( fname, validation_rate = 0, test_rate = 0, disp = False):
	class DataSets(object):
		pass
	data_sets = DataSets()

	pdr = pd.read_csv( fname)
	#xM_fp = jpd.pd_get_xM( pdr)
	#xM_key = jpd.pd_get_xM_MACCSkeys( pdr)
	xM_molw = jpd.pd_get_xM_molw( pdr)
	#xM_lasa = jpd.pd_get_xM_lasa( pdr)
	#xM = np.concatenate( [xM_fp, xM_key, xM_molw, xM_lasa], axis = 1)
	#xM = xM_molw
	xM = np.divide( xM_molw, np.std( xM_molw, axis = 0))

	yV = jpd.pd_get_yV( pdr, y_id = 'exp').A1
	yV = [1 if y > 0 else 0 for y in yV] # classification is performed

	X, Y = map( np.array, [xM, yV])
	assert X.shape[0] == Y.shape[0]

	if test_rate > 0:
		X, Y, X_test, Y_test = XY_split( X, Y, test_rate)
		data_sets.test = DataSet_CSV( X_test, Y_test, disp = disp)

	if validation_rate > 0:
		X, Y, X_val, Y_val = XY_split( X, Y, validation_rate)
		data_sets.validation = DataSet_CSV( X_val, Y_val, disp = disp)

	# If test_rate and validation_rate are both zero, 
	# all data is allocated to train dataset. 
	data_sets.train = DataSet_CSV( X, Y, disp = disp)

	data_sets.IMAGE_PIXELS = xM.shape[1]
	return data_sets	

	
def read_data_sets_mol_sd_key( fname, validation_rate = 0, test_rate = 0, disp = False):
	class DataSets(object):
		pass
	data_sets = DataSets()

	pdr = pd.read_csv( fname)
	#xM_fp = jpd.pd_get_xM( pdr)
	xM_key = jpd.pd_get_xM_MACCSkeys( pdr)
	#xM_molw = jpd.pd_get_xM_molw( pdr)
	#xM_lasa = jpd.pd_get_xM_lasa( pdr)
	#xM = np.concatenate( [xM_fp, xM_key, xM_molw, xM_lasa], axis = 1)
	xM = xM_key
	#xM = np.divide( xM_molw, np.std( xM_molw, axis = 0))

	yV = jpd.pd_get_yV( pdr, y_id = 'exp').A1
	yV = [1 if y > 0 else 0 for y in yV] # classification is performed

	X, Y = map( np.array, [xM, yV])
	assert X.shape[0] == Y.shape[0]

	if test_rate > 0:
		X, Y, X_test, Y_test = XY_split( X, Y, test_rate)
		data_sets.test = DataSet_CSV( X_test, Y_test, disp = disp)

	if validation_rate > 0:
		X, Y, X_val, Y_val = XY_split( X, Y, validation_rate)
		data_sets.validation = DataSet_CSV( X_val, Y_val, disp = disp)

	# If test_rate and validation_rate are both zero, 
	# all data is allocated to train dataset. 
	data_sets.train = DataSet_CSV( X, Y, disp = disp)

	data_sets.IMAGE_PIXELS = xM.shape[1]
	return data_sets	


def read_data_sets_mol( fname, validation_rate = 0, test_rate = 0, disp = False):
	class DataSets(object):
		pass
	data_sets = DataSets()

	pdr = pd.read_csv( fname)
	xM_fp = jpd.pd_get_xM( pdr)
	xM_key = jpd.pd_get_xM_MACCSkeys( pdr)
	xM_molw = jpd.pd_get_xM_molw( pdr)
	xM_lasa = jpd.pd_get_xM_lasa( pdr)
	xM = np.concatenate( [xM_fp, xM_key, xM_molw, xM_lasa], axis = 1)

	yV = jpd.pd_get_yV( pdr, y_id = 'exp').A1
	yV = [1 if y > 0 else 0 for y in yV] # classification is performed

	X, Y = map( np.array, [xM, yV])
	assert X.shape[0] == Y.shape[0]

	if test_rate > 0:
		X, Y, X_test, Y_test = XY_split( X, Y, test_rate)
		data_sets.test = DataSet_CSV( X_test, Y_test, disp = disp)

	if validation_rate > 0:
		X, Y, X_val, Y_val = XY_split( X, Y, validation_rate)
		data_sets.validation = DataSet_CSV( X_val, Y_val, disp = disp)

	# If test_rate and validation_rate are both zero, 
	# all data is allocated to train dataset. 
	data_sets.train = DataSet_CSV( X, Y, disp = disp)

	data_sets.IMAGE_PIXELS = xM.shape[1]
	return data_sets

def read_data_sets_mol_molw( fname, validation_rate = 0, test_rate = 0, disp = False):
	class DataSets(object):
		pass
	data_sets = DataSets()

	pdr = pd.read_csv( fname)
	#xM_fp = jpd.pd_get_xM( pdr)
	#xM_key = jpd.pd_get_xM_MACCSkeys( pdr)
	xM_molw = jpd.pd_get_xM_molw( pdr)
	#xM_lasa = jpd.pd_get_xM_lasa( pdr)
	#xM = np.concatenate( [xM_fp, xM_key, xM_molw, xM_lasa], axis = 1)

	"Normalize xM so as to be a set of unit norm random values"
	xM = np.divide( xM_molw, np.std( xM_molw, axis = 0))
	yV = jpd.pd_get_yV( pdr, y_id = 'exp').A1

	X, Y = map( np.array, [xM, yV])
	assert X.shape[0] == Y.shape[0]

	if test_rate > 0:
		X, Y, X_test, Y_test = XY_split( X, Y, test_rate)
		data_sets.test = DataSet_CSV( X_test, Y_test, disp = disp)

	if validation_rate > 0:
		X, Y, X_val, Y_val = XY_split( X, Y, validation_rate)
		data_sets.validation = DataSet_CSV( X_val, Y_val, disp = disp)

	# If test_rate and validation_rate are both zero, 
	# all data is allocated to train dataset. 
	data_sets.train = DataSet_CSV( X, Y, disp = disp)

	# The length of descriptors are fed back.
	data_sets.IMAGE_PIXELS = xM.shape[1]

	return data_sets

################################
### Stability
################################
def read_data_sets_sd_logK( fname, y_id = "log_K_hyd", th_l = ['<', 2.81], shuffle = True,
							validation_rate = 0, test_rate = 0, disp = False):

	#define closure class and functions 
	class DataSets(object):
		pass

	def get_xMyV( fname, y_id):
		pdr = pd.read_csv( fname)
		xM_fp = jpd.pd_get_xM( pdr)
		
		xM = xM_fp
		yV = jpd.pd_get_yV( pdr, y_id = y_id)
		return xM, yV

	def do_shuffle( xM, yV):
		idx_l = range(xM.shape[0])
		random.shuffle( idx_l) # inplace command
		xM_sf = xM[ idx_l, :]
		yV_sf = yV[ idx_l]
		return xM_sf, yV_sf

	def gen_bin_vec( yV, th_l):
		if th_l[0] == '>':
			yV_bin = [1 if y > th_l[1] else 0 for y in yV] # classification is performed
		else:
			yV_bin = [1 if y < th_l[1] else 0 for y in yV] # classification is performed

		return yV_bin
	#===================================================================

	# main codes are started
	data_sets = DataSets()

	xM, yV = get_xMyV( fname, y_id)
	yv = yV.A1

	if shuffle:
		xM, yv = do_shuffle( xM, yv)

	yv_bin = gen_bin_vec( yv, th_l)

	X, Y = map( np.array, [xM, yv_bin])
	assert X.shape[0] == Y.shape[0]

	if test_rate > 0:
		X, Y, X_test, Y_test = XY_split( X, Y, test_rate)
		data_sets.test = DataSet_CSV( X_test, Y_test, disp = disp)

	if validation_rate > 0:
		X, Y, X_val, Y_val = XY_split( X, Y, validation_rate)
		data_sets.validation = DataSet_CSV( X_val, Y_val, disp = disp)

	# If test_rate and validation_rate are both zero, 
	# all data is allocated to train dataset. 
	data_sets.train = DataSet_CSV( X, Y, disp = disp)

	data_sets.IMAGE_PIXELS = xM.shape[1]
	return data_sets

def read_data_sets_mol_gen( N_shape, sig = 0.1, validation_rate = 0, test_rate = 0, disp = False):
	"""
	Here, new data are generated using randn() in numpy.
	Depending on disp, now the type of datasets are notated before the shape 
	of them is shown.
	"""
	class DataSets(object):
		pass
	data_sets = DataSets()

	if type( N_shape) == type( 0):
		N_shape = ( N_shape, 1)

	X = np.random.randn( *N_shape)
	n = np.random.randn( N_shape[0], 1)
	w = np.random.randn( N_shape[1], 1)
	print "Weight vector is {}.".format( w)
	# sig = 0.1 # this value can be updated later on.
	Y = np.dot( X, w) + sig * n

	#X, Y = map( np.array, [xM, yV])
	assert X.shape[0] == Y.shape[0]

	if test_rate > 0:
		X, Y, X_test, Y_test = XY_split( X, Y, test_rate)
		if disp:
			print "Testing Dataset:"
		data_sets.test = DataSet_CSV( X_test, Y_test, disp = disp)

	if validation_rate > 0:
		X, Y, X_val, Y_val = XY_split( X, Y, validation_rate)
		if disp:
			print "Validation Dataset:"
		data_sets.validation = DataSet_CSV( X_val, Y_val, disp = disp)

	# If test_rate and validation_rate are both zero, 
	# all data is allocated to train dataset. 
	if disp:
		print "Training Dataset:"
	data_sets.train = DataSet_CSV( X, Y, disp = disp)

	return data_sets

  
"""
  ALL_DATA_PICKLE = 'minst.pkl'
  data_list = already_pikle( ALL_DATA_PICKLE, train_dir)
  if not data_list:
	TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
	TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
	TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
	TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
	VALIDATION_SIZE = 5000
  
	local_file = maybe_download(TRAIN_IMAGES, train_dir)
	train_images = extract_images(local_file)
  
	local_file = maybe_download(TRAIN_LABELS, train_dir)
	train_labels = extract_labels(local_file, one_hot=one_hot)
  
	local_file = maybe_download(TEST_IMAGES, train_dir)
	test_images = extract_images(local_file)
  
	local_file = maybe_download(TEST_LABELS, train_dir)
	test_labels = extract_labels(local_file, one_hot=one_hot)
  
	validation_images = train_images[:VALIDATION_SIZE]
	validation_labels = train_labels[:VALIDATION_SIZE]
	train_images = train_images[VALIDATION_SIZE:]
	train_labels = train_labels[VALIDATION_SIZE:]

	if fastload: # This is working only the flag of fastload is turned on. 
	  data_list = [train_images, train_labels, 
		validation_images, validation_labels, 
		test_images, test_labels]
	  save_pickle( ALL_DATA_PICKLE, train_dir, data_list)
  else:
	[train_images, train_labels, 
	  validation_images, validation_labels, 
	  test_images, test_labels] = data_list

  data_sets.train = DataSet(train_images, train_labels)
  data_sets.validation = DataSet(validation_images, validation_labels)
  data_sets.test = DataSet(test_images, test_labels)

  return data_sets
"""

class DataSet_CSV(object):

	def __init__(self, images, labels, disp = False):
		"""
		Construct a DataSet. one_hot arg is used only if fake_data is true.
		Fake mode is not supported. Moreover, onehot is also not supported. 
		"""
		assert images.shape[0] == labels.shape[0]
		self._data_size = images.shape[0]

		self._num_examples = images.shape[0]

		# Convert from [0, 255] -> [0.0, 1.0].
		images = images.astype(numpy.float32)
		# images = numpy.multiply(images, 1.0 / 255.0)

		self._images = images
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def data_size(self):
		return self._data_size

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size, fake_data=False):
		"""
		Return the next `batch_size` examples from this data set.
		Fake mode codes are removed. 
		fake_data is used for compatibilty, but not applicable in this code
		"""

		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._data_size:
			"""
			If the last batch is reached, snuffling is applied. 
			The remained data is omitted and data reading is restarted from 
			the initial point. while the new dataset is the snuffled version of
			the original data (or the previous data). 
			"""
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = numpy.arange(self._data_size)
			numpy.random.shuffle(perm)
			self._images = self._images[perm]
			self._labels = self._labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._data_size
		end = self._index_in_epoch
		return self._images[start:end], self._labels[start:end]

class DataSet(object):

	def __init__(self, images, labels, fake_data=False, one_hot=False):
		"""Construct a DataSet. one_hot arg is used only if fake_data is true."""

		if fake_data:
			self._num_examples = 10000
			self.one_hot = one_hot
		else:
			assert images.shape[0] == labels.shape[0], (
					'images.shape: %s labels.shape: %s' % (images.shape,
																								 labels.shape))
			self._num_examples = images.shape[0]

			# Convert shape from [num examples, rows, columns, depth]
			# to [num examples, rows*columns] (assuming depth == 1)
			assert images.shape[3] == 1
			images = images.reshape(images.shape[0],
															images.shape[1] * images.shape[2])
			# Convert from [0, 255] -> [0.0, 1.0].
			images = images.astype(numpy.float32)
			images = numpy.multiply(images, 1.0 / 255.0)
		self._images = images
		self._labels = labels
		self._epochs_completed = 0
		self._index_in_epoch = 0

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size, fake_data=False):
		"""Return the next `batch_size` examples from this data set."""
		if fake_data:
			fake_image = [1] * 784
			if self.one_hot:
				fake_label = [1] + [0] * 9
			else:
				fake_label = 0
			return [fake_image for _ in xrange(batch_size)], [
					fake_label for _ in xrange(batch_size)]
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_examples:
			# Finished epoch
			self._epochs_completed += 1
			# Shuffle the data
			perm = numpy.arange(self._num_examples)
			numpy.random.shuffle(perm)
			self._images = self._images[perm]
			self._labels = self._labels[perm]
			# Start next epoch
			start = 0
			self._index_in_epoch = batch_size
			assert batch_size <= self._num_examples
		end = self._index_in_epoch
		return self._images[start:end], self._labels[start:end]