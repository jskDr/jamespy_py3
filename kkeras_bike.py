# kkeras_bike.py
# Sungjin (Jmaes) Kim
# Aug 2, 2016

import numpy as np
import kkeras

class BIKE_DNN( kkeras.MLPR): # Later on, Viking will be built
	"""
	BIKE - BInary Kernel Ensemble (BIKE) method
	MLPR class should be built later on
	"""
	def __init__(self, A_list = [], X = None, l = [49, 30, 10, 1]):
		"""
		A is precomputed similarity matrix of xM(all)
		Depending on k-fold indexing, the associated A[train] and A[test] matrices will be selected.
		"""
		super().__init__( l = l)
		self.A_list = A_list
		self.X = X

	def gen_AX( self, xA_idx):
		"""
		xA_idx is a 2-D ndarray        
		"""
		AX_list = list()
		for A in self.A_list:
			# print( "A.shape", A.shape)
			AX_list.append( A[ [xA_idx, self.xA_train_idx_T]])
		# Now X will be added as well since it is also a part of descriptor set.
		if self.X is not None:
			# print( self.X.shape)
			# print( xA_idx.shape)
			xA_con = self.X[ xA_idx[:,0], :]
			# print( 'xA_con.shape = ', xA_con.shape)
			AX_list.append( xA_con)

		# print( len(AX_list))
		# print( AX_list[0].shape, AX_list[1].shape)

		# All kernel outputs and linear descriptors will be used as an input matrix.		
		return np.concatenate( AX_list, axis = 1)

	def gen_AX_r0( self, xM_idx):

		AX_list = list()
		for A in self.A_list:
			AX_list.append( A[ [xM_idx, self.xM_train_idx_T]])
		# Now X will be added as well since it is also a part of descriptor set.
		if self.X is not None:
			#print 'xM_idx[:,0] =', xM_idx[:,0]
			xM_con = self.X[ xM_idx[:,0], :]
			# <-- xM_con = self.X[ xM_idx, :]
			#print 'xM_con.shape = ', xM_con.shape
			AX_list.append( xM_con)

		# All kernel outputs and linear descriptors will be used as an input matrix.		
		return np.concatenate( AX_list, axis = 1)    
    
	def fit( self, xA_train_val_idx, ya_all, batch_size=10, nb_epoch=20, verbose = 0):
		"""
		A common part between fit() and predict() are made be a function, gen_AX
		Validation is departed from all information. 
		This validation information are not used for training in MLP
		while it is used in the linear regression methods such as ordinary, Ridge, LASO.

		Input
		-----
		xM_train_val_idx, list of int
		Both training and validation index of the input data array.
		"""
		val_ratio = 0.2

		# print( type(xA_train_val_idx)) 
		if type(xA_train_val_idx) == list:
			#Make a 2-D array from a 1-D list           
			# print( "1D List to 2D Array")             
			xA_train_val_idx = np.array( [xA_train_val_idx]).T

		ln_train_val = xA_train_val_idx.shape[0]
		ln_train = int( ln_train_val * (1.0 - val_ratio))
		xA_train_idx = xA_train_val_idx[:ln_train, :]
		xA_val_idx = xA_train_val_idx[ln_train:, :]

		self.xA_train_idx_T = xA_train_idx.T
		AX_train, ya_train = self.gen_AX( xA_train_idx), ya_all[ xA_train_idx[:,0]]
		AX_val, ya_val = self.gen_AX( xA_val_idx), ya_all[ xA_val_idx[:,0]]

		# print( "AX_train.shape", AX_train.shape)
		# print( "ya_train.shape", ya_train.shape)
		# print( "AX_val.shape", AX_val.shape)
		# print( "ya_val.shape", ya_val.shape)       
		return super().fit( AX_train, ya_train, AX_val, ya_val, 
			verbose = verbose, batch_size=batch_size, nb_epoch=nb_epoch)

	def predict( self, xA_test_idx): # t.b.d, ob developing
		"""
		The index vector of a train sequence will be used to pick up 
		testing similarity matrix (or precomputed kernel output matrix). 
		"""
		AX = self.gen_AX( xA_test_idx)
		return super().predict( AX)

	def score( self, xA_idx, ya):
		AX = self.gen_AX( xA_idx)
		return super().score( AX, ya)

class BIKE_DNN_N( kkeras.MLPR_N): # Later on, Viking will be built
	"""
	BIKE - BInary Kernel Ensemble (BIKE) method
	MLPR class should be built later on
	"""
	def __init__(self, A_list = [], X = None, l = [49, 30, 10, 1], l2_param = 0):
		"""
		A is precomputed similarity matrix of xM(all)
		Depending on k-fold indexing, the associated A[train] and A[test] matrices will be selected.
		"""
		super().__init__( l = l, l2_param = l2_param)
		self.A_list = A_list
		self.X = X

	def gen_AX( self, xA_idx):
		"""
		xA_idx is a 2-D ndarray        
		"""
		AX_list = list()
		for A in self.A_list:
			# print( "A.shape", A.shape)
			AX_list.append( A[ [xA_idx, self.xA_train_idx_T]])
		# Now X will be added as well since it is also a part of descriptor set.
		if self.X is not None:
			# print( self.X.shape)
			# print( xA_idx.shape)
			xA_con = self.X[ xA_idx[:,0], :]
			# print( 'xA_con.shape = ', xA_con.shape)
			AX_list.append( xA_con)

		# print( len(AX_list))
		# print( AX_list[0].shape, AX_list[1].shape)

		# All kernel outputs and linear descriptors will be used as an input matrix.		
		return np.concatenate( AX_list, axis = 1)

	def gen_AX_r0( self, xM_idx):

		AX_list = list()
		for A in self.A_list:
			AX_list.append( A[ [xM_idx, self.xM_train_idx_T]])
		# Now X will be added as well since it is also a part of descriptor set.
		if self.X is not None:
			#print 'xM_idx[:,0] =', xM_idx[:,0]
			xM_con = self.X[ xM_idx[:,0], :]
			# <-- xM_con = self.X[ xM_idx, :]
			#print 'xM_con.shape = ', xM_con.shape
			AX_list.append( xM_con)

		# All kernel outputs and linear descriptors will be used as an input matrix.		
		return np.concatenate( AX_list, axis = 1)    
    
	def fit( self, xA_train_val_idx, ya_all, batch_size=10, nb_epoch=20, verbose = 0):
		"""
		A common part between fit() and predict() are made be a function, gen_AX
		Validation is departed from all information. 
		This validation information are not used for training in MLP
		while it is used in the linear regression methods such as ordinary, Ridge, LASO.

		Input
		-----
		xM_train_val_idx, list of int
		Both training and validation index of the input data array.
		"""
		val_ratio = 0.2

		# print( type(xA_train_val_idx)) 
		if type(xA_train_val_idx) == list:
			#Make a 2-D array from a 1-D list           
			# print( "1D List to 2D Array")             
			xA_train_val_idx = np.array( [xA_train_val_idx]).T

		ln_train_val = xA_train_val_idx.shape[0]
		ln_train = int( ln_train_val * (1.0 - val_ratio))
		xA_train_idx = xA_train_val_idx[:ln_train, :]
		xA_val_idx = xA_train_val_idx[ln_train:, :]

		self.xA_train_idx_T = xA_train_idx.T
		AX_train, ya_train = self.gen_AX( xA_train_idx), ya_all[ xA_train_idx[:,0]]
		AX_val, ya_val = self.gen_AX( xA_val_idx), ya_all[ xA_val_idx[:,0]]

		# print( "AX_train.shape", AX_train.shape)
		# print( "ya_train.shape", ya_train.shape)
		# print( "AX_val.shape", AX_val.shape)
		# print( "ya_val.shape", ya_val.shape)       
		return super().fit( AX_train, ya_train, AX_val, ya_val, 
			verbose = verbose, batch_size=batch_size, nb_epoch=nb_epoch)

	def predict( self, xA_test_idx): # t.b.d, ob developing
		"""
		The index vector of a train sequence will be used to pick up 
		testing similarity matrix (or precomputed kernel output matrix). 
		"""
		AX = self.gen_AX( xA_test_idx)
		return super().predict( AX)

	def score( self, xA_idx, ya):
		AX = self.gen_AX( xA_idx)
		return super().score( AX, ya)	