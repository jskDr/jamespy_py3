"""
Binary_model - binary regression model including MBR
Developer: (James) Sung-Jin Kim, jaemssungjin.kim@gmail.com
Creation Date: July 11, 2015
Update Date: July 11, 2015
Version: ver 0.1 rev 0
"""

from sklearn import linear_model
import numpy as np

import j3x.jpyx


class BIKE_A_Ridge( linear_model.Ridge): # Later on, Viking will be built
	"""
	BIKE - BInary Kernel Ensemble (BIKE) method
	"""
	def __init__(self, A, alpha = 0.5):
		"""
		A is precomputed similarity matrix of xM(all)
		Depending on k-fold indexing, the associated A[train] and A[test] matrices will be selected.
		"""
		self.A = A
		super(BIKE_A_Ridge, self).__init__(alpha = alpha)

	def _fit( self, xM_train_idx, yV):
		self.train_idx = xM_train_idx[:,0]
		A_train = self.A[ np.ix_(xM_train_idx[:,0], self.train_idx)]
		super(BIKE_Ridge, self).fit( A_train, yV)

	def fit( self, xM_train_idx, yV):
		self.train_idx = xM_train_idx.T
		A_train = self.A[ [xM_train_idx, self.train_idx]]
		super(BIKE_A_Ridge, self).fit( A_train, yV)		

	def predict( self, xM_test_idx):
		"""
		The index vector of a train sequence will be used to pick up 
		testing similarity matrix (or precomputed kernel output matrix). 
		"""
		A_test = self.A[ [xM_test_idx, self.train_idx]]
		return super(BIKE_A_Ridge, self).predict( A_test)

class BIKE_Ridge( linear_model.Ridge): # Later on, Viking will be built
	"""
	BIKE - BInary Kernel Ensemble (BIKE) method
	"""
	def __init__(self, A_list = [], X = None, alpha = 0.5):
		"""
		A is precomputed similarity matrix of xM(all)
		Depending on k-fold indexing, the associated A[train] and A[test] matrices will be selected.
		"""
		self.A_list = A_list
		self.X = X
		super(BIKE_Ridge, self).__init__(alpha = alpha)

	def gen_AX( self, xM_idx):

		AX_list = list()
		for A in self.A_list:
			AX_list.append( A[ [xM_idx, self.xM_train_idx_T]])
		# Now X will be added as well since it is also a part of descriptor set.
		if self.X is not None:
			#print 'xM_idx[:,0] =', xM_idx[:,0]
			xM_con = self.X[ xM_idx[:,0], :]
			#print 'xM_con.shape = ', xM_con.shape
			AX_list.append( xM_con)

		# All kernel outputs and linear descriptors will be used as an input matrix.		
		return np.concatenate( AX_list, axis = 1)


	def fit( self, xM_train_idx, yV):
		"""
		A common part between fit() and predict() are made be a function, gen_AX
		"""
		self.xM_train_idx_T = xM_train_idx.T
		AX = self.gen_AX( xM_train_idx)
		super(BIKE_Ridge, self).fit( AX, yV)

	def predict( self, xM_test_idx):
		"""
		The index vector of a train sequence will be used to pick up 
		testing similarity matrix (or precomputed kernel output matrix). 
		"""
		AX = self.gen_AX( xM_test_idx)
		return super(BIKE_Ridge, self).predict( AX)


"""
MBR  Ensemble 
- Since 'direct' is a part of this resembling mode,
  the performance of the direct (MLR) cases can be evaluated with this MBR-Ensemble method. 
"""

class MBR_Ensemble_Ridge( linear_model.Ridge): 
	def __init__(self, alpha = 0.5, fsp_l = [], fpm_l = []):
		"""
		fsp_l: feature split points for spiting different descriptors 
		        - refer to np.split() 
		fpm_l: feature preprocessing mode  
		        - 'tanimoto', 'direct' (supporting), 'rbf', 'tm-rbf' (under-developing)
		Note: len( fsp_l) == len( fpm_1) - 1 to specify preprocessing modes for each feature group
		"""
		self.fsp_l = fsp_l
		if len(fpm_l) == 0:
			fpm_l = ['tanimoto'] * (len( fsp_l) + 1)
		else:
			if len( fsp_l) == len( fpm_l) - 1:
				self.fpm_l = fpm_l
			else:
				raise ValueError( "Check to be: len( fsp_l) == len( fpm_l) - 1")

		super(MBR_Ensemble_Ridge, self).__init__(alpha = alpha)

	def fit( self, xM_train_con, yV):
		self.xM_train_l = np.split( xM_train_con, self.fsp_l, axis = 1)
		
		#A_train_l = map(j3x.jpyx.calc_tm_sim_M, self.xM_train_l)
		A_train_l = list()
		for xM_train, fpm in zip( self.xM_train_l, self.fpm_l):
			# print 'fpm, xM_train.shape', '->', fpm, xM_train.shape
			if fpm == 'tanimoto':
				# Since tanimoto is applied, xM must be binary but
				# it is non binary type because it is combined with other type (float)
				A_train_l.append( j3x.jpyx.calc_tm_sim_M( xM_train.astype( int)))
			elif fpm == 'direct':
				A_train_l.append( xM_train)
			else:
				raise ValueError("For fpm, the given mode is not supported:" + fpm)

		A_train_ensemble = np.concatenate( A_train_l, axis = 1)
		super(MBR_Ensemble_Ridge, self).fit( A_train_ensemble, yV)

	def predict( self, xM_test_con):
		xM_test_l = np.split( xM_test_con, self.fsp_l, axis = 1)

		A_test_l = list()
		for xM_train, xM_test, fpm in zip(self.xM_train_l, xM_test_l, self.fpm_l):
			if fpm == 'tanimoto':
				xM_all = np.concatenate( (xM_train, xM_test), axis = 0) 
				A_all = j3x.jpyx.calc_tm_sim_M( xM_all.astype( int))
				A_test = A_all[ xM_train.shape[0]:, :xM_train.shape[0]] 
				A_test_l.append( A_test)
			elif fpm == 'direct':
				A_test_l.append( xM_test)
			else:
				raise ValueError("For fpm, the given mode is not supported:" + fpm)

		A_test_ensemble = np.concatenate( A_test_l, axis = 1)
		return super(MBR_Ensemble_Ridge, self).predict( A_test_ensemble)


"""
MBR EnsembleBin
- if MBR_Ensemble is meta class inherented from six.with_metaclass(ABCMeta, LinearModel),
  MBR_Ensemble_Ridge and MBR_Ensemble_Lasso can be more compact such as 
  describing only __init__ by inhereting both MBR_Ensemble and either 
  linear_model.Ridge or linear_model.Lasso depending on the mode. 
- Now it is implemnted more simply. Later, such deep implementation will be applied. 
"""

class _MBR_EnsembleBin_Ridge_r0( linear_model.Ridge): 
	def __init__(self, alpha = 0.5):
		super(MBR_EnsembleBin_Ridge, self).__init__(alpha = alpha)

	def fit( self, xM_train_l, yV):
		self.xM_train_l = xM_train_l
		A_train_l = list(map(j3x.jpyx.calc_tm_sim_M, xM_train_l))
		A_train_ensemble = np.concatenate( A_train_l, axis = 1)
		super(MBR_EnsembleBin_Ridge, self).fit( A_train_ensemble, yV)

	def predict( self, xM_test_l):
		xM_all_l = [np.concatenate( (xM_train, xM_test), axis = 0) 
			for xM_train, xM_test in zip( self.xM_train_l, xM_test_l)]
		A_all_l = list(map( j3x.jpyx.calc_tm_sim_M, xM_all_l))
		A_test_l = [A_all[ xM_train.shape[0]:, :xM_train.shape[0]] 
			for A_all, xM_train in zip( A_all_l, self.xM_train_l)]
		A_test_ensemble = np.concatenate( A_test_l, axis = 1)
		return super(MBR_EnsembleBin_Ridge, self).predict( A_test_ensemble)


class MBR_EnsembleBin_Ridge( linear_model.Ridge): 
	def __init__(self, alpha = 0.5, fsp_l = []):
		"""
		fsp_l = feature split points for spliting differnt descriptors 
		"""
		self.fsp_l = fsp_l
		super(MBR_EnsembleBin_Ridge, self).__init__(alpha = alpha)

	def fit( self, xM_train_con, yV):
		self.xM_train_l = np.split( xM_train_con, self.fsp_l, axis = 1)
		#self.xM_train_l = xM_train_l
		A_train_l = list(map(j3x.jpyx.calc_tm_sim_M, self.xM_train_l))
		A_train_ensemble = np.concatenate( A_train_l, axis = 1)
		super(MBR_EnsembleBin_Ridge, self).fit( A_train_ensemble, yV)

	def predict( self, xM_test_con):
		xM_test_l = np.split( xM_test_con, self.fsp_l, axis = 1)
		xM_all_l = [np.concatenate( (xM_train, xM_test), axis = 0) 
			for xM_train, xM_test in zip( self.xM_train_l, xM_test_l)]
		A_all_l = list(map( j3x.jpyx.calc_tm_sim_M, xM_all_l))
		A_test_l = [A_all[ xM_train.shape[0]:, :xM_train.shape[0]] 
			for A_all, xM_train in zip( A_all_l, self.xM_train_l)]
		A_test_ensemble = np.concatenate( A_test_l, axis = 1)
		return super(MBR_EnsembleBin_Ridge, self).predict( A_test_ensemble)

"""
MBR TM 
- Gamma is not considered.
"""

class MBR_TM_Ridge( linear_model.Ridge):
	def __init__(self, alpha = 0.5):
		super(MBR_TM_Ridge, self).__init__(alpha = alpha)

	def fit( self, xM_train, yV):
		self.xM_train = xM_train
		A_train = j3x.jpyx.calc_tm_sim_M( xM_train)
		super(MBR_TM_Ridge, self).fit( A_train, yV)

	def predict( self, xM_test):
		#A = j3x.jpyx.calc_bin_sim_M( xM_test.astype(int), gamma = self.gamma)
		xM_all = np.concatenate( (self.xM_train, xM_test), axis = 0)
		A_all = j3x.jpyx.calc_tm_sim_M( xM_all)
		A_test = A_all[ self.xM_train.shape[0]:, :self.xM_train.shape[0]]
		return super(MBR_TM_Ridge, self).predict( A_test)

class MBR_TM_Lasso( linear_model.Lasso):
	def __init__(self, alpha = 1.0, gamma = 1):
		super(MBR_TM_Ridge, self).__init__(alpha = alpha)

	def fit( self, xM_train, yV):
		self.xM_train = xM_train
		A_train = j3x.jpyx.calc_tm_sim_M( xM_train)
		super(MBR_TM_Lasso, self).fit( A_train, yV)

	def predict( self, xM_test):
		xM_all = np.concatenate( (self.xM_train, xM_test), axis = 0)
		A_all = j3x.jpyx.calc_tm_sim_M( xM_all)
		A_test = A_all[ self.xM_train.shape[0]:, :self.xM_train.shape[0]]
		return super(MBR_TM_Lasso, self).predict( A_test)

"""
MBR Sim
Similarityy control MBR
Original MBR does not have a functionality to change gamma,
although SVM has it. It will be considered later on. 
"""

class MBR_Ridge( linear_model.Ridge):
	def __init__(self, alpha = 0.5, gamma = 1):
		self.alpha = alpha
		self.gamma = gamma
		#self.clf = linear_model.Ridge( alpha = self.alpha)
		#linear_model.Ridge( self, alpha = self.alpha)
		super(MBR_Ridge, self).__init__(alpha = self.alpha)

	def fit( self, xM_train, yV):
		self.xM_train = xM_train
		A_train = j3x.jpyx.calc_bin_sim_M( xM_train, gamma = self.gamma)
		super(MBR_Ridge, self).fit( A_train, yV)

	def predict( self, xM_test):
		#A = j3x.jpyx.calc_bin_sim_M( xM_test.astype(int), gamma = self.gamma)
		xM_all = np.concatenate( (self.xM_train, xM_test), axis = 0)
		A_all = j3x.jpyx.calc_bin_sim_M( xM_all, gamma = self.gamma)
		A_test = A_all[ self.xM_train.shape[0]:, :self.xM_train.shape[0]]
		return super(MBR_Ridge, self).predict( A_test)

class MBR_Lasso( linear_model.Lasso):
	def __init__(self, alpha = 1.0, gamma = 1):
		self.alpha = alpha
		self.gamma = gamma
		#self.clf = linear_model.Ridge( alpha = self.alpha)
		#linear_model.Ridge( self, alpha = self.alpha)
		super(MBR_Lasso, self).__init__(alpha = self.alpha)

	def fit( self, xM_train, yV):
		self.xM_train = xM_train
		A_train = j3x.jpyx.calc_bin_sim_M( xM_train, gamma = self.gamma)
		super(MBR_Lasso, self).fit( A_train, yV)

	def predict( self, xM_test):
		#A = j3x.jpyx.calc_bin_sim_M( xM_test.astype(int), gamma = self.gamma)
		xM_all = np.concatenate( (self.xM_train, xM_test), axis = 0)
		A_all = j3x.jpyx.calc_bin_sim_M( xM_all, gamma = self.gamma)
		A_test = A_all[ self.xM_train.shape[0]:, :self.xM_train.shape[0]]
		return super(MBR_Lasso, self).predict( A_test)

"""
MBR_Dist
"""

def sim_to_dist( A):
	A *= -1
	A = A + 1
	A = np.power( np.abs(A), 2)
	np.exp( A, A)
	return A

class MBR_Dist_Lasso( linear_model.Lasso):
	def __init__(self, alpha = 1.0, gamma = 1):
		self.alpha = alpha
		self.gamma = gamma
		#self.clf = linear_model.Ridge( alpha = self.alpha)
		#linear_model.Ridge( self, alpha = self.alpha)
		super(MBR_Dist_Lasso, self).__init__(alpha = self.alpha)

	def fit( self, xM_train, yV):
		self.xM_train = xM_train
		A_train = j3x.jpyx.calc_bin_sim_M( xM_train, gamma = self.gamma)		
		super(MBR_Dist_Lasso, self).fit( sim_to_dist( A_train), yV)

	def predict( self, xM_test):
		#A = j3x.jpyx.calc_bin_sim_M( xM_test.astype(int), gamma = self.gamma)
		xM_all = np.concatenate( (self.xM_train, xM_test), axis = 0)
		A_all = j3x.jpyx.calc_bin_sim_M( xM_all, gamma = self.gamma)
		A_test = A_all[ self.xM_train.shape[0]:, :self.xM_train.shape[0]]
		return super(MBR_Dist_Lasso, self).predict( sim_to_dist(A_test))
