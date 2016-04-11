# some of sklearn codes are updated. 

import numpy as np
from sklearn import cross_validation, metrics

def _cross_val_score_loo_r0( lm, X, y):
	"""
	mean_square_error metric is used from sklearn.metric.

	Return 
	--------
	The mean squared error values are returned. 
	"""

	if len( y.shape) == 1:
		y = np.array( [y]).T

	kf = cross_validation.LeaveOneOut( y.shape[0])
	score_l = list()
	for tr, te in kf:
		lm.fit( X[tr,:], y[tr,:])
		yp = lm.predict( X[te, :])
		score_l.append( metrics.mean_squared_error( y[te,:], yp))

	return score_l

		

def cross_val_score_loo( lm, X, y):
	"""
	mean_square_error metric is used from sklearn.metric.

	Return 
	--------
	The mean squared error values are returned. 
	"""
	# Transformed to array if they are list, np.mat
	X = np.array( X)
	y = np.array( y)
	# Later, assert can be used to define the size of X and y

	if len( y.shape) == 1:
		y = np.array( [y]).T

	kf = cross_validation.LeaveOneOut( y.shape[0])
	# flatterned error vectors for each point are stored in this vector.
	errors_l = list()
	for tr, te in kf:
		lm.fit( X[tr,:], y[tr,:])
		yp = lm.predict( X[te, :])
		errors_l.extend( (y[te,:] - yp).flatten().tolist())

	return errors_l

		