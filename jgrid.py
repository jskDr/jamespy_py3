"""
grid search codes for machine learning
"""

from sklearn import cross_validation, cross_validation, grid_search, linear_model, svm, metrics
import numpy as np
import pandas as pd
from operator import itemgetter

import jutil
import j3x.jpyx
from jsklearn import binary_model

def gs_Lasso( xM, yV, alphas_log = (-1, 1, 9), n_folds=5, n_jobs = -1):

	print(xM.shape, yV.shape)

	clf = linear_model.Lasso()
	#parmas = {'alpha': np.logspace(1, -1, 9)}
	parmas = {'alpha': np.logspace( *alphas_log)}
	kf5 = cross_validation.KFold( xM.shape[0], n_folds = n_folds, shuffle=True)
	gs = grid_search.GridSearchCV( clf, parmas, scoring = 'r2', cv = kf5, n_jobs = n_jobs)

	gs.fit( xM, yV)

	return gs

def gs_Lasso_norm( xM, yV, alphas_log = (-1, 1, 9)):

	print(xM.shape, yV.shape)

	clf = linear_model.Lasso( normalize = True)
	#parmas = {'alpha': np.logspace(1, -1, 9)}
	parmas = {'alpha': np.logspace( *alphas_log)}
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=5, shuffle=True)
	gs = grid_search.GridSearchCV( clf, parmas, scoring = 'r2', cv = kf5, n_jobs = -1)

	gs.fit( xM, yV)

	return gs

def gs_Lasso_kf( xM, yV, alphas_log_l):

	kf5_ext = cross_validation.KFold( xM.shape[0], n_folds=5, shuffle=True)

	score_l = []
	for ix, (tr, te) in enumerate( kf5_ext):

		print('{}th fold external validation stage ============================'.format( ix + 1))
		xM_in = xM[ tr, :]
		yV_in = yV[ tr, 0]

		print('First Lasso Stage')
		gs1 = gs_Lasso( xM_in, yV_in, alphas_log_l[0])
		print('Best score:', gs1.best_score_)
		print('Best param:', gs1.best_params_)
		print(gs1.grid_scores_)


		nz_idx = gs1.best_estimator_.sparse_coef_.indices
		xM_in_nz = xM_in[ :, nz_idx]

		print('Second Lasso Stage')
		gs2 = gs_Lasso( xM_in_nz, yV_in, alphas_log_l[1])
		print('Best score:', gs2.best_score_)
		print('Best param:', gs2.best_params_)
		print(gs2.grid_scores_)

		print('External Validation Stage')
		xM_out = xM[ te, :]
		yV_out = yV[ te, 0]
		xM_out_nz = xM_out[:, nz_idx]
		score = gs2.score( xM_out_nz, yV_out)

		print(score)		
		score_l.append( score)

		print('')

	print('all scores:', score_l)
	print('average scores:', np.mean( score_l))

	return score_l

def gs_Lasso_kf_ext( xM, yV, alphas_log_l):

	kf5_ext = cross_validation.KFold( xM.shape[0], n_folds=5, shuffle=True)

	score_l = []
	for ix, (tr, te) in enumerate( kf5_ext):

		print('{}th fold external validation stage ============================'.format( ix + 1))
		xM_in = xM[ tr, :]
		yV_in = yV[ tr, 0]

		print('First Lasso Stage')
		gs1 = gs_Lasso( xM_in, yV_in, alphas_log_l[0])
		print('Best score:', gs1.best_score_)
		print('Best param:', gs1.best_params_)
		print(gs1.grid_scores_)


		nz_idx = gs1.best_estimator_.sparse_coef_.indices
		xM_in_nz = xM_in[ :, nz_idx]

		print('Second Lasso Stage')
		gs2 = gs_Lasso( xM_in_nz, yV_in, alphas_log_l[1])
		print('Best score:', gs2.best_score_)
		print('Best param:', gs2.best_params_)
		print(gs2.grid_scores_)

		print('External Validation Stage')
		# Obtain prediction model by whole data including internal validation data
		alpha = gs2.best_params_['alpha']
		clf = linear_model.Lasso( alpha = alpha)
		clf.fit( xM_in_nz, yV_in)

		xM_out = xM[ te, :]
		yV_out = yV[ te, 0]
		xM_out_nz = xM_out[:, nz_idx]
		score = clf.score( xM_out_nz, yV_out)

		print(score)		
		score_l.append( score)

		print('')

	print('all scores:', score_l)
	print('average scores:', np.mean( score_l))

	return score_l

def _gs_Ridge_r0( xM, yV, alphas_log = (1, -1, 9)):

	print(xM.shape, yV.shape)

	clf = linear_model.Ridge()
	#parmas = {'alpha': np.logspace(1, -1, 9)}
	parmas = {'alpha': np.logspace( *alphas_log)}
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=5, shuffle=True)
	gs = grid_search.GridSearchCV( clf, parmas, scoring = 'r2', cv = kf5, n_jobs = 1)

	gs.fit( xM, yV)

	return gs

def gs_Ridge_Asupervising_2fp( xM1, xM2, yV, s_l, alpha_l):
	"""
	This 2fp case uses two fingerprints at the same in order to 
	combines their preprocessing versions separately. 
	"""
	r2_l2 = list()	
	for alpha in alpha_l:
		print(alpha)
		r2_l = cv_Ridge_Asupervising_2fp( xM1, xM2, yV, s_l, alpha)
		r2_l2.append( r2_l)
	return r2_l2


def _cv_LinearRegression_r0( xM, yV):

	print(xM.shape, yV.shape)

	clf = linear_model.Ridge()
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=5, shuffle=True)
	cv_scores = cross_validation.cross_val_score( clf, xM, yV, scoring = 'r2', cv = kf5, n_jobs = -1)

	return cv_scores

def _cv_LinearRegression_r1( xM, yV):

	print(xM.shape, yV.shape)

	clf = linear_model.LinearRegression()
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=5, shuffle=True)
	cv_scores = cross_validation.cross_val_score( clf, xM, yV, scoring = 'r2', cv = kf5, n_jobs = -1)

	print('R^2 mean, std -->', np.mean( cv_scores), np.std( cv_scores))

	return cv_scores

def _cv_LinearRegression_r2( xM, yV, scoring = 'r2'):

	print(xM.shape, yV.shape)

	clf = linear_model.LinearRegression()
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=5, shuffle=True)
	cv_scores = cross_validation.cross_val_score( clf, xM, yV, scoring = scoring, cv = kf5, n_jobs = -1)

	print('{}: mean, std -->'.format( scoring), np.mean( cv_scores), np.std( cv_scores))

	return cv_scores

def cv_LinearRegression( xM, yV, n_folds = 5, scoring = 'median_absolute_error', disp = False):
	"""
	metrics.explained_variance_score(y_true, y_pred)	Explained variance regression score function
	metrics.mean_absolute_error(y_true, y_pred)	Mean absolute error regression loss
	metrics.mean_squared_error(y_true, y_pred[, ...])	Mean squared error regression loss
	metrics.median_absolute_error(y_true, y_pred)	Median absolute error regression loss
	metrics.r2_score(y_true, y_pred[, ...])	R^2 (coefficient of determination) regression score function.
	"""  
	
	if disp:
		print(xM.shape, yV.shape)

	clf = linear_model.LinearRegression()
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=n_folds, shuffle=True)
	
	cv_score_l = list()
	for train, test in kf5:
		# clf.fit( xM[train,:], yV[train,:])
		# yV is vector but not a metrix here. Hence, it should be treated as a vector
		clf.fit( xM[train,:], yV[train])
		
		yVp_test = clf.predict( xM[test,:])
		if scoring == 'median_absolute_error':
			cv_score_l.append( metrics.median_absolute_error(yV[test], yVp_test))
		else:
			raise ValueError( "{} scoring is not supported.".format( scoring))

	if disp: # Now only this flag is on, the output will be displayed. 
		print('{}: mean, std -->'.format( scoring), np.mean( cv_score_l), np.std( cv_score_l))

	return cv_score_l

def cv_LinearRegression_ci( xM, yV, n_folds = 5, scoring = 'median_absolute_error', disp = False):
	"""
	metrics.explained_variance_score(y_true, y_pred)	Explained variance regression score function
	metrics.mean_absolute_error(y_true, y_pred)	Mean absolute error regression loss
	metrics.mean_squared_error(y_true, y_pred[, ...])	Mean squared error regression loss
	metrics.median_absolute_error(y_true, y_pred)	Median absolute error regression loss
	metrics.r2_score(y_true, y_pred[, ...])	R^2 (coefficient of determination) regression score function.
	"""  
	
	if disp:
		print(xM.shape, yV.shape)

	clf = linear_model.LinearRegression()
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=n_folds, shuffle=True)
	
	cv_score_l = list()
	ci_l = list()
	for train, test in kf5:
		# clf.fit( xM[train,:], yV[train,:])
		# yV is vector but not a metrix here. Hence, it should be treated as a vector
		clf.fit( xM[train,:], yV[train])
		
		yVp_test = clf.predict( xM[test,:])
		
		# Additionally, coef_ and intercept_ are stored. 
		ci_l.append( (clf.coef_, clf.intercept_))
		if scoring == 'median_absolute_error':
			cv_score_l.append( metrics.median_absolute_error(yV[test], yVp_test))
		else:
			raise ValueError( "{} scoring is not supported.".format( scoring))

	if disp: # Now only this flag is on, the output will be displayed. 
		print('{}: mean, std -->'.format( scoring), np.mean( cv_score_l), np.std( cv_score_l))

	return cv_score_l, ci_l

def cv_LinearRegression_ci_pred( xM, yV, n_folds = 5, scoring = 'median_absolute_error', disp = False):
	"""
	metrics.explained_variance_score(y_true, y_pred)	Explained variance regression score function
	metrics.mean_absolute_error(y_true, y_pred)	Mean absolute error regression loss
	metrics.mean_squared_error(y_true, y_pred[, ...])	Mean squared error regression loss
	metrics.median_absolute_error(y_true, y_pred)	Median absolute error regression loss
	metrics.r2_score(y_true, y_pred[, ...])	R^2 (coefficient of determination) regression score function.
	"""  
	
	if disp:
		print(xM.shape, yV.shape)

	clf = linear_model.LinearRegression()
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=n_folds, shuffle=True)
	
	cv_score_l = list()
	ci_l = list()
	yVp = yV.copy() 
	for train, test in kf5:
		# clf.fit( xM[train,:], yV[train,:])
		# yV is vector but not a metrix here. Hence, it should be treated as a vector
		clf.fit( xM[train,:], yV[train])
		
		yVp_test = clf.predict( xM[test,:])
		yVp[test] = yVp_test
		
		# Additionally, coef_ and intercept_ are stored. 
		coef = np.array(clf.coef_).tolist()
		intercept = np.array(clf.intercept_).tolist()
		ci_l.append( (clf.coef_, clf.intercept_))
		if scoring == 'median_absolute_error':
			cv_score_l.append( metrics.median_absolute_error(yV[test], yVp_test))
		else:
			raise ValueError( "{} scoring is not supported.".format( scoring))

	if disp: # Now only this flag is on, the output will be displayed. 
		print('{}: mean, std -->'.format( scoring), np.mean( cv_score_l), np.std( cv_score_l))

	return cv_score_l, ci_l, yVp.A1.tolist()

def cv_LinearRegression_ci_pred_full_Ridge( xM, yV, alpha, n_folds = 5, shuffle=True, disp = False):
	"""
	Note - scoring is not used. I may used later. Not it is remained for compatibility purpose.
	metrics.explained_variance_score(y_true, y_pred)	Explained variance regression score function
	metrics.mean_absolute_error(y_true, y_pred)	Mean absolute error regression loss
	metrics.mean_squared_error(y_true, y_pred[, ...])	Mean squared error regression loss
	metrics.median_absolute_error(y_true, y_pred)	Median absolute error regression loss
	metrics.r2_score(y_true, y_pred[, ...])	R^2 (coefficient of determination) regression score function.
	"""  
	
	if disp:
		print(xM.shape, yV.shape)

	# print( 'alpha of Ridge is', alpha)
	clf = linear_model.Ridge( alpha)
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=n_folds, shuffle=shuffle)
	
	cv_score_l = list()
	ci_l = list()
	yVp = yV.copy() 
	for train, test in kf5:
		# clf.fit( xM[train,:], yV[train,:])
		# yV is vector but not a metrix here. Hence, it should be treated as a vector
		clf.fit( xM[train,:], yV[train])
		
		yVp_test = clf.predict( xM[test,:])
		yVp[test] = yVp_test
		
		# Additionally, coef_ and intercept_ are stored. 		
		ci_l.append( (clf.coef_, clf.intercept_))
		y_a = np.array( yV[test])[:,0]
		yp_a = np.array( yVp_test)[:,0]
		cv_score_l.extend( np.abs(y_a - yp_a).tolist())

	return cv_score_l, ci_l, yVp.A1.tolist()


def cv_LinearRegression_ci_pred_full( xM, yV, n_folds = 5, shuffle=True, disp = False):
	"""
	Note - scoring is not used. I may used later. Not it is remained for compatibility purpose.
	metrics.explained_variance_score(y_true, y_pred)	Explained variance regression score function
	metrics.mean_absolute_error(y_true, y_pred)	Mean absolute error regression loss
	metrics.mean_squared_error(y_true, y_pred[, ...])	Mean squared error regression loss
	metrics.median_absolute_error(y_true, y_pred)	Median absolute error regression loss
	metrics.r2_score(y_true, y_pred[, ...])	R^2 (coefficient of determination) regression score function.
	"""  
	
	if disp:
		print(xM.shape, yV.shape)

	clf = linear_model.LinearRegression()
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=n_folds, shuffle=shuffle)
	
	cv_score_l = list()
	ci_l = list()
	yVp = yV.copy() 
	for train, test in kf5:
		# clf.fit( xM[train,:], yV[train,:])
		# yV is vector but not a metrix here. Hence, it should be treated as a vector
		clf.fit( xM[train,:], yV[train])
		
		yVp_test = clf.predict( xM[test,:])
		yVp[test] = yVp_test
		
		# Additionally, coef_ and intercept_ are stored. 		
		ci_l.append( (clf.coef_, clf.intercept_))
		y_a = np.array( yV[test])[:,0]
		yp_a = np.array( yVp_test)[:,0]
		cv_score_l.extend( np.abs(y_a - yp_a).tolist())

	return cv_score_l, ci_l, yVp.A1.tolist()


def cv_LinearRegression_It( xM, yV, n_folds = 5, scoring = 'median_absolute_error', N_it = 10, disp = False, ldisp = False):
	"""
	N_it times iteration is performed for cross_validation in order to make further average effect. 
	The flag of 'disp' is truned off so each iteration will not shown.  
	"""
	cv_score_le = list()
	for ni in range( N_it):
		cv_score_l = cv_LinearRegression( xM, yV, n_folds = n_folds, scoring = scoring, disp = disp)
		cv_score_le.extend( cv_score_l)
		
	o_d = {'mean': np.mean( cv_score_le),
		   'std': np.std( cv_score_le),
		   'list': cv_score_le}
	
	if disp or ldisp:
		print('{0}: mean(+/-std) --> {1}(+/-{2})'.format( scoring, o_d['mean'], o_d['std']))
		
	return o_d

def cv_LinearRegression_ci_It( xM, yV, n_folds = 5, scoring = 'median_absolute_error', N_it = 10, disp = False, ldisp = False):
	"""
	N_it times iteration is performed for cross_validation in order to make further average effect. 
	The flag of 'disp' is truned off so each iteration will not shown.  
	"""
	cv_score_le = list()
	ci_le = list()
	for ni in range( N_it):
		cv_score_l, ci_l = cv_LinearRegression_ci( xM, yV, n_folds = n_folds, scoring = scoring, disp = disp)
		cv_score_le.extend( cv_score_l)
		ci_le.extend( ci_l)
		
	o_d = {'mean': np.mean( cv_score_le),
		   'std': np.std( cv_score_le),
		   'list': cv_score_le,
		   'ci': ci_le}
	
	if disp or ldisp:
		print('{0}: mean(+/-std) --> {1}(+/-{2})'.format( scoring, o_d['mean'], o_d['std']))
		
	return o_d

def cv_LinearRegression_ci_pred_It( xM, yV, n_folds = 5, scoring = 'median_absolute_error', N_it = 10, disp = False, ldisp = False):
	"""
	N_it times iteration is performed for cross_validation in order to make further average effect. 
	The flag of 'disp' is truned off so each iteration will not shown.  
	"""
	cv_score_le = list()
	ci_le = list()
	yVp_ltype_l = list() # yVp_ltype is list type of yVp not matrix type
	for ni in range( N_it):
		cv_score_l, ci_l, yVp_ltype = cv_LinearRegression_ci_pred( xM, yV, n_folds = n_folds, scoring = scoring, disp = disp)
		cv_score_le.extend( cv_score_l)
		ci_le.extend( ci_l)
		yVp_ltype_l.append( yVp_ltype)
		
	o_d = {'mean': np.mean( cv_score_le),
		   'std': np.std( cv_score_le),
		   'list': cv_score_le,
		   'ci': ci_le,
		   'yVp': yVp_ltype_l}
	
	if disp or ldisp:
		print('{0}: mean(+/-std) --> {1}(+/-{2})'.format( scoring, o_d['mean'], o_d['std']))
		
	return o_d

def cv_LOO( xM, yV, disp = False, ldisp = False):
	"""
	This is a specialized function for LOO crossvadidation. 
	"""
	# print("This is cv_LOO().")

	n_folds = xM.shape[0] # for LOO CV
	return cv_LinearRegression_ci_pred_full_It( xM, yV, n_folds = n_folds, N_it = 1, 
									shuffle = False, disp = disp, ldisp = ldisp)

def cv_LOO_Ridge( xM, yV, alpha, disp = False, ldisp = False):
	"""
	This is a specialized function for LOO crossvadidation. 
	"""
	n_folds = xM.shape[0] # for LOO CV
	return cv_LinearRegression_ci_pred_full_It_Ridge( xM, yV, alpha, n_folds = n_folds, N_it = 1, 
									shuffle = False, disp = disp, ldisp = ldisp)


def cv_LinearRegression_ci_pred_full_It_Ridge( xM, yV, alpha, n_folds = 5, N_it = 10, 
									shuffle = True, disp = False, ldisp = False):
	"""
	N_it times iteration is performed for cross_validation in order to make further average effect. 
	The flag of 'disp' is truned off so each iteration will not shown.  
	"""
	cv_score_le = list()
	ci_le = list()
	yVp_ltype_l = list() # yVp_ltype is list type of yVp not matrix type
	for ni in range( N_it):
		cv_score_l, ci_l, yVp_ltype = cv_LinearRegression_ci_pred_full_Ridge( xM, yV, alpha,
									n_folds = n_folds, shuffle = shuffle, disp = disp)
		cv_score_le.extend( cv_score_l)
		ci_le.extend( ci_l)
		yVp_ltype_l.append( yVp_ltype)

	# List is not used if N_it is one
	if N_it == 1:
		yVp_ltype_l = yVp_ltype_l[0]
		
	o_d = {'median_abs_err': np.median( cv_score_le),
		   'mean_abs_err': np.mean( cv_score_le),
		   'std_abs_err': np.std( cv_score_le),
		   'list': cv_score_le,
		   'ci': ci_le,
		   'yVp': yVp_ltype_l}
	
	return o_d

def cv_LinearRegression_ci_pred_full_It( xM, yV, n_folds = 5, N_it = 10, 
									shuffle = True, disp = False, ldisp = False):
	"""
	N_it times iteration is performed for cross_validation in order to make further average effect. 
	The flag of 'disp' is truned off so each iteration will not shown.  
	"""
	cv_score_le = list()
	ci_le = list()
	yVp_ltype_l = list() # yVp_ltype is list type of yVp not matrix type
	for ni in range( N_it):
		cv_score_l, ci_l, yVp_ltype = cv_LinearRegression_ci_pred_full( xM, yV, 
									n_folds = n_folds, shuffle = shuffle, disp = disp)
		cv_score_le.extend( cv_score_l)
		ci_le.extend( ci_l)
		yVp_ltype_l.append( yVp_ltype)

	# List is not used if N_it is one
	if N_it == 1:
		yVp_ltype_l = yVp_ltype_l[0]
		
	o_d = {'median_abs_err': np.median( cv_score_le),
		   'mean_abs_err': np.mean( cv_score_le),
		   'std_abs_err': np.std( cv_score_le),
		   'list': cv_score_le,
		   'ci': ci_le,
		   'yVp': yVp_ltype_l}
	
	return o_d


def mdae_no_regression( xM, yV, disp = False, ldisp = False):
	"""
	Median absloute error (Mdae) is calculated without any (linear) regression.
	"""
	xM_a = np.array( xM)
	yV_a = np.array( yV)

	ae_l = [ np.abs(x - y) for x, y in zip(xM_a[:,0], yV_a[:, 0])]

	return np.median( ae_l)


def cv_LinearRegression_A( xM, yV, s_l):
	lr = linear_model.LinearRegression()
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=5, shuffle=True)
	r2_l = list()
	for train, test in kf5:
		xM_shuffle = np.concatenate( (xM[ train, :], xM[ test, :]), axis = 0)
		# print xM_shuffle.shape

		A_all = j3x.jpyx.calc_tm_sim_M( xM_shuffle)
		A = A_all

		s_l_shuffle = [s_l[x] for x in train] #train
		s_l_shuffle.extend( [s_l[x] for x in test] ) #test
		molw_l = jchem.rdkit_molwt( s_l_shuffle)

		A_molw = A

		A_molw_train = A_molw[:len(train), :]
		A_molw_test = A_molw[len(train):, :]

		print(A_molw_train.shape, yV[ train, 0].shape)
		lr.fit( A_molw_train, yV[ train, 0])

		#print A_molw_test.shape, yV[ test, 0].shape
		r2_l.append( lr.score( A_molw_test, yV[ test, 0]))

	print('R^2 mean, std -->', np.mean( r2_l), np.std( r2_l))

	return r2_l	

def cv_LinearRegression_Asupervising( xM, yV, s_l):
	lr = linear_model.LinearRegression()
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=5, shuffle=True)
	r2_l = list()
	for train, test in kf5:
		xM_shuffle = np.concatenate( (xM[ train, :], xM[ test, :]), axis = 0)
		#print xM_shuffle.shape

		A_all = j3x.jpyx.calc_tm_sim_M( xM_shuffle)
		A = A_all[ :, :len(train)]

		s_l_shuffle = [s_l[x] for x in train] #train
		s_l_shuffle.extend( [s_l[x] for x in test] ) #test
		molw_l = jchem.rdkit_molwt( s_l_shuffle)

		A_molw = A

		A_molw_train = A_molw[:len(train), :]
		A_molw_test = A_molw[len(train):, :]

		print(A_molw_train.shape, yV[ train, 0].shape)
		lr.fit( A_molw_train, yV[ train, 0])

		#print A_molw_test.shape, yV[ test, 0].shape
		r2_l.append( lr.score( A_molw_test, yV[ test, 0]))

	print('R^2 mean, std -->', np.mean( r2_l), np.std( r2_l))

	return r2_l	

def cv_LinearRegression_Asupervising_molw( xM, yV, s_l):
	
	lr = linear_model.LinearRegression()
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=5, shuffle=True)
	r2_l = list()
	
	for train, test in kf5:
		xM_shuffle = np.concatenate( (xM[ train, :], xM[ test, :]), axis = 0)
		# print xM_shuffle.shape

		A_all = j3x.jpyx.calc_tm_sim_M( xM_shuffle)
		A = A_all[ :, :len(train)]
		#print 'A.shape', A.shape

		s_l_shuffle = [s_l[x] for x in train] #train
		s_l_shuffle.extend( [s_l[x] for x in test] ) #test
		molw_l = jchem.rdkit_molwt( s_l_shuffle)

		A_molw = jchem.add_new_descriptor( A, molw_l)

		A_molw_train = A_molw[:len(train), :]
		A_molw_test = A_molw[len(train):, :]

		#print A_molw_train.shape, yV[ train, 0].shape
		lr.fit( A_molw_train, yV[ train, 0])

		#print A_molw_test.shape, yV[ test, 0].shape
		r2_l.append( lr.score( A_molw_test, yV[ test, 0]))

	print('R^2 mean, std -->', np.mean( r2_l), np.std( r2_l))

	return r2_l

def cv_Ridge_Asupervising_molw( xM, yV, s_l, alpha):
	
	lr = linear_model.Ridge( alpha = alpha)
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=5, shuffle=True)
	r2_l = list()
	
	for train, test in kf5:
		xM_shuffle = np.concatenate( (xM[ train, :], xM[ test, :]), axis = 0)
		# print xM_shuffle.shape

		A_all = j3x.jpyx.calc_tm_sim_M( xM_shuffle)
		A = A_all[ :, :len(train)]
		#print 'A.shape', A.shape

		s_l_shuffle = [s_l[x] for x in train] #train
		s_l_shuffle.extend( [s_l[x] for x in test] ) #test
		molw_l = jchem.rdkit_molwt( s_l_shuffle)

		A_molw = jchem.add_new_descriptor( A, molw_l)

		A_molw_train = A_molw[:len(train), :]
		A_molw_test = A_molw[len(train):, :]

		#print A_molw_train.shape, yV[ train, 0].shape
		lr.fit( A_molw_train, yV[ train, 0])

		#print A_molw_test.shape, yV[ test, 0].shape
		r2_l.append( lr.score( A_molw_test, yV[ test, 0]))

	print('R^2 mean, std -->', np.mean( r2_l), np.std( r2_l))

	return r2_l

def cv_Ridge_Asupervising_2fp( xM1, xM2, yV, s_l, alpha):
	
	lr = linear_model.Ridge( alpha = alpha)
	kf5 = cross_validation.KFold( len(s_l), n_folds=5, shuffle=True)
	r2_l = list()
	
	for train, test in kf5:
		xM1_shuffle = np.concatenate( (xM1[ train, :], xM1[ test, :]), axis = 0)
		xM2_shuffle = np.concatenate( (xM2[ train, :], xM2[ test, :]), axis = 0)
		# print xM_shuffle.shape

		A1_redundant = j3x.jpyx.calc_tm_sim_M( xM1_shuffle)
		A1 = A1_redundant[ :, :len(train)]
		A2_redundant = j3x.jpyx.calc_tm_sim_M( xM2_shuffle)
		A2 = A2_redundant[ :, :len(train)]
		#print 'A.shape', A.shape

		s_l_shuffle = [s_l[x] for x in train] #train
		s_l_shuffle.extend( [s_l[x] for x in test] ) #test
		molw_l = jchem.rdkit_molwt( s_l_shuffle)
		molwV = np.mat( molw_l).T

		#A_molw = jchem.add_new_descriptor( A, molw_l)
		print(A1.shape, A2.shape, molwV.shape)
		# A_molw = np.concatenate( (A1, A2, molwV), axis = 1)
		A_molw = np.concatenate( (A1, A2), axis = 1)
		print(A_molw.shape)

		A_molw_train = A_molw[:len(train), :]
		A_molw_test = A_molw[len(train):, :]

		#print A_molw_train.shape, yV[ train, 0].shape
		lr.fit( A_molw_train, yV[ train, 0])

		#print A_molw_test.shape, yV[ test, 0].shape
		r2_l.append( lr.score( A_molw_test, yV[ test, 0]))

	print('R^2 mean, std -->', np.mean( r2_l), np.std( r2_l))

	return r2_l

def cv_Ridge_Asupervising_2fp_molw( xM1, xM2, yV, s_l, alpha):
	
	lr = linear_model.Ridge( alpha = alpha)
	kf5 = cross_validation.KFold( len(s_l), n_folds=5, shuffle=True)
	r2_l = list()
	
	for train, test in kf5:
		xM1_shuffle = np.concatenate( (xM1[ train, :], xM1[ test, :]), axis = 0)
		xM2_shuffle = np.concatenate( (xM2[ train, :], xM2[ test, :]), axis = 0)
		# print xM_shuffle.shape

		A1_redundant = j3x.jpyx.calc_tm_sim_M( xM1_shuffle)
		A1 = A1_redundant[ :, :len(train)]
		A2_redundant = j3x.jpyx.calc_tm_sim_M( xM2_shuffle)
		A2 = A2_redundant[ :, :len(train)]
		#print 'A.shape', A.shape

		s_l_shuffle = [s_l[x] for x in train] #train
		s_l_shuffle.extend( [s_l[x] for x in test] ) #test
		molw_l = jchem.rdkit_molwt( s_l_shuffle)
		molwV = np.mat( molw_l).T

		#A_molw = jchem.add_new_descriptor( A, molw_l)
		print(A1.shape, A2.shape, molwV.shape)
		A_molw = np.concatenate( (A1, A2, molwV), axis = 1)
		print(A_molw.shape)

		A_molw_train = A_molw[:len(train), :]
		A_molw_test = A_molw[len(train):, :]

		#print A_molw_train.shape, yV[ train, 0].shape
		lr.fit( A_molw_train, yV[ train, 0])

		#print A_molw_test.shape, yV[ test, 0].shape
		r2_l.append( lr.score( A_molw_test, yV[ test, 0]))

	print('R^2 mean, std -->', np.mean( r2_l), np.std( r2_l))

	return r2_l

def gs_Ridge_Asupervising_2fp_molw( xM1, xM2, yV, s_l, alpha_l):
	"""
	This 2fp case uses two fingerprints at the same in order to 
	combines their preprocessing versions separately. 
	"""
	r2_l2 = list()	
	for alpha in alpha_l:
		print(alpha)
		r2_l = cv_Ridge_Asupervising_2fp_molw( xM1, xM2, yV, s_l, alpha)
		r2_l2.append( r2_l)
	return r2_l2

def gs_Ridge_Asupervising_molw( xM, yV, s_l, alpha_l):
	r2_l2 = list()	
	for alpha in alpha_l:
		print(alpha)
		r2_l = cv_Ridge_Asupervising_molw( xM, yV, s_l, alpha)
		r2_l2.append( r2_l)
	return r2_l2

def gs_Ridge_Asupervising( xM, yV, s_l, alpha_l):
	r2_l2 = list()	
	for alpha in alpha_l:
		print(alpha)
		r2_l = cv_Ridge_Asupervising( xM, yV, s_l, alpha)
		r2_l2.append( r2_l)
	return r2_l2

def cv_Ridge_Asupervising( xM, yV, s_l, alpha):
	
	lr = linear_model.Ridge( alpha = alpha)
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=5, shuffle=True)
	r2_l = list()
	
	for train, test in kf5:
		xM_shuffle = np.concatenate( (xM[ train, :], xM[ test, :]), axis = 0)
		# print xM_shuffle.shape

		A_all = j3x.jpyx.calc_tm_sim_M( xM_shuffle)
		A = A_all[ :, :len(train)]
		#print 'A.shape', A.shape

		s_l_shuffle = [s_l[x] for x in train] #train
		s_l_shuffle.extend( [s_l[x] for x in test] ) #test
		molw_l = jchem.rdkit_molwt( s_l_shuffle)

		A_molw = A

		A_molw_train = A_molw[:len(train), :]
		A_molw_test = A_molw[len(train):, :]

		#print A_molw_train.shape, yV[ train, 0].shape
		lr.fit( A_molw_train, yV[ train, 0])

		#print A_molw_test.shape, yV[ test, 0].shape
		r2_l.append( lr.score( A_molw_test, yV[ test, 0]))

	print('R^2 mean, std -->', np.mean( r2_l), np.std( r2_l))

	return r2_l

def gs_RidgeByLasso_kf_ext( xM, yV, alphas_log_l):

	kf5_ext = cross_validation.KFold( xM.shape[0], n_folds=5, shuffle=True)

	score_l = []
	for ix, (tr, te) in enumerate( kf5_ext):

		print('{}th fold external validation stage ============================'.format( ix + 1))
		xM_in = xM[ tr, :]
		yV_in = yV[ tr, 0]

		print('First Ridge Stage')
		gs1 = gs_Lasso( xM_in, yV_in, alphas_log_l[0])
		print('Best score:', gs1.best_score_)
		print('Best param:', gs1.best_params_)
		print(gs1.grid_scores_)


		nz_idx = gs1.best_estimator_.sparse_coef_.indices
		xM_in_nz = xM_in[ :, nz_idx]

		print('Second Lasso Stage')
		gs2 = gs_Ridge( xM_in_nz, yV_in, alphas_log_l[1])
		print('Best score:', gs2.best_score_)
		print('Best param:', gs2.best_params_)
		print(gs2.grid_scores_)

		print('External Validation Stage')
		# Obtain prediction model by whole data including internal validation data
		alpha = gs2.best_params_['alpha']
		clf = linear_model.Ridge( alpha = alpha)
		clf.fit( xM_in_nz, yV_in)

		xM_out = xM[ te, :]
		yV_out = yV[ te, 0]
		xM_out_nz = xM_out[:, nz_idx]
		score = clf.score( xM_out_nz, yV_out)

		print(score)		
		score_l.append( score)

		print('')

	print('all scores:', score_l)
	print('average scores:', np.mean( score_l))

	return score_l

def _gs_SVR_r0( xM, yV, svr_params):

	print(xM.shape, yV.shape)

	clf = svm.SVR()
	#parmas = {'alpha': np.logspace(1, -1, 9)}
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=5, shuffle=True)
	gs = grid_search.GridSearchCV( clf, svr_params, scoring = 'r2', cv = kf5, n_jobs = -1)

	gs.fit( xM, yV.A1)

	return gs

def _gs_SVR_r1( xM, yV, svr_params, n_folds = 5):

	print(xM.shape, yV.shape)

	clf = svm.SVR()
	#parmas = {'alpha': np.logspace(1, -1, 9)}
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=n_folds, shuffle=True)
	gs = grid_search.GridSearchCV( clf, svr_params, scoring = 'r2', cv = kf5, n_jobs = -1)

	gs.fit( xM, yV.A1)

	return gs

def gs_SVR( xM, yV, svr_params, n_folds = 5, n_jobs = -1):

	print(xM.shape, yV.shape)

	clf = svm.SVR()
	#parmas = {'alpha': np.logspace(1, -1, 9)}
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=n_folds, shuffle=True)
	gs = grid_search.GridSearchCV( clf, svr_params, scoring = 'r2', cv = kf5, n_jobs = n_jobs)

	gs.fit( xM, yV.A1)

	return gs

def _gs_SVC_r0( xM, yVc, params):
	"""
	Since classification is considered, we use yVc which includes digital values 
	whereas yV can include float point values.
	"""

	print(xM.shape, yVc.shape)

	clf = svm.SVC()
	#parmas = {'alpha': np.logspace(1, -1, 9)}
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=5, shuffle=True)
	gs = grid_search.GridSearchCV( clf, params, cv = kf5, n_jobs = -1)

	gs.fit( xM, yVc)

	return gs

def gs_SVC( xM, yVc, params, n_folds = 5):
	"""
	Since classification is considered, we use yVc which includes digital values 
	whereas yV can include float point values.
	"""

	print(xM.shape, yVc.shape)

	clf = svm.SVC()
	#parmas = {'alpha': np.logspace(1, -1, 9)}
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=n_folds, shuffle=True)
	gs = grid_search.GridSearchCV( clf, params, cv = kf5, n_jobs = -1)

	gs.fit( xM, yVc)

	return gs


def gs_SVRByLasso_kf_ext( xM, yV, alphas_log, svr_params):

	kf5_ext = cross_validation.KFold( xM.shape[0], n_folds=5, shuffle=True)

	score_l = []
	for ix, (tr, te) in enumerate( kf5_ext):

		print('{}th fold external validation stage ============================'.format( ix + 1))
		xM_in = xM[ tr, :]
		yV_in = yV[ tr, 0]

		print('First Ridge Stage')
		gs1 = gs_Lasso( xM_in, yV_in, alphas_log)
		print('Best score:', gs1.best_score_)
		print('Best param:', gs1.best_params_)
		print(gs1.grid_scores_)


		nz_idx = gs1.best_estimator_.sparse_coef_.indices
		xM_in_nz = xM_in[ :, nz_idx]

		print('Second Lasso Stage')
		gs2 = gs_SVR( xM_in_nz, yV_in, svr_params)
		print('Best score:', gs2.best_score_)
		print('Best param:', gs2.best_params_)
		print(gs2.grid_scores_)

		print('External Validation Stage')
		# Obtain prediction model by whole data including internal validation data
		C = gs2.best_params_['C']
		gamma = gs2.best_params_['gamma']
		epsilon = gs2.best_params_['epsilon']

		clf = svm.SVR( C = C, gamma = gamma, epsilon = epsilon)
		clf.fit( xM_in_nz, yV_in.A1)

		xM_out = xM[ te, :]
		yV_out = yV[ te, 0]
		xM_out_nz = xM_out[:, nz_idx]
		score = clf.score( xM_out_nz, yV_out.A1)

		print(score)		
		score_l.append( score)

		print('')

	print('all scores:', score_l)
	print('average scores:', np.mean( score_l))

	return score_l	

def gs_SVRByLasso( xM, yV, alphas_log, svr_params):

	kf5_ext = cross_validation.KFold( xM.shape[0], n_folds=5, shuffle=True)

	score1_l = []
	score_l = []
	for ix, (tr, te) in enumerate( kf5_ext):

		print('{}th fold external validation stage ============================'.format( ix + 1))
		xM_in = xM[ tr, :]
		yV_in = yV[ tr, 0]

		print('First Ridge Stage')
		gs1 = gs_Lasso( xM_in, yV_in, alphas_log)
		print('Best score:', gs1.best_score_)
		print('Best param:', gs1.best_params_)
		print(gs1.grid_scores_)
		score1_l.append( gs1.best_score_)


		nz_idx = gs1.best_estimator_.sparse_coef_.indices
		xM_in_nz = xM_in[ :, nz_idx]

		print('Second Lasso Stage')
		gs2 = gs_SVR( xM_in_nz, yV_in, svr_params)
		print('Best score:', gs2.best_score_)
		print('Best param:', gs2.best_params_)
		print(gs2.grid_scores_)

		print('External Validation Stage')
		# Obtain prediction model by whole data including internal validation data
		C = gs2.best_params_['C']
		gamma = gs2.best_params_['gamma']
		epsilon = gs2.best_params_['epsilon']

		clf = svm.SVR( C = C, gamma = gamma, epsilon = epsilon)
		clf.fit( xM_in_nz, yV_in.A1)

		xM_out = xM[ te, :]
		yV_out = yV[ te, 0]
		xM_out_nz = xM_out[:, nz_idx]
		score = clf.score( xM_out_nz, yV_out.A1)

		print(score)		
		score_l.append( score)

		print('')

	print('all scores:', score_l)
	print('average scores:', np.mean( score_l))

	print('First stage scores', score1_l)
	print('Average first stage scores', np.mean( score1_l))

	return score_l, score1_l

def gs_ElasticNet( xM, yV, en_params):

	print(xM.shape, yV.shape)

	clf = linear_model.ElasticNet()
	kf5 = cross_validation.KFold( xM.shape[0], n_folds=5, shuffle=True)
	gs = grid_search.GridSearchCV( clf, en_params, scoring = 'r2', cv = kf5, n_jobs = -1)

	gs.fit( xM, yV)

	return gs

def gs_SVRByElasticNet( xM, yV, en_params, svr_params):

	kf5_ext = cross_validation.KFold( xM.shape[0], n_folds=5, shuffle=True)

	score1_l = []
	score_l = []
	for ix, (tr, te) in enumerate( kf5_ext):

		print('{}th fold external validation stage ============================'.format( ix + 1))
		xM_in = xM[ tr, :]
		yV_in = yV[ tr, 0]

		print('First Ridge Stage')
		gs1 = gs_ElasticNet( xM_in, yV_in, en_params)
		print('Best score:', gs1.best_score_)
		print('Best param:', gs1.best_params_)
		print(gs1.grid_scores_)
		score1_l.append( gs1.best_score_)


		nz_idx = gs1.best_estimator_.sparse_coef_.indices
		xM_in_nz = xM_in[ :, nz_idx]

		print('Second Lasso Stage')
		gs2 = gs_SVR( xM_in_nz, yV_in, svr_params)
		print('Best score:', gs2.best_score_)
		print('Best param:', gs2.best_params_)
		print(gs2.grid_scores_)

		print('External Validation Stage')
		# Obtain prediction model by whole data including internal validation data
		C = gs2.best_params_['C']
		gamma = gs2.best_params_['gamma']
		epsilon = gs2.best_params_['epsilon']

		clf = svm.SVR( C = C, gamma = gamma, epsilon = epsilon)
		clf.fit( xM_in_nz, yV_in.A1)

		xM_out = xM[ te, :]
		yV_out = yV[ te, 0]
		xM_out_nz = xM_out[:, nz_idx]
		score = clf.score( xM_out_nz, yV_out.A1)

		print(score)		
		score_l.append( score)

		print('')

	print('all scores:', score_l)
	print('average scores:', np.mean( score_l))

	print('First stage scores', score1_l)
	print('Average first stage scores', np.mean( score1_l))

	return score_l, score1_l

def gs_GPByLasso( xM, yV, alphas_log):

	kf5_ext = cross_validation.KFold( xM.shape[0], n_folds=5, shuffle=True)

	score1_l = []
	score_l = []
	for ix, (tr, te) in enumerate( kf5_ext):

		print('{}th fold external validation stage ============================'.format( ix + 1))
		xM_in = xM[ tr, :]
		yV_in = yV[ tr, 0]

		print('First Ridge Stage')
		gs1 = gs_Lasso( xM_in, yV_in, alphas_log)
		print('Best score:', gs1.best_score_)
		print('Best param:', gs1.best_params_)
		print(gs1.grid_scores_)
		score1_l.append( gs1.best_score_)


		nz_idx = gs1.best_estimator_.sparse_coef_.indices
		xM_in_nz = xM_in[ :, nz_idx]

		print('Second GP Stage')		
		Xa_in_nz = np.array( xM_in_nz)
		ya_in = np.array( yV_in)

		xM_out = xM[ te, :]
		yV_out = yV[ te, 0]
		xM_out_nz = xM_out[:, nz_idx]
		Xa_out_nz = np.array( xM_out_nz)
		ya_out = np.array( yV_out)

		#jgp = gp.GaussianProcess( Xa_in_nz, ya_in, Xa_out_nz, ya_out)
		# the y array should be send as [:,0] form to be sent as vector array
		jgp = gp.GaussianProcess( Xa_in_nz, ya_in[:,0], Xa_out_nz, ya_out[:,0])
		jgp.optimize_noise_and_amp()
		jgp.run_gp()

		#ya_out_pred = np.mat(jgp.predicted_targets)
		ya_out_pred = jgp.predicted_targets
		#print ya_out[:,0].shape, jgp.predicted_targets.shape

		r2, rmse = regress_show( ya_out[:,0], ya_out_pred)

		score = r2
		print(score)		
		score_l.append( score)

		print('')

	print('all scores:', score_l)
	print('average scores:', np.mean( score_l))

	print('First stage scores', score1_l)
	print('Average first stage scores', np.mean( score1_l))

	return score_l, score1_l

def show_gs_alpha( grid_scores):
	alphas = np.array([ x[0]['alpha'] for x in grid_scores])
	r2_mean = np.array([ x[1] for x in grid_scores])
	r2_std = np.array([ np.std(x[2]) for x in grid_scores])
	
	r2_mean_pos = r2_mean + r2_std
	r2_mean_neg = r2_mean - r2_std

	plt.semilogx( alphas, r2_mean, 'x-', label = 'E[$r^2$]')
	plt.semilogx( alphas, r2_mean_pos, ':k', label = 'E[$r^2$]+$\sigma$')
	plt.semilogx( alphas, r2_mean_neg, ':k', label = 'E[$r^2$]-$\sigma$')
	plt.grid()
	plt.legend( loc = 2)
	plt.show()

	best_idx = np.argmax( r2_mean)
	best_r2_mean = r2_mean[ best_idx]
	best_r2_std = r2_std[ best_idx]
	best_alpha = alphas[ best_idx]

	print("Best: r2(alpha = {0}) -> mean:{1}, std:{2}".format( best_alpha, best_r2_mean, best_r2_std))

"""
Specialized code for extract results
"""

def make_scoring( scoring):
	"""
	Score is reversed if greater_is_better is False.
	"""
	if scoring == 'r2':
		return metrics.make_scorer( metrics.r2_score)
	elif scoring == 'mean_absolute_error':
		return metrics.make_scorer( metrics.mean_absolute_error, greater_is_better=False)
	elif scoring == 'mean_squared_error':
		return metrics.make_scorer( metrics.mean_squared_error, greater_is_better=False)
	elif scoring == 'median_absolute_error':
		return metrics.make_scorer( metrics.median_absolute_error, greater_is_better=False)
	else:
		raise ValueError("Not supported scoring")


def gs_Ridge( xM, yV, alphas_log = (1, -1, 9), n_folds = 5, n_jobs = -1, scoring = 'r2'):
	"""
	Parameters
	-------------
	scoring: mean_absolute_error, mean_squared_error, median_absolute_error, r2
	"""
	print( 'If scoring is not r2 but error metric, output score is revered for scoring!')
	print(xM.shape, yV.shape)

	clf = linear_model.Ridge()
	#parmas = {'alpha': np.logspace(1, -1, 9)}
	parmas = {'alpha': np.logspace( *alphas_log)}
	kf_n = cross_validation.KFold( xM.shape[0], n_folds=n_folds, shuffle=True)
	gs = grid_search.GridSearchCV( clf, parmas, scoring = scoring, cv = kf_n, n_jobs = n_jobs)

	gs.fit( xM, yV)

	return gs

def gs_Ridge_BIKE( A_list, yV, XX = None, alphas_log = (1, -1, 9), n_folds = 5, n_jobs = -1):
	"""
	As is a list of A matrices where A is similarity matrix. 
	X is a concatened linear descriptors. 
	If no X is used, X can be empty
	"""

	clf = binary_model.BIKE_Ridge( A_list, XX)
	parmas = {'alpha': np.logspace( *alphas_log)}
	ln = A_list[0].shape[0] # ls is the number of molecules.

	kf_n = cross_validation.KFold( ln, n_folds=n_folds, shuffle=True)
	gs = grid_search.GridSearchCV( clf, parmas, scoring = 'r2', cv = kf_n, n_jobs = n_jobs)
	
	AX_idx = np.array([list(range( ln))]).T
	gs.fit( AX_idx, yV)

	return gs

def gs_BIKE_Ridge( A_list, yV, alphas_log = (1, -1, 9), X_concat = None, n_folds = 5, n_jobs = -1):
	"""
	As is a list of A matrices where A is similarity matrix. 
	X is a concatened linear descriptors. 
	If no X is used, X can be empty
	"""

	clf = binary_model.BIKE_Ridge( A_list, X_concat)
	parmas = {'alpha': np.logspace( *alphas_log)}
	ln = A_list[0].shape[0] # ls is the number of molecules.

	kf_n = cross_validation.KFold( ln, n_folds=n_folds, shuffle=True)
	gs = grid_search.GridSearchCV( clf, parmas, scoring = 'r2', cv = kf_n, n_jobs = n_jobs)
	
	AX_idx = np.array([list(range( ln))]).T
	gs.fit( AX_idx, yV)

	return gs


def _cv_r0( method, xM, yV, alpha, n_folds = 5, n_jobs = -1, grid_std = None, graph = True):
	"""
	method can be 'Ridge', 'Lasso'
	cross validation is performed so as to generate prediction output for all input molecules
	"""	
	print(xM.shape, yV.shape)

	clf = getattr( linear_model, method)( alpha = alpha)
	kf_n = cross_validation.KFold( xM.shape[0], n_folds=n_folds, shuffle=True)
	yV_pred = cross_validation.cross_val_predict( clf, xM, yV, cv = kf_n, n_jobs = n_jobs)

	if graph:
		print('The prediction output using cross-validation is given by:')
		jutil.cv_show( yV, yV_pred, grid_std = grid_std)

	return yV_pred

def cv( method, xM, yV, alpha, n_folds = 5, n_jobs = -1, grid_std = None, graph = True, shuffle = True):
	"""
	method can be 'Ridge', 'Lasso'
	cross validation is performed so as to generate prediction output for all input molecules
	Return
	--------
	yV_pred
	"""	
	print(xM.shape, yV.shape)

	clf = getattr( linear_model, method)( alpha = alpha)
	kf_n = cross_validation.KFold( xM.shape[0], n_folds=n_folds, shuffle=shuffle)
	yV_pred = cross_validation.cross_val_predict( clf, xM, yV, cv = kf_n, n_jobs = n_jobs)

	if graph:
		print('The prediction output using cross-validation is given by:')
		jutil.cv_show( yV, yV_pred, grid_std = grid_std)

	return yV_pred

def _cv_LOO_r0( method, xM, yV, alpha, n_jobs = -1, grid_std = None, graph = True):
	"""
	method can be 'Ridge', 'Lasso'
	cross validation is performed so as to generate prediction output for all input molecules
	"""	
	n_folds = xM.shape[0]

	print(xM.shape, yV.shape)

	clf = getattr( linear_model, method)( alpha = alpha)
	kf_n = cross_validation.KFold( xM.shape[0], n_folds=n_folds)
	yV_pred = cross_validation.cross_val_predict( clf, xM, yV, cv = kf_n, n_jobs = n_jobs)

	if graph:
		print('The prediction output using cross-validation is given by:')
		jutil.cv_show( yV, yV_pred, grid_std = grid_std)

	return yV_pred	


def cv_Ridge_BIKE( A_list, yV, XX = None, alpha = 0.5, n_folds = 5, n_jobs = -1, grid_std = None):
	"""
	Older version than cv_Ridge_BIKE
	"""

	clf = binary_model.BIKE_Ridge( A_list, XX, alpha = alpha)
	ln = A_list[0].shape[0] # ls is the number of molecules.
	kf_n = cross_validation.KFold( ln, n_folds=n_folds, shuffle=True)

	AX_idx = np.array([list(range( ln))]).T
	yV_pred = cross_validation.cross_val_predict( clf, AX_idx, yV, cv = kf_n, n_jobs = n_jobs)

	print('The prediction output using cross-validation is given by:')
	jutil.cv_show( yV, yV_pred, grid_std = grid_std)

	return yV_pred

def cv_BIKE_Ridge( A_list, yV, alpha = 0.5, XX = None, n_folds = 5, n_jobs = -1, grid_std = None):

	clf = binary_model.BIKE_Ridge( A_list, XX, alpha = alpha)
	ln = A_list[0].shape[0] # ls is the number of molecules.
	kf_n = cross_validation.KFold( ln, n_folds=n_folds, shuffle=True)

	AX_idx = np.array([list(range( ln))]).T
	yV_pred = cross_validation.cross_val_predict( clf, AX_idx, yV, cv = kf_n, n_jobs = n_jobs)

	print('The prediction output using cross-validation is given by:')
	jutil.cv_show( yV, yV_pred, grid_std = grid_std)

	return yV_pred	


def topscores( gs):
	"""
	return only top scores for ridge and lasso with the best parameters
	"""
	top_score = sorted(gs.grid_scores_, key=itemgetter(1), reverse=True)[0]
	
	print(top_score.parameters)
	print(top_score.cv_validation_scores)

	return top_score.parameters, top_score.cv_validation_scores

def pd_dataframe( param, scores, descriptor = "Morgan(r=6,nB=4096)", graph = True):

	pdw_score = pd.DataFrame()
	
	k_KF = len(  scores)
	pdw_score["descriptor"] = [ descriptor] * k_KF
	pdw_score["regularization"] = ["Ridge"] * k_KF
	pdw_score["alpha"] = [param['alpha']] * k_KF
	pdw_score["KFold"] = list(range( 1, k_KF + 1))
	pdw_score["r2"] = scores

	if graph:
		pdw_score['r2'].plot( kind = 'box')

	return pdw_score


def ridge( xM, yV, alphas_log, descriptor = "Morgan(r=6,nB=4096)", n_folds = 5, n_jobs = -1):
	gs = gs_Ridge( xM, yV, alphas_log = alphas_log, n_folds = n_folds, n_jobs = n_jobs)

	param, scores = topscores( gs)

	pdw_score = pd_dataframe( param, scores, descriptor = descriptor)
	
	print('Ridge(alpha={0}) = {1}'.format( gs.best_params_['alpha'], gs.best_score_))
	print(pdw_score)

	return pdw_score


def gs( method, xM, yV, alphas_log):
	"""
	It simplifies the function name into the two alphabets
	by adding one more argument which is the name of a method.
	"""

	if method == "Lasso":
		return gs_Lasso( xM, yV, alphas_log)
	elif method == "Ridge":
		return gs_Ridge( xM, yV, alphas_log)	
	else:
		raise NameError("The method of {} is not supported".format( method))
