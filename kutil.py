"""
some utility which I made.
Editor - Sungjin Kim, 2015-4-17
"""
#Common library
from sklearn import linear_model, svm, model_selection, metrics

import matplotlib.pyplot as plt
import numpy as np
import time

#import subprocess
import pandas as pd
import itertools
import random

#My personal library
import j3x.jpyx
import kchem

def _sleast_r0( a = '1000', ln = 10):
	"It returns 0 filled string with the length of ln."
	if ln > len(a):
		return '0'*(ln - len(a)) + a
	else:
		return a[-ln:]

def sleast( a = '1000', ln = 10):
	"It returns 0 filled string with the length of ln."
	if ln > len(a):
		return a + '0'*(ln - len(a))
	else:
		return a


def int_bp( b_ch):
	"map '0' --> -1, '1' --> -1"
	b_int = int( b_ch)
	return 1 - 2 * b_int

def prange( pat, st, ed, ic=1):
	ar = []
	for ii in range( st, ed, ic):
		ar.extend( [ii + jj for jj in pat])

	return [x for x in ar if x < ed]

class Timer:    
	def __enter__(self):
		self.start = time.clock()
		return self

	def __exit__(self, *args):
		self.end = time.clock()
		self.interval = self.end - self.start
		print(( 'Elapsed time: {}sec'.format(self.interval)))

def mlr( RM, yE, disp = True, graph = True):
	clf = linear_model.LinearRegression()
	clf.fit( RM, yE)
	mlr_show( clf, RM, yE, disp = disp, graph = graph)

def mlr3( RM, yE, disp = True, graph = True):
	clf = linear_model.LinearRegression()
	clf.fit( RM, yE)
	mlr_show3( clf, RM, yE, disp = disp, graph = graph)

def mlr3_coef( RM, yE, disp = True, graph = True):
	clf = linear_model.LinearRegression()
	clf.fit( RM, yE)
	mlr_show3( clf, RM, yE, disp = disp, graph = graph)	

	return clf.coef_, clf.intercept_

def mlr4_coef( RM, yE, disp = True, graph = True):
	clf = linear_model.LinearRegression()
	clf.fit( RM, yE)
	mlr_show4( clf, RM, yE, disp = disp, graph = graph)	

	return clf.coef_, clf.intercept_

def mlr_ridge( RM, yE, alpha = 0.5, disp = True, graph = True):
	clf = linear_model.Ridge( alpha = alpha)
	clf.fit( RM, yE)
	mlr_show( clf, RM, yE, disp = disp, graph = graph)

def mlr3_coef_ridge( RM, yE, alpha = 0.5, disp = True, graph = True):
	"""
	Return regression coefficients and intercept
	"""
	clf = linear_model.Ridge( alpha = alpha)
	clf.fit( RM, yE)
	mlr_show( clf, RM, yE, disp = disp, graph = graph)

	return clf.coef_, clf.intercept_

def ann_post( yv, disp = True, graph = True):
	"""
	After ann_pre and shell command, ann_post can be used.
	"""
	df_ann = pd.read_csv( 'ann_out.csv')
	yv_ann = np.mat( df_ann['out'].tolist()).T
	
	r_sqr, RMSE = ann_show( yv, yv_ann, disp = disp, graph = graph)

	return r_sqr, RMSE

def ann_post_range( range_tr, range_val, yv, disp = True, graph = True):
	"""
	After ann_pre and shell command, ann_post can be used.
	"""
	df_ann = pd.read_csv( 'ann_out.csv')
	yv_ann = np.mat( df_ann['out'].tolist()).T
	
	print("Traning:")
	ann_show( yv[range_tr, 0], yv_ann[range_tr, 0], disp = disp, graph = graph)

	print("Validation:")
	r_sqr, RMSE = ann_show( yv[range_val, 0] , yv_ann[range_val, 0], disp = disp, graph = graph)

	return r_sqr, RMSE

def _regress_show_r0( yEv, yEv_calc, disp = True, graph = True, plt_title = None):

	# if the output is a vector and the original is a metrix, 
	# the output is translated to a matrix. 
	if len( np.shape(yEv)) == 2 and len( np.shape(yEv_calc)) == 1:
		yEv_calc = np.mat( yEv_calc).T

	r_sqr, RMSE = kchem.estimate_accuracy( yEv, yEv_calc, disp = disp)
	if graph:
		plt.scatter( yEv.tolist(), yEv_calc.tolist())
		ax = plt.gca()
		lims = [
			np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
			np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
		]
		# now plot both limits against eachother
		#ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
		ax.plot(lims, lims, '-', color = 'pink')
		plt.xlabel('Target')
		plt.ylabel('Prediction')
		if plt_title:
			plt.title( plt_title)
		plt.show()
	return r_sqr, RMSE

def regress_show( yEv, yEv_calc, disp = True, graph = True, plt_title = None):

	# if the output is a vector and the original is a metrix, 
	# the output is translated to a matrix. 
	if len( np.shape(yEv_calc)) == 1:	
		yEv_calc = np.mat( yEv_calc).T
	if len( np.shape(yEv)) == 1:
		yEv = np.mat( yEv).T

	r_sqr, RMSE = kchem.estimate_accuracy( yEv, yEv_calc, disp = disp)
	if graph:
		#plt.scatter( yEv.tolist(), yEv_calc.tolist())	
		plt.figure()	
		ms_sz = max(min( 4000 / yEv.shape[0], 8), 1)
		plt.plot( yEv.tolist(), yEv_calc.tolist(), '.', ms = ms_sz) # Change ms 
		ax = plt.gca()
		lims = [
			np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
			np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
		]
		# now plot both limits against eachother
		#ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
		ax.plot(lims, lims, '-', color = 'pink')
		plt.xlabel('Experiment')
		plt.ylabel('Prediction')
		if plt_title:
			plt.title( plt_title)
		else:
			plt.title( '$r^2$ = {0:.2e}, RMSE = {1:.2e}'.format( r_sqr, RMSE))
		plt.show()
	return r_sqr, RMSE

def regress_show3( yEv, yEv_calc, disp = True, graph = True, plt_title = None):

	# if the output is a vector and the original is a metrix, 
	# the output is translated to a matrix. 

	r_sqr, RMSE, MAE = kchem.estimate_score3( yEv, yEv_calc, disp = disp)
	
	if graph:
		#plt.scatter( yEv.tolist(), yEv_calc.tolist())	
		plt.figure()	
		ms_sz = max(min( 6000 / yEv.shape[0], 8), 3)
		plt.plot( yEv.tolist(), yEv_calc.tolist(), '.', ms = ms_sz) # Change ms 
		ax = plt.gca()
		lims = [
			np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
			np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
		]
		# now plot both limits against eachother
		#ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
		ax.plot(lims, lims, '-', color = 'pink')
		plt.xlabel('Experiment')
		plt.ylabel('Prediction')
		if plt_title:
			plt.title( plt_title)
		else:
			plt.title( '$r^2$ = {0:.2e}, RMSE = {1:.2e}, MAE = {2:.2e}'.format( r_sqr, RMSE, MAE))
		plt.show()
	
	return r_sqr, RMSE, MAE

def _regress_show4_r0( yEv, yEv_calc, disp = True, graph = True, plt_title = None, ms_sz = None):

	# if the output is a vector and the original is a metrix, 
	# the output is translated to a matrix. 

	r_sqr, RMSE, MAE, DAE = estimate_accuracy4( yEv, yEv_calc, disp = disp)
	
	if graph:
		#plt.scatter( yEv.tolist(), yEv_calc.tolist())	
		plt.figure()	
		if ms_sz is None:
			ms_sz = max(min( 6000 / yEv.shape[0], 8), 3)
		plt.plot( yEv.tolist(), yEv_calc.tolist(), '.', ms = ms_sz) # Change ms 
		ax = plt.gca()
		lims = [
			np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
			np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
		]
		# now plot both limits against eachother
		#ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
		ax.plot(lims, lims, '-', color = 'pink')
		plt.xlabel('Experiment')
		plt.ylabel('Prediction')
		if plt_title is None:
			plt.title( '$r^2$={0:.1e}, RMSE={1:.1e}, MAE={2:.1e}, MedAE={3:.1e}'.format( r_sqr, RMSE, MAE, DAE))
		elif plt_title != "": 
			plt.title( plt_title)
		# plt.show()
	
	return r_sqr, RMSE, MAE, DAE

def regress_show4( yEv, yEv_calc, disp = True, graph = True, plt_title = None, ms_sz = None):

	# if the output is a vector and the original is a metrix, 
	# the output is translated to a matrix. 

	r_sqr, RMSE, MAE, DAE = estimate_accuracy4( yEv, yEv_calc, disp = disp)
	
	if graph:
		#plt.scatter( yEv.tolist(), yEv_calc.tolist())	
		plt.figure()	
		if ms_sz is None:
			ms_sz = max(min( 6000 / yEv.shape[0], 8), 3)
		# plt.plot( yEv.tolist(), yEv_calc.tolist(), '.', ms = ms_sz) # Change ms 
		plt.scatter( yEv.tolist(), yEv_calc.tolist(), s = ms_sz) 
		ax = plt.gca()
		lims = [
			np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
			np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
		]
		# now plot both limits against eachother
		#ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
		ax.plot(lims, lims, '-', color = 'pink')
		plt.xlabel('Experiment')
		plt.ylabel('Prediction')
		if plt_title is None:
			plt.title( '$r^2$={0:.1e}, RMSE={1:.1e}, MAE={2:.1e}, MedAE={3:.1e}'.format( r_sqr, RMSE, MAE, DAE))
		elif plt_title != "": 
			plt.title( plt_title)
		# plt.show()
	
	return r_sqr, RMSE, MAE, DAE

def cv_show( yEv, yEv_calc, disp = True, graph = True, grid_std = None):

	# if the output is a vector and the original is a metrix, 
	# the output is translated to a matrix. 
	if len( np.shape(yEv_calc)) == 1:	
		yEv_calc = np.mat( yEv_calc).T
	if len( np.shape(yEv)) == 1:
		yEv = np.mat( yEv).T

	r_sqr, RMSE = kchem.estimate_accuracy( yEv, yEv_calc, disp = disp)
	if graph:
		#plt.scatter( yEv.tolist(), yEv_calc.tolist())	
		plt.figure()	
		ms_sz = max(min( 4000 / yEv.shape[0], 8), 1)
		plt.plot( yEv.tolist(), yEv_calc.tolist(), '.', ms = ms_sz) # Change ms 
		ax = plt.gca()
		lims = [
			np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
			np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
		]
		# now plot both limits against eachother
		#ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
		ax.plot(lims, lims, '-', color = 'pink')
		plt.xlabel('Experiment')
		plt.ylabel('Prediction')
		if grid_std:
			plt.title( '($r^2$, std) = ({0:.2e}, {1:.2e}), RMSE = {2:.2e}'.format( r_sqr, grid_std, RMSE))
		else:
			plt.title( '$r^2$ = {0:.2e}, RMSE = {1:.2e}'.format( r_sqr, RMSE))
		plt.show()
	return r_sqr, RMSE

ann_show = regress_show

def mlr_show( clf, RMv, yEv, disp = True, graph = True):
	yEv_calc = clf.predict( RMv)

	if len( np.shape(yEv)) == 2 and len( np.shape(yEv_calc)) == 1:
		yEv_calc = np.mat( yEv_calc).T

	r_sqr, RMSE = kchem.estimate_accuracy( yEv, yEv_calc, disp = disp)
	if graph:
		plt.figure()
		ms_sz = max(min( 4000 / yEv.shape[0], 8), 1)
		plt.plot( yEv.tolist(), yEv_calc.tolist(), '.', ms = ms_sz)
		ax = plt.gca()
		lims = [
			np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
			np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
		]
		# now plot both limits against eachother
		#ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
		ax.plot(lims, lims, '-', color = 'pink')
		plt.xlabel('Experiment')
		plt.ylabel('Prediction')
		plt.title( '$r^2$ = {0:.2e}, RMSE = {1:.2e}'.format( r_sqr, RMSE))
		plt.show()

	return r_sqr, RMSE

def estimate_accuracy4(yEv, yEv_calc, disp = False):
	r_sqr = metrics.r2_score( yEv, yEv_calc)
	RMSE = np.sqrt( metrics.mean_squared_error( yEv, yEv_calc))
	MAE = metrics.mean_absolute_error( yEv, yEv_calc)
	DAE = metrics.median_absolute_error( yEv, yEv_calc)

	if disp:
		print("r^2={0:.2e}, RMSE={1:.2e}, MAE={2:.2e}, DAE={3:.2e}".format( r_sqr, RMSE, MAE, DAE))

	return r_sqr, RMSE, MAE, DAE

def mlr_show3( clf, RMv, yEv, disp = True, graph = True):
	yEv_calc = clf.predict( RMv)

	if len( np.shape(yEv)) == 2 and len( np.shape(yEv_calc)) == 1:
		yEv_calc = np.mat( yEv_calc).T

	r_sqr, RMSE, aae = kchem.estimate_accuracy3( yEv, yEv_calc, disp = disp)
	if graph:
		plt.figure()
		ms_sz = max(min( 4000 / yEv.shape[0], 8), 1)
		plt.plot( yEv.tolist(), yEv_calc.tolist(), '.', ms = ms_sz)
		ax = plt.gca()
		lims = [
			np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
			np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
		]
		# now plot both limits against eachother
		#ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
		ax.plot(lims, lims, '-', color = 'pink')
		plt.xlabel('Experiment')
		plt.ylabel('Prediction')
		plt.title( '$r^2$={0:.2e}, RMSE={1:.2e}, AAE={2:.2e}'.format( r_sqr, RMSE, aae))
		plt.show()

	return r_sqr, RMSE, aae

def mlr_show4( clf, RMv, yEv, disp = True, graph = True):
	yEv_calc = clf.predict( RMv)

	if len( np.shape(yEv)) == 2 and len( np.shape(yEv_calc)) == 1:
		yEv_calc = np.mat( yEv_calc).T

	r_sqr, RMSE, MAE, DAE = estimate_accuracy4( yEv, yEv_calc, disp = disp)

	if graph:
		plt.figure()
		ms_sz = max(min( 4000 / yEv.shape[0], 8), 1)
		plt.plot( yEv.tolist(), yEv_calc.tolist(), '.', ms = ms_sz)
		ax = plt.gca()
		lims = [
			np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
			np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
		]
		# now plot both limits against eachother
		#ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
		ax.plot(lims, lims, '-', color = 'pink')
		plt.xlabel('Experiment')
		plt.ylabel('Prediction')
		#plt.title( '$r^2$={0:.2e}, RMSE={1:.2e}, AAE={2:.2e}'.format( r_sqr, RMSE, aae))
		plt.title( '$r^2$={0:.1e},$\sigma$={1:.1e},MAE={2:.1e},DAE={3:.1e}'.format( r_sqr, RMSE, MAE, DAE))
		plt.show()

	return r_sqr, RMSE, MAE, DAE


def _mlr_val_r0( RM, yE, disp = True, graph = True):
	clf = linear_model.LinearRegression()
	clf.fit( RM[::2,:], yE[::2,0])

	print('Training result')
	mlr_show( clf, RM[::2, :], yE[::2, 0], disp = disp, graph = graph)

	print('Validation result')
	mlr_show( clf, RM[1::2, :], yE[1::2, 0], disp = disp, graph = graph)

def mlr_val( RM, yE, disp = True, graph = True, rate = 2, more_train = True, center = None):
	"""
	Validation is peformed as much as the given ratio.
	"""
	RMt, yEt, RMv, yEv = kchem.get_valid_mode_data( RM, yE, rate = rate, more_train = more_train, center = center)

	clf = linear_model.LinearRegression()	
	clf.fit( RMt, yEt)

	print('Training result')
	mlr_show( clf, RMt, yEt, disp = disp, graph = graph)

	print('Validation result')
	r_sqr, RMSE = mlr_show( clf, RMv, yEv, disp = disp, graph = graph)

	return r_sqr, RMSE

def svr_val( RM, yE, C = 1.0, epsilon = 0.1, disp = True, graph = True, rate = 2, more_train = True, center = None):
	"""
	Validation is peformed as much as the given ratio.
	"""
	RMt, yEt, RMv, yEv = kchem.get_valid_mode_data( RM, yE, rate = rate, more_train = more_train, center = center)

	clf = svm.SVR( C = C, epsilon = epsilon)	
	clf.fit( RMt, yEt.A1)

	print('Training result')
	mlr_show( clf, RMt, yEt, disp = disp, graph = graph)

	print('Validation result')
	r_sqr, RMSE = mlr_show( clf, RMv, yEv, disp = disp, graph = graph)

	return r_sqr, RMSE	

def mlr_val_ridge( RM, yE, rate = 2, more_train = True, center = None, alpha = 0.5, disp = True, graph = True):
	"""
	Validation is peformed as much as the given ratio.
	"""
	RMt, yEt, RMv, yEv = kchem.get_valid_mode_data( RM, yE, rate = rate, more_train = more_train, center = center)

	print("Ridge: alpha = {}".format( alpha))
	clf = linear_model.Ridge( alpha = alpha)	
	clf.fit( RMt, yEt)

	print('Weight value')
	#print clf.coef_.flatten()
	plt.plot( clf.coef_.flatten())
	plt.grid()
	plt.xlabel('Tap')
	plt.ylabel('Weight')
	plt.title('Linear Regression Weights')
	plt.show()

	print('Training result')
	mlr_show( clf, RMt, yEt, disp = disp, graph = graph)

	print('Validation result')
	r_sqr, RMSE = mlr_show( clf, RMv, yEv, disp = disp, graph = graph)

	return r_sqr, RMSE

def mlr_val_avg_2( RM, yE, disp = False, graph = False):
	"""
	Validation is peformed as much as the given ratio.
	"""
	r_sqr_list, RMSE_list = [], []
	vseq_list = []
	org_seq = list(range( len( yE)))
	for	v_seq in itertools.combinations( org_seq, 2):
		t_seq = [x for x in org_seq if x not in v_seq]

		RMt, yEt = RM[ t_seq, :], yE[ t_seq, 0]
		RMv, yEv = RM[ v_seq, :], yE[ v_seq, 0]

		clf = linear_model.LinearRegression()	
		clf.fit( RMt, yEt)

		#print 'Training result'
		mlr_show( clf, RMt, yEt, disp = disp, graph = graph)

		#print 'Validation result'
		r_sqr, RMSE = mlr_show( clf, RMv, yEv, disp = disp, graph = graph)

		"""
		#This is blocked since vseq_list is returned.
		if r_sqr < 0:
			print 'v_seq:', v_seq, '--> r_sqr = ', r_sqr
		"""

		r_sqr_list.append( r_sqr)
		RMSE_list.append( RMSE)
		vseq_list.append( v_seq)

	print("average r_sqr = {0}, average RMSE = {1}".format( np.average( r_sqr_list), np.average( RMSE_list)))

	return r_sqr_list, RMSE_list, v_seq

def gen_rand_seq( ln, rate):
	vseq = choose( ln, int( ln / rate))
	org_seq = list(range( ln))
	tseq = [x for x in org_seq if x not in vseq]

	return tseq, vseq

def mlr_val_vseq( RM, yE, v_seq, disp = True, graph = True):
	"""
	Validation is performed using vseq indexed values.
	"""
	org_seq = list(range( len( yE)))
	t_seq = [x for x in org_seq if x not in v_seq]

	RMt, yEt = RM[ t_seq, :], yE[ t_seq, 0]
	RMv, yEv = RM[ v_seq, :], yE[ v_seq, 0]

	clf = linear_model.LinearRegression()	
	clf.fit( RMt, yEt)

	print('Weight value')
	#print clf.coef_.flatten()
	plt.plot( clf.coef_.flatten())
	plt.grid()
	plt.xlabel('Tap')
	plt.ylabel('Weight')
	plt.title('Linear Regression Weights')
	plt.show()

	if disp: print('Training result')
	mlr_show( clf, RMt, yEt, disp = disp, graph = graph)

	if disp: print('Validation result')
	r_sqr, RMSE = mlr_show( clf, RMv, yEv, disp = disp, graph = graph)

	#if r_sqr < 0:
	#	print 'v_seq:', v_seq, '--> r_sqr = ', r_sqr

	return r_sqr, RMSE

def mlr_val_vseq_rand(RM, yE, disp = True, graph = True, rate = 5):
	"""
	Validation is peformed using vseq indexed values.
	vseq is randmly selected with respect to rate. 
	"""
	vseq = choose( len( yE), int(len( yE) / rate));

	r_sqr, RMSE = mlr_val_vseq( RM, yE, vseq, disp = disp, graph = graph)

	return r_sqr, RMSE

def mlr_val_vseq_ridge_rand( RM, yE, alpha = .5, rate = 2, disp = True, graph = True):
	vseq = choose( len( yE), int(len( yE) / rate));

	r_sqr, RMSE = mlr_val_vseq_ridge( RM, yE, vseq, alpha = alpha, disp = disp, graph = graph)

	return r_sqr, RMSE

def mlr_val_vseq_lasso_rand( RM, yE, alpha = .5, rate = 2, disp = True, graph = True):
	vseq = choose( len( yE), int(len( yE) / rate));

	r_sqr, RMSE = mlr_val_vseq_lasso( RM, yE, vseq, alpha = alpha, disp = disp, graph = graph)

	return r_sqr, RMSE

def mlr_val_vseq_MMSE_rand( RM, yE, alpha = .5, rate = 2, disp = True, graph = True):
	vseq = choose( len( yE), int(len( yE) / rate));

	r_sqr, RMSE = mlr_val_vseq_MMSE( RM, yE, vseq, alpha = alpha, disp = disp, graph = graph)

	return r_sqr, RMSE

def mlr_val_vseq_ridge_rand_profile( RM, yE, alpha = .5, rate = 2, iterN = 10, disp = True, graph = False, hist = True):

	r2_rms_list = []
	for ii in range( iterN):
		vseq = choose( len( yE), int(len( yE) / rate));
		r_sqr, RMSE = mlr_val_vseq_ridge( RM, yE, vseq, alpha = alpha, disp = disp, graph = graph)
		r2_rms_list.append( (r_sqr, RMSE))

	r2_list, rms_list = list(zip( *r2_rms_list))

	#Showing r2 as histogram
	pd_r2 = pd.DataFrame( {'r_sqr': r2_list})
	pd_r2.plot( kind = 'hist', alpha = 0.5)

	#Showing rms as histogram
	pd_rms = pd.DataFrame( {'rms': rms_list})
	pd_rms.plot( kind = 'hist', alpha = 0.5)

	print("r2: mean = {0}, std = {1}".format( np.mean( r2_list), np.std( r2_list)))
	print("RMSE: mean = {0}, std = {1}".format( np.mean( rms_list), np.std( rms_list)))

	return r2_list, rms_list

def mlr_val_vseq_lasso_rand_profile( RM, yE, alpha = .001, rate = 2, iterN = 10, disp = True, graph = False, hist = True):

	r2_rms_list = []
	for ii in range( iterN):
		vseq = choose( len( yE), int(len( yE) / rate));
		r_sqr, RMSE = mlr_val_vseq_lasso( RM, yE, vseq, alpha = alpha, disp = disp, graph = graph)
		r2_rms_list.append( (r_sqr, RMSE))

	r2_list, rms_list = list(zip( *r2_rms_list))

	#Showing r2 as histogram
	pd_r2 = pd.DataFrame( {'r_sqr': r2_list})
	pd_r2.plot( kind = 'hist', alpha = 0.5)

	#Showing rms as histogram
	pd_rms = pd.DataFrame( {'rms': rms_list})
	pd_rms.plot( kind = 'hist', alpha = 0.5)

	print("r2: mean = {0}, std = {1}".format( np.mean( r2_list), np.std( r2_list)))
	print("RMSE: mean = {0}, std = {1}".format( np.mean( rms_list), np.std( rms_list)))

	return r2_list, rms_list

def mlr_val_vseq_ridge( RM, yE, v_seq, alpha = .5, disp = True, graph = True):
	"""
	Validation is peformed using vseq indexed values.
	"""
	org_seq = list(range( len( yE)))
	t_seq = [x for x in org_seq if x not in v_seq]

	RMt, yEt = RM[ t_seq, :], yE[ t_seq, 0]
	RMv, yEv = RM[ v_seq, :], yE[ v_seq, 0]

	clf = linear_model.Ridge( alpha = alpha)
	clf.fit( RMt, yEt)

	if disp: print('Training result')
	mlr_show( clf, RMt, yEt, disp = disp, graph = graph)

	if disp: print('Validation result')
	r_sqr, RMSE = mlr_show( clf, RMv, yEv, disp = disp, graph = graph)

	#if r_sqr < 0:
	#	print 'v_seq:', v_seq, '--> r_sqr = ', r_sqr

	return r_sqr, RMSE

def mlr_val_vseq_lasso( RM, yE, v_seq, alpha = .5, disp = True, graph = True):
	"""
	Validation is peformed using vseq indexed values.
	"""
	org_seq = list(range( len( yE)))
	t_seq = [x for x in org_seq if x not in v_seq]

	RMt, yEt = RM[ t_seq, :], yE[ t_seq, 0]
	RMv, yEv = RM[ v_seq, :], yE[ v_seq, 0]

	clf = linear_model.Lasso( alpha = alpha)
	clf.fit( RMt, yEt)

	if disp: print('Training result')
	mlr_show( clf, RMt, yEt, disp = disp, graph = graph)

	if disp: print('Validation result')
	r_sqr, RMSE = mlr_show( clf, RMv, yEv, disp = disp, graph = graph)

	#if r_sqr < 0:
	#	print 'v_seq:', v_seq, '--> r_sqr = ', r_sqr

	return r_sqr, RMSE


def mlr_val_vseq_MMSE( RM, yE, v_seq, alpha = .5, disp = True, graph = True):
	"""
	Validation is peformed using vseq indexed values.
	"""
	org_seq = list(range( len( yE)))
	t_seq = [x for x in org_seq if x not in v_seq]

	RMt, yEt = RM[ t_seq, :], yE[ t_seq, 0]
	RMv, yEv = RM[ v_seq, :], yE[ v_seq, 0]

	w, RMt_1 = mmse_with_bias( RMt, yEt)
	yEt_c = RMt_1*w

	print('Weight values')
	#print clf.coef_.flatten()
	plt.plot( w.A1)
	plt.grid()
	plt.xlabel('Tap')
	plt.ylabel('Weight')
	plt.title('Linear Regression Weights')
	plt.show()

	RMv_1 = add_bias_xM( RMv)
	yEv_c = RMv_1*w

	if disp: print('Training result')
	regress_show( yEt, yEt_c, disp = disp, graph = graph)

	if disp: print('Validation result')
	r_sqr, RMSE = regress_show( yEv, yEv_c, disp = disp, graph = graph)

	#if r_sqr < 0:
	#	print 'v_seq:', v_seq, '--> r_sqr = ', r_sqr

	return r_sqr, RMSE	

def ann_val_pre( RM, yE, rate = 2, more_train = True, center = None):
	"""
	In ann case, pre and post processing are used
	while in mlr case, all processing is completed by one function (mlr).
	ann processing will be performed by shell command

	Now, any percentage of validation will be possible. 
	Later, random selection will be included, while currently 
	deterministic selection is applied. 
	"""
	RMt, yEt, RMv, yEv = kchem.get_valid_mode_data( RM, yE, rate = rate, more_train = more_train, center = center)
	kchem.gen_input_files_valid( RMt, yEt, RM)

def _ann_val_post_r0( yE, disp = True, graph = True):
	"""
	After ann_pre and shell command, ann_post can be used.
	"""
	df_ann = pd.read_csv( 'ann_out.csv')
	yv_ann = np.mat( df_ann['out'].tolist()).T
	
	print('Trainig result')
	ann_show( yE[::2,0], yv_ann[::2,0], disp = disp, graph = graph)

	print('Validation result')
	r_sqr, RMSE = ann_show( yE[1::2,0], yv_ann[1::2,0], disp = disp, graph = graph)

	return r_sqr, RMSE

def ann_val_post( yE, disp = True, graph = True, rate = 2, more_train = True, center = None):
	"""
	After ann_pre and shell command, ann_post can be used.
	"""
	df_ann = pd.read_csv( 'ann_out.csv')
	yE_c = np.mat( df_ann['out'].tolist()).T

	yEt, yEt_c, yEv, yEv_c = kchem.get_valid_mode_data( yE, yE_c, rate = rate, more_train = more_train, center = center)
	
	print('Trainig result')
	ann_show( yEt, yEt_c, disp = disp, graph = graph)

	print('Validation result')
	r_sqr, RMSE = ann_show( yEv, yEv_c, disp = disp, graph = graph)

	return r_sqr, RMSE

def writeparam_txt( fname = 'param.txt', dic = {"num_neurons_hidden": 4, "desired_error": 0.00001}):
	"save param.txt with dictionary"
	
	with open(fname, 'w') as f:
		print("Saving", fname)
		for di in dic:
			f.write("{} {}\n".format( di, dic[di]))

def choose(N, n):
	"""
	Returns n randomly chosen values between 0 to N-1.
	"""
	x = list(range( N))
	n_list = []

	for ii in range( n):
		xi = random.choice( x)
		n_list.append( xi)
		x.remove( xi)

	return n_list

def pd_remove_duplist_ID( pdr, dup_l):
	pdw = pdr.copy()
	for d in dup_l:
		for x in d[1:]:
			print(x, pdw.ID[ x], pdw.Smile[ x]) 
			pdw = pdw[ pdw.ID != pdr.ID[ x]]

	return pdw

def pd_remove_faillist_ID( pdr, fail_l):
	pdw = pdr.copy()
	for x in fail_l:
		pdw = pdw[ pdw.ID != pdr.ID[ x]]

	return pdw

def mmse( xM_1, yV):
	Rxx = xM_1.T * xM_1
	Rxy = xM_1.T * yV
	w = np.linalg.pinv( Rxx) * Rxy

	return w

def add_bias_xM( xM):
	xMT_list = xM.T.tolist()
	xMT_list.append( np.ones( xM.shape[0], dtype = int).tolist())
	xM_1 = np.mat( xMT_list).T

	return xM_1	

def mmse_with_bias( xM, yV):
	xM_1 = add_bias_xM( xM)
	w_1 = mmse( xM_1, yV)

	return w_1, xM_1
	
def svm_SVR_C( xM, yV, c_l, graph = True):
	"""
	SVR is performed iteratively with different C values
	until all C in the list are used.
	"""

	r2_l, sd_l = [], []
	for C in c_l:
		print('sklearn.svm.SVR(C={})'.format( C))
		clf = svm.SVR( C = C)
		clf.fit( xM, yV.A1)
		yV_pred = clf.predict(xM)		
		
		r2, sd = regress_show( yV, np.mat( yV_pred).T, graph = graph)
		for X, x in [[r2_l, r2], [sd_l, sd]]:
			X.append( x)

	print('average r2, sd are', np.mean( r2_l), np.mean( sd_l))


	if graph:
		pdw = pd.DataFrame( { 'log10(C)': np.log10(c_l), 'r2': r2_l, 'sd': sd_l})
		pdw.plot( x = 'log10(C)')

	return r2_l, sd_l

def corr_xy( x_vec, y_vec):

	print(type( x_vec), type( y_vec))
	if type( x_vec) != np.matrixlib.defmatrix.matrix:
		molw_x = np.mat( x_vec).T
	else:
		molw_x = x_vec

	if type( y_vec) != np.matrixlib.defmatrix.matrix:
		yV = np.mat( y_vec).T
	else:
		yV = y_vec

	print(molw_x.shape, yV.shape)

	normal_molw_x = molw_x / np.linalg.norm( molw_x)
	yV0 = yV - np.mean( yV)
	normal_yV0 = yV0 / np.linalg.norm( yV0)

	return normal_molw_x.T * normal_yV0

def gs_Lasso( xM, yV, alphas_log = (-1, 1, 9)):

	print(xM.shape, yV.shape)

	clf = linear_model.Lasso()
	#parmas = {'alpha': np.logspace(1, -1, 9)}
	parmas = {'alpha': np.logspace( *alphas_log)}
	kf5_c = model_selection.KFold( n_folds=5, shuffle=True)
	kf5 = kf5_c.split( xM)

	gs = model_selection.GridSearchCV( clf, parmas, scoring = 'r2', cv = kf5, n_jobs = 1)

	gs.fit( xM, yV)

	return gs

def gs_Lasso_norm( xM, yV, alphas_log = (-1, 1, 9)):

	print(xM.shape, yV.shape)

	clf = linear_model.Lasso( normalize = True)
	#parmas = {'alpha': np.logspace(1, -1, 9)}
	parmas = {'alpha': np.logspace( *alphas_log)}
	kf5_c = model_selection.KFold( n_folds=5, shuffle=True)
	kf5 = kf5_c.split( xM)

	gs = model_selection.GridSearchCV( clf, parmas, scoring = 'r2', cv = kf5, n_jobs = -1)

	gs.fit( xM, yV)

	return gs

def gs_Lasso_kf( xM, yV, alphas_log_l):

	kf5_ext_c = model_selection.KFold( n_folds=5, shuffle=True)	
	kf5_ext = kf5_ext_c.split( xM)

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

	kf5_ext_c = model_selection.KFold( n_folds=5, shuffle=True)
	kf5_ext = kf5_ext_c.split( xM)

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

def gs_Ridge( xM, yV, alphas_log = (1, -1, 9)):

	print(xM.shape, yV.shape)

	clf = linear_model.Ridge()
	#parmas = {'alpha': np.logspace(1, -1, 9)}
	parmas = {'alpha': np.logspace( *alphas_log)}
	kf5_c = model_selection.KFold( n_folds=5, shuffle=True)
	kf5 = kf5_c.split( xM)

	gs = model_selection.GridSearchCV( clf, parmas, scoring = 'r2', cv = kf5, n_jobs = 1)

	gs.fit( xM, yV)

	return gs

def gs_Ridge( xM, yV, alphas_log = (1, -1, 9), n_folds = 5):

	print(xM.shape, yV.shape)

	clf = linear_model.Ridge()
	#parmas = {'alpha': np.logspace(1, -1, 9)}
	parmas = {'alpha': np.logspace( *alphas_log)}
	kf_n_c = model_selection.KFold( n_folds=n_folds, shuffle=True)
	kf_n = kf5_c.split( xM)

	gs = model_selection.GridSearchCV( clf, parmas, scoring = 'r2', cv = kf_n, n_jobs = 1)

	gs.fit( xM, yV)

	return gs

def _cv_LinearRegression_r0( xM, yV):

	print(xM.shape, yV.shape)

	clf = linear_model.Ridge()
	kf5_c = model_selection.KFold( n_folds=5, shuffle=True)
	kf5 = kf5_c.split( xM)

	cv_scores = model_selection.cross_val_score( clf, xM, yV, scoring = 'r2', cv = kf5, n_jobs = -1)

	return cv_scores

def _cv_LinearRegression_r1( xM, yV):

	print(xM.shape, yV.shape)

	clf = linear_model.LinearRegression()
	kf5_c = model_selection.KFold( n_folds=5, shuffle=True)
	kf5 = kf5_c.split( xM)
	cv_scores = model_selection.cross_val_score( clf, xM, yV, scoring = 'r2', cv = kf5, n_jobs = -1)

	print('R^2 mean, std -->', np.mean( cv_scores), np.std( cv_scores))

	return cv_scores

def cv_LinearRegression( xM, yV, n_jobs = -1):

	print(xM.shape, yV.shape)

	clf = linear_model.LinearRegression()
	kf5_c = model_selection.KFold( n_folds=5, shuffle=True)
	kf5 = kf5_c.split( xM)
	cv_scores = model_selection.cross_val_score( clf, xM, yV, scoring = 'r2', cv = kf5, n_jobs = n_jobs)

	print('R^2 mean, std -->', np.mean( cv_scores), np.std( cv_scores))

	return cv_scores

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

def gs_RidgeByLasso_kf_ext( xM, yV, alphas_log_l):

	kf5_ext_c = model_selection.KFold( n_folds=5, shuffle=True)	
	kf5_ext = kf5_ext_c.split( xM)

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

def gs_SVR( xM, yV, svr_params):

	print(xM.shape, yV.shape)

	clf = svm.SVR()
	#parmas = {'alpha': np.logspace(1, -1, 9)}
	kf5_c = model_selection.KFold( n_folds=5, shuffle=True)
	kf5 = kf5_c.split( xM)
	gs = model_selection.GridSearchCV( clf, svr_params, scoring = 'r2', cv = kf5, n_jobs = -1)

	gs.fit( xM, yV.A1)

	return gs

def gs_SVRByLasso_kf_ext( xM, yV, alphas_log, svr_params):

	kf5_ext_c = model_selection.KFold( n_folds=5, shuffle=True)	
	kf5_ext = kf5_ext_c.split( xM)

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

	kf5_ext_c = model_selection.KFold( n_folds=5, shuffle=True)	
	kf5_ext = kf5_ext_c.split( xM)

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
	kf5_c = model_selection.KFold( n_folds=5, shuffle=True)
	kf5 = kf5_c.split( xM)
	gs = model_selection.GridSearchCV( clf, en_params, scoring = 'r2', cv = kf5, n_jobs = -1)

	gs.fit( xM, yV)

	return gs

def gs_SVRByElasticNet( xM, yV, en_params, svr_params):

	kf5_ext_c = model_selection.KFold( n_folds=5, shuffle=True)	
	kf5_ext = kf5_ext_c.split( xM)

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

	kf5_ext_c = model_selection.KFold( n_folds=5, shuffle=True)	
	kf5_ext = kf5_ext_c.split( xM)

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

def count( a_l, a, inverse = False):
	"""
	It returns the number of elements which are equal to 
	the target value. 
	In order to resolve when x is an array with more than
	one dimensions, converstion from array to list is used. 
	"""
	if inverse == False:
		x = np.where( np.array( a_l) == a)
	else: 
		x = np.where( np.array( a_l) != a)

	return len(x[0].tolist())

def show_cdf( data, xlabel_str = None, label_str = ''): 
	"""
	Show cdf graph of data which should be list or array in 1-D from.
	xlabel_str is the name of x-axis.
	show() is not included for aggregated plot controlling later. 
	"""
	data_sorted = np.sort( data)

	# calculate the proportional values of samples
	p = 1. * np.arange(len(data)) / (len(data) - 1)

	plt.plot( data_sorted, p, label = label_str)
	if xlabel_str:
		plt.xlabel( xlabel_str)
	plt.ylabel( 'Cumulative Fraction')

def mlr_show4_pred( clf, RMv, yEv, disp = True, graph = True):
	yEv_calc = clf.predict( RMv)

	if len( np.shape(yEv)) == 2 and len( np.shape(yEv_calc)) == 1:
		yEv_calc = np.mat( yEv_calc).T

	r_sqr, RMSE, MAE, DAE = estimate_accuracy4( yEv, yEv_calc, disp = disp)

	if graph:
		plt.figure()
		ms_sz = max(min( 4000 / yEv.shape[0], 8), 1)
		plt.plot( yEv.tolist(), yEv_calc.tolist(), '.', ms = ms_sz)
		ax = plt.gca()
		lims = [
			np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
			np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
		]
		# now plot both limits against eachother
		#ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
		ax.plot(lims, lims, '-', color = 'pink')
		plt.xlabel('Experiment')
		plt.ylabel('Prediction')
		#plt.title( '$r^2$={0:.2e}, RMSE={1:.2e}, AAE={2:.2e}'.format( r_sqr, RMSE, aae))
		plt.title( '$r^2$={0:.1e},$\sigma$={1:.1e},MAE={2:.1e},DAE={3:.1e}'.format( r_sqr, RMSE, MAE, DAE))
		plt.show()

	return (r_sqr, RMSE, MAE, DAE), yEv_calc
    
def mlr4_coef_pred( RM, yE, disp = True, graph = True):
	"""
	Return: coef_, intercept_, yEp
	"""
	clf = linear_model.LinearRegression()
	clf.fit( RM, yE)
	_, yEp = mlr_show4_pred( clf, RM, yE, disp = disp, graph = graph)	

	return clf.coef_, clf.intercept_, yEp