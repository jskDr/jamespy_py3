# Python 3
# pandalearn pdl

import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
import numpy as np

def estimate_accuracy(yEv, yEv_calc, disp = False):
	"""
	It was originally located in jchem. However now it is allocated here
	since the functionality is more inline with jutil than jchem. 
	"""

	r_sqr = metrics.r2_score( yEv, yEv_calc)
	RMSE = np.sqrt( metrics.mean_squared_error( yEv, yEv_calc))
	MAE = metrics.mean_absolute_error( yEv, yEv_calc)
	DAE = metrics.median_absolute_error( yEv, yEv_calc)

	if disp:
		print("r^2={0:.2e}, RMSE={1:.2e}, MAE={2:.2e}, DAE={3:.2e}".format( r_sqr, RMSE, MAE, DAE))

	return r_sqr, RMSE, MAE, DAE

def mlr_show( clf, RMv, yEv, disp = True, graph = True):
	yEv_calc = clf.predict( RMv)

	if len( np.shape(yEv)) == 2 and len( np.shape(yEv_calc)) == 1:
		yEv_calc = np.mat( yEv_calc).T

	r_sqr, RMSE, MAE, DAE = estimate_accuracy( yEv, yEv_calc, disp = disp)

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

def mlr( x, y, df, disp = True, graph = True):
	xM = np.mat( df[x]).T
	yV = np.mat( df[y]).T

	clf = linear_model.LinearRegression()
	clf.fit( xM, yV)
	mlr_show( clf, xM, yV, disp = disp, graph = graph)	

	return clf

