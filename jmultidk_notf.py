# Python 3
"""
Codes for MultiDK are included.

Author: (James) Sung-Jin Kim
Date: April 2, 2016 ~ Now

Editorial notes
* April2, 2016
  I will use pd.DataFrame as base class to make new class for R2_DF, 
  which is specialized for r^2 scores and support MultiDK. 
  Hence, it can be copied to jpandas later on. 

* Tensorflow related routine is rmoved. 
"""

from time import time
import pandas as pd
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# April 24, 2016
from sklearn import cross_validation, metrics
#import tensorflow.contrib.learn as skflow

import jpandas as jpd
import jchem, jgrid
import j3x.jpyx
import jseaborn as jsns
import kutil

def list_agg_n( n_folds):
	"""
	Generate a function to aggregate multiple lists
	for functools.reduce()
	"""
	f = lambda s, x: s + [x] * n_folds
	return f

class R2_Scores( object):
	def __init__(self, fname = None, Nm = 10, n_alphas = 10, n_folds = 20):
		"""
		Make a class for r^2 scores based on pd.DataFrame
		[Input]
		fname: file name for r^2 dataframe
		Nm: the number of methods

		E.g. 
		fname = "sheet/wang3705_MDMK2to23_{}methods.csv".format(Nm)
		"""
		
		if fname is not None:
			self.fname = fname
			self.df = pd.read_csv( fname)
		else:
			self.df = pd.DataFrame()
		self.Nm = Nm 
		self.n_alphas = n_alphas
		self.n_folds = n_folds		

	def updata_for_multidk(self, fname_out = None):
		"""
		1. alpha_ID is added on the DataFrame since float point values such as alpha
		can not be found by value. An index of the best alpha value for each method
		will be stored as well as the best alpha so that the index will be used 
		to filter the best values of mean(r2) and std(r2).
		"""
		n_alphas = self.n_alphas
		n_folds = self.n_folds

		# Step 1: adding alpha_ID		
		self.df["alpha_ID"] = reduce( list_agg_n( n_folds), range( n_alphas), []) * self.Nm

		# Step 2: change names MDMKx to MultiDKx
		rn_d = {'MDMK2to11':"MultiDK2-11", 'MDMK2to21': "MultiDK2-21", 
				 'MDMK2to23': "MultiDK2-23", 'MDMK2to13': "MultiDK2-13", "MDMK1to10":"MultiDK1-10",
				 "MDMK": "MultiDK"} # MDMK is included for legacy cases such as redox potential prediction
		df_method_l = self.df["Method"].tolist()
		rn_l = []
		for m in df_method_l:
			if m in rn_d.keys():
				rn_l.append( rn_d[m])
			else:
				rn_l.append( m)
		self.df["Method"] = rn_l

		if fname_out is not None:
			self.df.to_csv( fname_out, index = False)
		elif self.fname is not None:
			fname_out = self.fname[:-4] + '_refine.csv'
			print( "Default: self.df is saved to", fname_out)
			self.df.to_csv( fname_out, index = False)

		return self.df

	def mean_std(self, fname_out = None):
		df_g = self.df.groupby(["Method", "alpha"])
		self.df_gr = df_g.agg({"r2":[np.mean, np.std]})["r2"]
		# Index should be stored. 
		if fname_out is not None:
			"""
			index should be saved so 'index = True' as default
			"""
			self.df_gr.to_csv( fname_out)
		elif self.fname is not None:
			fname_out = self.fname[:-4] + '_mean_std.csv'
			print( "Default: self.df_gr is saved to", fname_out)
			self.df_gr.to_csv( fname_out)

		return self.df_gr

	def max_mean_r2( self, fname_out = None):
		"""
		Extact all method names and get a set with only unique name
		"""
		self.method_l = set(self.df_gr.index.get_level_values(0))
		pdi_l = list()
		for m in self.method_l:
			p_m = self.df_gr.loc[ m]
			alpha_l = p_m.index.tolist()
			m_r2 = p_m["mean"].values
			std_r2 = p_m["std"].values
			i_max = m_r2.argmax()
			pdi = pd.DataFrame( [[m, i_max, alpha_l[i_max], m_r2[i_max], std_r2[i_max]]], 
									columns=["Method", "best_alpha_ID", "best_alpha", "E[r2]", "std(r2)"])
			pdi_l.append( pdi)
		
		pdo_best = pd.concat( pdi_l, ignore_index=True).sort_values("Method")
		self.df_best = pdo_best.set_index("Method")
		
		if fname_out is not None:
			self.df_best.to_csv( fname_out) # index should be stored.
		elif self.fname is not None:
			fname_out = self.fname[:-4] + '_best4bar.csv'
			print( 'Default: self.df_best is saved to', fname_out)
			self.df_best.to_csv( fname_out) # index should be stored.

		return self.df_best

	def get_box_data( self, fname_out = None):
		"""
		DataFrame is arranged for box plot. 
		"""
		pdo = self.df

		cond = None
		for m in self.method_l:
			best_alpha_ID = self.df_best.loc[ m]["best_alpha_ID"]
			if cond is None:
				cond = (pdo.Method == m) & (pdo.alpha_ID == best_alpha_ID)
			else:
				cond |= (pdo.Method == m) & (pdo.alpha_ID == best_alpha_ID)
		self.df_best_expand = self.df[ cond].reset_index( drop = True)

		if fname_out is not None:
			self.df_best_expand.to_csv( fname_out) # index should be stored.
		elif self.fname is not None:
			fname_out = self.fname[:-4] + '_best4box.csv'
			print( 'Default: self.df_best_expand is saved to', fname_out)
			self.df_best_expand.to_csv( fname_out) # index should be stored.

		return self.df_best_expand

	def run( self):
		self.updata_for_multidk()	
		self.mean_std()
		self.max_mean_r2()
		self.get_box_data()

		return self.df_best, self.df_best_expand

	def plot_bar( self, fname_out = None):
		self.df_best.plot( kind = 'bar', y = "E[r2]", yerr="std(r2)", legend=False)	

		if fname_out is not None:
			plt.savefig( fname_out) # index should be stored.
		elif self.fname is not None:
			fname_out = self.fname[:-4] + '_bar.eps'
			print( 'Default: the figure of self.df_best is saved to', fname_out)
			plt.savefig( fname_out) 

	def plot_box( self, fname_out = None):		
		sns.boxplot(x="Method", y="r2", data=self.df_best_expand, palette="PRGn")
		sns.despine(offset=10, trim=True)
		plt.ylabel( r"$r^2$")
		plt.xlabel( "Methods")

		if fname_out is not None:
			plt.savefig( fname_out) # index should be stored.
		elif self.fname is not None:
			fname_out = self.fname[:-4] + '_box.eps'
			print( 'Default: the figure of self.df_best_expand is saved to', fname_out)
			plt.savefig( fname_out) 

	def run_with_plot( self):
		self.run()
		self.plot_bar()
		plt.show()
		self.plot_box()
		plt.show()

def set_X_23( s_l, xM_logP):
	# s_l = self.s_l
	# xM_logP = self.xM_logP

	# Body 
	xM_d = dict()
	xM_d["MFP"] = jchem.get_xM( s_l, radius=4, nBits=2048)
	xM_d["MACCS"] = jchem.get_xM_MACCSkeys( s_l)
	xM_d["MolW"] = jchem.get_xM_molw( s_l)
	xM_d["LASA"] = jchem.get_xM_lasa( s_l)
	xM_d["logP"] = xM_logP

	for d_s in ["MolW", "LASA", "logP"]:
		# x_a = xM_d[ d_s]
		# x_a = np.divide( x_a, np.std( x_a, axis = 0)) # Normalize
		xM_d[ d_s] = np.divide( xM_d[ d_s], np.std( xM_d[ d_s], axis = 0)) 

	xM_2 = np.concatenate( [xM_d["MFP"], xM_d["MACCS"]], axis = 1)
	xM_p = np.concatenate( [xM_d[ d_s] for d_s in ["MolW", "LASA", "logP"]], axis = 1) # Concatenation of associated properties
	print( xM_2.shape, xM_p.shape)

	# Output processing
	#self.xM_d = xM_d
	#self.xM_2 = xM_2 
	#self.xM_p = xM_p

	return xM_d, xM_2, xM_p


def set_X_23_M2( s_l, xM_logP):
	# s_l = self.s_l
	# xM_logP = self.xM_logP

	# Body 
	xM_d = dict()
	xM_d["MFP"] = jchem.get_xM( s_l, radius=4, nBits=2048)
	xM_d["MACCS"] = jchem.get_xM_MACCSkeys( s_l)
	xM_d["MolW"] = jchem.get_xM_molw( s_l)
	xM_d["MolW2"] = np.power( jchem.get_xM_molw( s_l), 2)
	xM_d["LASA"] = jchem.get_xM_lasa( s_l)
	xM_d["LASA2"] = jchem.get_xM_lasa( s_l)
	xM_d["logP"] = xM_logP

	for d_s in ["MolW", "MolW2", "LASA", "LASA2", "logP"]:
		xM_d[ d_s] = np.divide( xM_d[ d_s], np.std( xM_d[ d_s], axis = 0)) 

	xM_2 = np.concatenate( [xM_d["MFP"], xM_d["MACCS"]], axis = 1)
	xM_p = np.concatenate( [xM_d[ d_s] for d_s in ["MolW", "LASA", "logP"]], axis = 1) # Concatenation of associated properties
	print( xM_2.shape, xM_p.shape)

	# Output processing
	#self.xM_d = xM_d
	#self.xM_2 = xM_2 
	#self.xM_p = xM_p

	return xM_d, xM_2, xM_p

def set_A_2( xM_d, xM_2):
	# Input processing
	#xM_d = self.xM_d
	#xM_2 = self.xM_2

	# Body
	A_d = dict()
	for key in ["MFP", "MACCS"]:
		print( key)
		A_d[ key] = j3x.jpyx.calc_tm_sim_M( xM_d[key])
	A_2 = j3x.jpyx.calc_tm_sim_M( xM_2)

	# Output processing
	#self.A_d = A_d
	#self.A_2 = A_2

	return A_d, A_2

def set_alpha_log( a_st = -2, a_ed = 2, a_n = 2):
	"""
	Generate alpha_log with a range from a_st to s_ed
	with a_n for each unit.
	"""

	a_N = (a_ed - a_st)*a_n + 1
	return (a_st, a_ed, a_N)

class MultiDK():
	def __init__(self, fname = 'sheet/wang3705_with_logP.csv'):
		self.fname_core = fname[:-14] 
		self.pdr = pd.read_csv( fname)
		self.alphas_log = set_alpha_log( -2, 2, 2)
		
	def set_xy(self):
		"""
		if self is changed self will be a return value
		for feedback all outputs using only a variable
		"""
		pdr = self.pdr

		self.s_l = self.pdr.SMILES.tolist()
		self.xM_logP = np.mat(self.pdr.logP.values).T
		self.yV = jpd.pd_get_yV( self.pdr, y_id="exp")

		return self

	def set_X( self):
		# Input processing
		s_l = self.s_l
		xM_logP = self.xM_logP

		# BOdy
		xM_d, xM_2, xM_p = set_X_23( s_l, xM_logP)

		# Output processing
		self.xM_d = xM_d
		self.xM_2 = xM_2 
		self.xM_p = xM_p

		return self

	def set_X_M2( self):
		# Input processing
		s_l = self.s_l
		xM_logP = self.xM_logP

		# BOdy
		xM_d, xM_2, xM_p = set_X_23_M2( s_l, xM_logP)

		# Output processing
		self.xM_d = xM_d
		self.xM_2 = xM_2 
		self.xM_p = xM_p

		return self

	def set_A( self):
		# Input processing
		xM_d = self.xM_d
		xM_2 = self.xM_2

		# Body
		A_d, A_2 = set_A_2( xM_d, xM_2)
	
		# Output processing
		self.A_d = A_d
		self.A_2 = A_2

		return self

	def grid_search_sd( self):
		# input processing
		xM_d = self.xM_d
		xM_p = self.xM_p

		yV = self.yV

		A_d = self.A_d
		A_2 = self.A_2 

		#Body
		t = time() 
		pdi_d = dict()

		pdi_d["SD"] = jsns.pdi_gs_full( "SD", [xM_d["MFP"]], yV, expension = True, n_jobs = 1)
		print('Elasped time is', time() - t, 'sec')

		pdi_d["MD21"] = jsns.pdi_gs_full( "MD21", [xM_d[ d_s] for d_s in ["MFP", "MACCS", "MolW"]], yV, 
										 expension = True)
		print('Elasped time is', time() - t, 'sec')

		pdi_d["MD23"] = jsns.pdi_gs_full( "MD23", list(xM_d.values()), yV, expension = True)
		print('Elasped time is', time() - t, 'sec')

		pdi_d["MDMK1to10"] = jsns.pdi_gs_full( "MDMK1to10", [A_d["MFP"]], yV, 
											  mode = "BIKE_Ridge", expension = True)
		print('Elasped time is', time() - t, 'sec')

		pdi_d["MDMK2to11"] = jsns.pdi_gs_full( "MDMK2to11", [A_2], yV, X_concat = xM_d["MolW"], 
											  mode = "BIKE_Ridge", expension = True)
		print('Elasped time is', time() - t, 'sec')

		pdi_d["MDMK2to13"] = jsns.pdi_gs_full( "MDMK2to13", [A_2], yV, X_concat = xM_p, 
											  mode = "BIKE_Ridge", expension = True)
		print('Elasped time is', time() - t, 'sec')

		pdi_d["MDMK2to21"] = jsns.pdi_gs_full( "MDMK2to21", [A_d["MFP"], A_d["MACCS"]], yV, X_concat = xM_d["MolW"], 
											  mode = "BIKE_Ridge", expension = True)
		print('Elasped time is', time() - t, 'sec')

		pdi_d["MDMK2to23"] = jsns.pdi_gs_full( "MDMK2to23", [A_d["MFP"], A_d["MACCS"]], yV, X_concat = xM_p, 
										  mode = "BIKE_Ridge", expension = True, n_jobs = 1)
		print('Elasped time is', time() - t, 'sec')

		pdo = pd.concat( pdi_d.values())
		print( pdo.shape)

		Nm = len(pdi_d)
		#print( "The number of methods now is", Nm)
		fname_out = self.fname_core + "_MDMK2to23_{}methods.csv".format(Nm)
		print("The performance data are save to", fname_out)
		pdo.to_csv( fname_out, index = False)

		self.pdo = pdo

		return self

	def grid_search( self):
		# input processing
		xM_d = self.xM_d
		xM_p = self.xM_p

		yV = self.yV

		A_d = self.A_d
		A_2 = self.A_2 

		#Body
		t = time() 
		pdi_d = dict()

		for k in xM_d:
			s = "SD({})".format( k)
			pdi_d[s] = jsns.pdi_gs_full( s, [xM_d[k]], yV, expension = True, n_jobs = 1)
			print('Elasped time is', time() - t, 'sec')

		# pdi_d["SD(MFP)"] = jsns.pdi_gs_full( "SD(MFP)", [xM_d["MFP"]], yV, expension = True, n_jobs = 1)
		# print('Elasped time is', time() - t, 'sec')

		# pdi_d["SD(MACCS)"] = jsns.pdi_gs_full( "SD(MACCS)", [xM_d["MACCS"]], yV, expension = True, n_jobs = 1)
		# print('Elasped time is', time() - t, 'sec')

		# pdi_d["SD(MolW)"] = jsns.pdi_gs_full( "SD(MolW)", [xM_d["MolW"]], yV, expension = True, n_jobs = 1)
		# print('Elasped time is', time() - t, 'sec')

		# pdi_d["SD(LASA)"] = jsns.pdi_gs_full( "SD(LASA)", [xM_d["MolW"]], yV, expension = True, n_jobs = 1)
		# print('Elasped time is', time() - t, 'sec')

		# pdi_d["SD(logP)"] = jsns.pdi_gs_full( "SD(logP)", [xM_d["logP"]], yV, expension = True, n_jobs = 1)
		# print('Elasped time is', time() - t, 'sec')

		pdi_d["MD21"] = jsns.pdi_gs_full( "MD21", [xM_d[ d_s] for d_s in ["MFP", "MACCS", "MolW"]], yV, 
										 expension = True)
		print('Elasped time is', time() - t, 'sec')

		pdi_d["MD23"] = jsns.pdi_gs_full( "MD23", list(xM_d.values()), yV, expension = True)
		print('Elasped time is', time() - t, 'sec')

		pdi_d["MDMK1to10"] = jsns.pdi_gs_full( "MDMK1to10", [A_d["MFP"]], yV, 
											  mode = "BIKE_Ridge", expension = True)
		print('Elasped time is', time() - t, 'sec')

		pdi_d["MDMK2to11"] = jsns.pdi_gs_full( "MDMK2to11", [A_2], yV, X_concat = xM_d["MolW"], 
											  mode = "BIKE_Ridge", expension = True)
		print('Elasped time is', time() - t, 'sec')

		pdi_d["MDMK2to13"] = jsns.pdi_gs_full( "MDMK2to13", [A_2], yV, X_concat = xM_p, 
											  mode = "BIKE_Ridge", expension = True)
		print('Elasped time is', time() - t, 'sec')

		pdi_d["MDMK2to21"] = jsns.pdi_gs_full( "MDMK2to21", [A_d["MFP"], A_d["MACCS"]], yV, X_concat = xM_d["MolW"], 
											  mode = "BIKE_Ridge", expension = True)
		print('Elasped time is', time() - t, 'sec')

		pdi_d["MDMK2to23"] = jsns.pdi_gs_full( "MDMK2to23", [A_d["MFP"], A_d["MACCS"]], yV, X_concat = xM_p, 
										  mode = "BIKE_Ridge", expension = True, n_jobs = 1)
		print('Elasped time is', time() - t, 'sec')

		pdo = pd.concat( pdi_d.values())
		print( pdo.shape)

		Nm = len(pdi_d)
		#print( "The number of methods now is", Nm)
		fname_out = self.fname_core + "_MDMK2to23_{}methods.csv".format(Nm)
		print("The performance data are save to", fname_out)
		pdo.to_csv( fname_out, index = False)

		self.pdo = pdo

		return self

	def cv_MultiDK23( self, alpha, n_jobs = 1):
		"""
		Return
		--------
		yV_pred: np.array(), mostly 1D
		return prediction results.

		"""
		self.set_xy()
		self.set_X()
		self.set_A() 

		xM_d = self.xM_d
		xM_p = self.xM_p

		yV = self.yV

		# A_d = self.A_d
		A_2 = self.A_2 

		#Body
		t = time() 
		yV_pred = jgrid.cv_BIKE_Ridge( [A_2], yV, alpha = alpha, XX = xM_p, n_folds = 20, n_jobs = n_jobs, grid_std = None)
		print('Elasped time is', time() - t, 'sec')

		return yV_pred


	def plot( self):
		sns.tsplot(data=self.pdo, time="alpha", unit="unit", condition="Method", value="r2")
		plt.xscale('log')
		plt.ylabel( r'$r^2$')

	def _run_r0( self):
		self.set_xy()
		self.set_X()
		self.set_A() 
		self.grid_search()
		self.plot()

	def run( self, SDx = False):
		self.set_xy()
		self.set_X()
		self.set_A() 
		if SDx:
			self.grid_search()
		else:
			self.grid_search_sd()		
		self.plot()

		return self

	def set_XA( self):
		self.set_xy()
		self.set_X()
		self.set_A()

		return self
