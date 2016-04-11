# Python 3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import pandalearn.pdlearn as pdl
import jgrid

def gen_s( N, y):
	if y == 0: # BPSK
		m = np.array( [1, -1])
		ix = np.random.randint( 0, 2, size = N)
		s = m[ ix]
	elif y == 1: # QPSK, http://www.rfwireless-world.com/Terminology/QPSK.html
		m = np.array( [-1-1j, -1+1j, 1+1j, 1-1j]) / np.sqrt(2)
		ix = np.random.randint( 0, 4, size = N)
		s = m[ ix]        
	else:
		raise ValueError( "Modulation mode of y={} is not defined.".format(y))
	return s

def gen_r( N = 10, y = 0, SNRdB = 0, f0T = 0):
	alpha = np.sqrt( np.power( 10.0, SNRdB / 10.0))
	s_n = gen_s( N, y)
	g_n = (np.random.randn( N) + 1j*np.random.randn( N)) / np.sqrt(2)
	r_n = alpha * np.exp( 1j*f0T*np.arange(N)) * s_n + g_n
	return r_n

def pd_gen_r( y = 0, SNRdB = 8, f0T = 0,  N = 250):
	r_n = gen_r( N = N, y = y, SNRdB = SNRdB, f0T = 0)
	
	pdi = pd.DataFrame()
	pdi["y"] = [y] * N
	pdi["SNRdB"] = [SNRdB] * N
	pdi["f0T"] = [f0T] * N
	pdi["n"] = range( N)
	pdi["r_n"] = r_n
	return pdi

class AMC_DataFrame( pd.DataFrame):
	def __init__( self, y = 0, SNRdB = 8, f0T = 0, N = 250):
		super().__init__()

		r_n = gen_r( N = N, y = y, SNRdB = SNRdB, f0T = f0T)

		self["y"] = [y] * N
		self["SNRdB"] = [SNRdB] * N
		self["f0T"] = [f0T] * N
		self["n"] = range( N)
		self["r_n"] = r_n

	def re_im_abs( self, inplace = False):
		if inplace:
			pdr = self
		else:
			pdr = self.copy()

		r_n = pdr["r_n"].values
		pdr["Re(r_n)"] = np.real( r_n)
		pdr["Im(r_n)"] = np.real( r_n)
		pdr["abs(r_n)"] = np.abs( r_n)

		if inplace is False:
			return pdr

class AMC_DataFrame_l( list):
	def __init__( self, L = 500, y_l = [0, 1], SNRdB = 8, f0T = 0, N = 250):		
		"""
		L: number of realizations 
		"""
		super().__init__()
		
		for l in range( L):
			self.extend( map( lambda y: AMC_DataFrame( y = y, SNRdB = SNRdB, f0T = f0T, N = N), y_l))
		
		if L > 0:
			self.shape_X = (L * len( y_l), N)
			self.dtype_X = type( self[0].r_n[0])
			# print( self.dtype_X)
			self.shape_y = (L * len( y_l), )
			self.dtype_y = type( self[0].y[0])
			# print( self.dtype_y)
			
	def getxy( self, return_flag = True):
		"""
		Collect X, y from DataFrame_l
		return them and save them as member variables.
		"""
		self.X = np.zeros( self.shape_X, dtype = self.dtype_X)
		self.y = np.zeros( self.shape_y, dtype = self.dtype_y)

		for ly in range( self.y.shape[0]):
			self.X[ly, :] = self[ ly].r_n.values  
			self.y[ly] = self[ ly].y.values[0] # y must be the same in a DataFrame

		if return_flag:
			return self.X, self.y

def AMC_SVR( f0T, graph = False):
	"""
	Return
	------
	p_g_mean : pandas.DataFrame
		mean of score
	p_g_std : pandas.DataFrame
		std of score
	"""
	df_l = AMC_DataFrame_l( f0T = f0T)
	X, y = df_l.getxy()
	X_re = np.real( X)

	params = {'C': np.logspace(0, 2, 3),  'gamma': np.logspace(-4,-2,6)}
	gs = jgrid.gs_SVC( X_re, y, params)
	p_df = pdl.grid_scores( gs.grid_scores_)

	if graph:
		sns.tsplot( p_df, time = "gamma", unit = "Unit", value = "Score", condition = "C")
		plt.xscale('log')
		plt.ylabel( r"Score: $1 - P_e$")

	p_g = p_df.groupby(['C', 'gamma'])
	p_g_mean = p_g.mean()[['Score']]
	p_g_std = p_g.std()[['Score']]

	return p_g_mean, p_g_std

def AMC_SVR_l( f0T_l, disp = True, graph = False):
	"""
	Simulate the performance of AMC-SVR with respect to different frequency error
	"""
	p_l = list()
	for f0T in f0T_l: #np.logspace( -4,-1,12):
		pg_mean, pg_std = AMC_SVR( f0T, graph = False)
		pi = pd.DataFrame()
		pi["f0T"] = [f0T]
		ix = pg_mean["Score"].values.argmax()
		pi["E[Score]"] = [pg_mean["Score"].values[ ix]]
		pi["std(Score)"] = [pg_std["Score"].values[ix]]

		if disp:
			print("----------")
			print( "foT: {0} --> Mean and s.t.d of scores: {1}, {2}".format( f0T, 
				pg_mean["Score"].values[ ix], pg_std["Score"].values[ix]))
			# pg_mean["Score"].values[ ix] < values should be added to be used like np.array()
			print( "E[Score] :", pg_mean["Score"].values)
			print( "Argmax E[Score] :", ix)
			print( "Max E[Score]:", pg_mean["Score"].values[ ix])
	  
		p_l.append( pi)

	p_df = pd.concat( p_l, ignore_index=True).set_index( 'f0T')
	if graph:
		p_df.plot( kind = 'line', y = "E[Score]" , yerr = "std(Score)", logx = True, legend = False )
		plt.ylabel("E[Score]: E[$1-P_e$]")
	return p_df