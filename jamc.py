# Python 3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import numexpr as ne

import pandalearn.pdlearn as pdl
import kgrid

def test( fn):
	print( "Testing:", fn.__name__)
	print()

	def test_gen_m():
		for y_ix in [0,1,2,3,4]:
			print( "Modulation mode:", y_ix)
			print( "---------------------")
			m = gen_m( y_ix)
			print( "Direct values:", m)
			print( "Abs(m)", np.abs( m))
			print( "Angle(m)", np.angle( m) / (np.pi * 2 / 360))
			print( "Average power(==1)", np.average( np.power(np.abs(m),2)))
			print()

	if fn.__name__ == "gen_m":
		test_gen_m()
	else:
		raise ValueError("Not supported function for testing", fn.__name__)

def gen_m( y):
	if y == 0: # BPSK
		m = np.array( [1, -1])
	elif y == 1: # QPSK, http://www.rfwireless-world.com/Terminology/QPSK.html
		m = np.array( [1+1j, -1+1j, -1-1j, 1-1j])
	elif y == 2: # 8PSK
		m = np.exp( (1j*2*np.pi)*np.linspace( 0, 1, 8, endpoint = False))
	elif y == 3: # 16QAM
		m_2D = np.zeros( (4,4), dtype = complex)
		m_each = np.array(range( -3, 3 + 1, 2))
		for m_i in range( len( m_each)):
			for m_j in range( len( m_each)):
				m_2D[ m_i, m_j] = m_each[ m_i] + 1j*m_each[ m_j]
		m = m_2D.flatten()
	elif y == 4: # 64QAM
		m_2D = np.zeros( (8,8), dtype = complex)
		m_each = np.array(range( -7, 7 + 1, 2))
		for m_i in range( len( m_each)):
			for m_j in range( len( m_each)):
				m_2D[ m_i, m_j] = m_each[ m_i] + 1j*m_each[ m_j]
		m = m_2D.flatten()
	else:
		raise ValueError( "Modulation mode of y={} is not defined.".format(y))

	m = m / np.std( m)

	return m

def gen_s_with_m( N, y):
	m = gen_m( y)
	ix = np.random.randint( 0, m.shape[0], size = N)
	s = m[ ix]
	return s

def gen_r( N = 10, y = 0, SNRdB = 0, f0T = 0):
	"""
	Apr 23, 2016
	------------
	m will be get as argument and gen_s(m, y) will be called. 
	"""
	alpha = np.sqrt( np.power( 10.0, SNRdB / 10.0))
	s_n = gen_s_with_m( N, y)
	g_n = (np.random.randn( N) + 1j*np.random.randn( N)) / np.sqrt(2)
	r_n = np.exp( 1j*f0T*np.arange(N)) * s_n + g_n / alpha
	return r_n	

def gen_r_L( L, N = 10, y = 0, SNRdB = 0, f0T = 0):
	X_1 = np.zeros( (L, N), dtype = complex)
	for l in range(L):
		X_1[l, :] = gen_r( N = N, y = y, SNRdB = SNRdB, f0T = f0T)
	return X_1

def gen_s_with_m( N, y):
	m_y = gen_m( y)
	
	ix = np.random.randint( 0, m_y.shape[0], size = N)
	s = m_y[ ix]
	return s

def getXy( L = 500, y_l = [0, 1], SNRdB = 8, f0T = 0, N = 250):
	"""
	Apr 23, 2016
	------------
	- gen_r_L() will be changed to calculate gen_m() initially. 
	- This part will be moved to pyx codes.
	"""

	X = np.zeros( (L * len( y_l), N), dtype = complex)
	y = np.zeros( L * len( y_l), dtype = int)

	for y_idx in range( len(y_l)):
		st = y_idx * L
		ed = (y_idx + 1) * L
		y[ st:ed] = np.ones( L) * y_l[ y_idx]
		X[ st:ed, :] = gen_r_L( L = L, N = N, y = y_l[ y_idx], SNRdB = SNRdB, f0T = f0T)

	return X, y

def _getXy_m5_r0( L = 500, y_l = range(5), SNRdB = 8, f0T = 0, N = 250):
	"""
	Apr 23, 2016
	------------
	- gen_r_L() will be changed to calculate gen_m() initially. 
	- This part will be moved to pyx codes.
	"""

	X = np.zeros( (L * len( y_l), N), dtype = complex)
	y = np.zeros( L * len( y_l), dtype = int)

	for y_idx in range( len(y_l)):
		st = y_idx * L
		ed = (y_idx + 1) * L
		y[ st:ed] = np.ones( L) * y_l[ y_idx]
		X[ st:ed, :] = gen_r_L( L = L, N = N, y = y_l[ y_idx], SNRdB = SNRdB, f0T = f0T)

	return X, y

## Signal generation
def gen_m( y):
	if y == 0: # BPSK
		m = np.array( [1, -1])
	elif y == 1: # QPSK, http://www.rfwireless-world.com/Terminology/QPSK.html
		m = np.array( [1+1j, -1+1j, -1-1j, 1-1j])
	elif y == 2: # 8PSK
		m = np.exp( (1j*2*np.pi)*np.linspace( 0, 1, 8, endpoint = False))
	elif y == 3: # 16QAM
		m_2D = np.zeros( (4,4), dtype = complex)
		m_each = np.array(range( -3, 3 + 1, 2))
		for m_i in range( len( m_each)):
			for m_j in range( len( m_each)):
				m_2D[ m_i, m_j] = m_each[ m_i] + 1j*m_each[ m_j]
		m = m_2D.flatten()
	elif y == 4: # 64QAM
		m_2D = np.zeros( (8,8), dtype = complex)
		m_each = np.array(range( -7, 7 + 1, 2))
		for m_i in range( len( m_each)):
			for m_j in range( len( m_each)):
				m_2D[ m_i, m_j] = m_each[ m_i] + 1j*m_each[ m_j]
		m = m_2D.flatten()
	else:
		raise ValueError( "Modulation mode of y={} is not defined.".format(y))

	m = m / np.std( m)

	return m

def gen_s_fast( N, m_y = None):
	"""
	s is generated without rebuiding m_y	
	"""
	ix = np.random.randint( 0, m_y.shape[0], size = N)
	s = m_y[ ix]
	return s

def gen_r_with_m( m_y, N = 10, SNRdB = 0, f0T = 0):
	"""
	Apr 23, 2016
	------------
	m will be get as argument and gen_s(m, y) will be called. 
	"""
	alpha = np.sqrt( np.power( 10.0, SNRdB / 10.0))
	s_n = gen_s_fast( N, m_y = m_y)
	g_n = (np.random.randn( N) + 1j*np.random.randn( N)) / np.sqrt(2)
	r_n = np.exp( 1j*f0T*np.arange(N)) * s_n + g_n / alpha
	return r_n	

def gen_r_with_m_L( m_y, L, N = 10, SNRdB = 0, f0T = 0):
	X_1 = np.zeros( (L, N), dtype = complex)
	for l in range(L):
		X_1[l, :] = gen_r_with_m( m_y, N = N, SNRdB = SNRdB, f0T = f0T)
	return X_1

def get_Xy( L = 500, y_l = range(5), SNRdB = 8, f0T = 0, N = 250):
	"""
	Apr 23, 2016
	------------
	- gen_r_L() will be changed to calculate gen_m() initially. 
	- This part will be moved to pyx codes.
	"""

	X = np.zeros( (L * len( y_l), N), dtype = complex)
	y = np.zeros( L * len( y_l), dtype = int)

	for y_idx in range( len(y_l)):
		st = y_idx * L
		ed = (y_idx + 1) * L
		y[ st:ed] = np.ones( L) * y_l[ y_idx]
		m_y = gen_m( y_l[ y_idx])
		X[ st:ed, :] = gen_r_with_m_L( m_y, L = L, N = N, SNRdB = SNRdB, f0T = f0T)

	return X, y	

getXy_m5 = get_Xy 

def _gen_r_1( N = 10, y = 0, SNRdB = 0, f0T = 0):
	alpha = np.sqrt( np.power( 10.0, SNRdB / 10.0))
	s_n = gen_s( N, y)
	g_n = (np.random.randn( N) + 1j*np.random.randn( N)) / np.sqrt(2)
	r_n = np.exp( 1j*f0T*np.arange(N)) * s_n + g_n / alpha
	return r_n	

def _gen_r_r0( N = 10, y = 0, SNRdB = 0, f0T = 0):
	alpha = np.sqrt( np.power( 10.0, SNRdB / 10.0))
	s_n = gen_s( N, y)
	g_n = (np.random.randn( N) + 1j*np.random.randn( N)) / np.sqrt(2)
	r_n = alpha * np.exp( 1j*f0T*np.arange(N)) * s_n + g_n
	return r_n

def gen_r_fast( N = 10, y = 0, SNRdB = 0, f0T = 0):
	alpha = np.sqrt( np.power( 10.0, SNRdB / 10.0))
	s_n = gen_s( N, y)

	rd_1 = np.random.randn( N)
	rd_2 = np.random.randn( N)
	sqrt_2 = np.sqrt(2)
	g_n = ne.evaluate( "(rd_1 + 1j*rd_2) / sqrt_2")

	rd_3 = np.arange(N)
	r_n = ne.evaluate( "alpha * exp( 1j*f0T*rd_3) * s_n + g_n")
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

def getxy_amc_dataframe_l( L = 500, y_l = [0, 1], SNRdB = 8, f0T = 0, N = 250):
	"""
	Directly call getxy to get generated channel data

	Return
	--------
	X, y: machine learning input and output data
	"""
	df_l = AMC_DataFrame_l( L = L, y_l = y_l, SNRdB = SNRdB, f0T = f0T, N=N)
	return df_l.getxy()

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