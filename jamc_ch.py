# Py3

import numpy as np
import pandas as pd
from j3x import jamc_ch_pyx

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

def get_Xy_CSS( L = 250, y_l = range(5), SNRdB = 8, f0T = 0.0, N = 250):
	"""
	Now, jamc_ch_pyx.get_Xy() supports no noise case (the ideal case) by 
	setting SNRdB as None instead of specifying a certain value.
	"""
	# Nm = 5
	# N = 250
	if SNRdB is not None:
		X_org, y= jamc_ch_pyx.get_Xy( L = L, y_l = y_l, SNRdB = SNRdB, f0T = f0T, N = N) 
	else:
		X_org, y= jamc_ch_pyx.get_Xy_nonoise( L = L, y_l = y_l, f0T = f0T, N = N) 		
	#X_org, y= jamc.getXy( f0T = 0.0, SNRdB = 100, L = 5000) 
	print(X_org.shape, y.shape)

	X_abs = np.abs( X_org)
	X_angle = np.mod( np.angle( X_org) / (2*np.pi), 1.0)

	X_abs.sort( axis = 1)
	X_angle.sort( axis = 1)

	X = np.concatenate( [X_abs, X_angle], axis = 1)
	print( X_abs.shape, X_abs.dtype)
	print( X_angle.shape, X_angle.dtype)
	print( X.shape)

	return X, y

def get_Xy_CSS_fading( L = 250, y_l = range(5), SNRdB = None, f0T = 0.0, N = 250, isfading = False):
	"""
	Now, jamc_ch_pyx.get_Xy() supports no noise case (the ideal case) by 
	setting SNRdB as None instead of specifying a certain value.
	"""
	# Nm = 5
	# N = 250
	X_org, y= jamc_ch_pyx.get_Xy_fading( L = L, y_l = y_l, SNRdB = SNRdB, f0T = f0T, N = N, isfading = isfading) 		
	#X_org, y= jamc.getXy( f0T = 0.0, SNRdB = 100, L = 5000) 
	print(X_org.shape, y.shape)

	X_abs = np.abs( X_org)
	X_angle = np.mod( np.angle( X_org) / (2*np.pi), 1.0)

	X_abs.sort( axis = 1)
	X_angle.sort( axis = 1)

	X = np.concatenate( [X_abs, X_angle], axis = 1)
	print( X_abs.shape, X_abs.dtype)
	print( X_angle.shape, X_angle.dtype)
	print( X.shape)

	return X, y

if __name__ == '__main__':
	import sys

	print('Number of arguments:', len(sys.argv), 'arguments.')
	print('Argument List:', str(sys.argv))
	print()

	n_argv = len(sys.argv)
	if n_argv != 7:
		print("Usage")
		print("-----")
		print("python3 jamc_ch.py L y_l SNRdB f0T N")
		print("1. L (int) - #data for each modulation")
		print("2. y_l ([int...]- list of modulation index")
		print("   0=BPSK, 1=QPSK, 2=8PSK, 3=16QAM, 4=64QAM")
		print("3. SNRdB (float) - SNR in dB scale")
		print("4. f0T (float) - Normalized frequency error (0<=f0T<=1)")
		print("5. N (int) - # of symbols (#feature)")
		print("6. fname_out (string) - file name of generated channel data")
		print()
		print("E.g.")
		print("python3 jamc_ch.py 500 [0,1,2,3,4] 8 0.0 250 data.csv")
		print()
	else:
		L = int( sys.argv[1])
		print("  L (int) - #data for each modulation:", L)

		y_l = eval( sys.argv[2])
		print("  y_l ([int...]) - modulation index list:", y_l)

		SNRdB = float( sys.argv[3])
		print("  SNRdB (float) - SNR in dB scale:", SNRdB)

		f0T = float( sys.argv[4])
		print("  f0T (float) - Normalized frequency error (0<=f0T<=1):", f0T)

		N = int( sys.argv[5])
		print("  N (int) - # of symbols (#feature):", N)

		fname_out = sys.argv[6]
		print("  fname_out (string) - file name of generated channel data:", fname_out)

		X, y = get_Xy( L = L, y_l = y_l, SNRdB = SNRdB, f0T = f0T, N = N)

		x_c = ["x%d" % ix for ix in range( X.shape[1])]
		Xy_df = pd.DataFrame( X, columns=x_c)
		Xy_df["y"] = y
		Xy_df.to_csv("data.csv", index = False)

		print("-----------------")
		print("Your data with X{0}, y{1} are saved to a file of {2}".format( 
			X.shape, y.shape, fname_out))


