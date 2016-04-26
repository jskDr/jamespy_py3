# Py3

import numpy as np

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