cimport numpy as np
import numpy as np
# import pandas as pd

## Signal generation
def gen_m( int y):
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

def gen_m_a( int y):

	cdef np.ndarray[np.complex_t, ndim = 1] m

	if y == 0: # BPSK
		m = np.array( [1, -1], dtype = complex)
	elif y == 1: # QPSK, http://www.rfwireless-world.com/Terminology/QPSK.html
		m = np.array( [1+1j, -1+1j, -1-1j, 1-1j], dtype = complex)
	elif y == 2: # 8PSK
		m = np.exp( (1j*2*np.pi)*np.linspace( 0, 1, 8, endpoint = False), dtype = complex)
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

def gen_s_fast( int N, np.ndarray[np.complex_t, ndim=1] m_y_a):
	"""
	s is generated without rebuiding m_y	
	"""
	cdef np.ndarray[np.int_t, ndim=1] ix = np.random.randint( 0, m_y_a.shape[0], size = N)
	cdef np.ndarray[np.complex_t, ndim=1] s = m_y_a[ ix]
	return s

def gen_r_with_m_nonoise( np.ndarray[np.complex_t, ndim=1] m_y_a, int N = 10, float f0T = 0):
	"""
	Apr 23, 2016
	------------
	m will be get as argument and gen_s(m, y) will be called. 
	"""
	cdef np.ndarray[np.complex_t, ndim=1] s_n = gen_s_fast( N, m_y_a)
	cdef np.ndarray[np.complex_t, ndim=1] r_n 

	r_n = np.exp( 1j*f0T*np.arange(N)) * s_n
	return r_n	

def gen_r_with_m( np.ndarray[np.complex_t, ndim=1] m_y_a, int N = 10, SNRdB = None, float f0T = 0, isfading = False):
	"""
	Apr 23, 2016
	------------
	m will be get as argument and gen_s(m, y) will be called. 
	"""
	cdef float alpha 
	cdef np.ndarray[np.complex_t, ndim=1] s_n = gen_s_fast( N, m_y_a)
	cdef np.ndarray[np.complex_t, ndim=1] g_n
	cdef np.ndarray[np.complex_t, ndim=1] ch_n	
	cdef np.ndarray[np.complex_t, ndim=1] r_n

	if isfading:
		ch_n = (np.random.randn( N) + 1j*np.random.randn( N)) / np.sqrt(2)	
		if SNRdB is not None:
			alpha = np.sqrt( np.power( 10.0, SNRdB / 10.0))
			g_n = (np.random.randn( N) + 1j*np.random.randn( N)) / np.sqrt(2)
			r_n = np.exp( 1j*f0T*np.arange(N)) * (ch_n * s_n) + g_n / alpha
		else:
			r_n = np.exp( 1j*f0T*np.arange(N)) * (ch_n * s_n)	
	else:
		if SNRdB is not None:
			alpha = np.sqrt( np.power( 10.0, SNRdB / 10.0))
			g_n = (np.random.randn( N) + 1j*np.random.randn( N)) / np.sqrt(2)
			r_n = np.exp( 1j*f0T*np.arange(N)) * s_n + g_n / alpha
		else:
			r_n = np.exp( 1j*f0T*np.arange(N)) * s_n	

	return r_n	

def _gen_r_with_m_r0( np.ndarray[np.complex_t, ndim=1] m_y_a, int N = 10, float SNRdB = 0, float f0T = 0):
	"""
	Apr 23, 2016
	------------
	m will be get as argument and gen_s(m, y) will be called. 
	"""
	cdef float alpha = np.sqrt( np.power( 10.0, SNRdB / 10.0))
	cdef np.ndarray[np.complex_t, ndim=1] s_n = gen_s_fast( N, m_y_a)
	cdef np.ndarray[np.complex_t, ndim=1] g_n = (np.random.randn( N) + 1j*np.random.randn( N)) / np.sqrt(2)
	cdef np.ndarray[np.complex_t, ndim=1] r_n = np.exp( 1j*f0T*np.arange(N)) * s_n + g_n / alpha
	return r_n	

def gen_r_with_m_L( np.ndarray[np.complex_t, ndim=1] m_y_a, int L, int N = 10, SNRdB = None, float f0T = 0, isfading = False):
	cdef np.ndarray[np.complex_t, ndim=2] X_1 = np.zeros( (L, N), dtype = complex)

	for l in range(L):
		X_1[l, :] = gen_r_with_m( m_y_a, N = N, SNRdB =SNRdB, f0T = f0T, isfading = isfading)
	return X_1

def _gen_r_with_m_L_r0( np.ndarray[np.complex_t, ndim=1] m_y_a, int L, int N = 10, float SNRdB = 0, float f0T = 0):
	cdef np.ndarray[np.complex_t, ndim=2] X_1 = np.zeros( (L, N), dtype = complex)

	for l in range(L):
		X_1[l, :] = gen_r_with_m( m_y_a, N = N, SNRdB = SNRdB, f0T = f0T)
	return X_1

def gen_r_with_m_L_nonoise( np.ndarray[np.complex_t, ndim=1] m_y_a, int L, int N = 10, float f0T = 0):
	cdef np.ndarray[np.complex_t, ndim=2] X_1 = np.zeros( (L, N), dtype = complex)

	for l in range(L):
		X_1[l, :] = gen_r_with_m_nonoise( m_y_a, N = N, f0T = f0T)
	return X_1

def get_Xy( int L = 500, y_l = range(5), float SNRdB = 8, float f0T = 0, int N = 250, isfading = False):
	"""
	get_Xy( int L = 500, y_l = range(5), float SNRdB = 8, float f0T = 0, int N = 250):
	"""

	cdef np.ndarray[np.complex_t, ndim=2] X = np.zeros( (L * len( y_l), N), dtype = complex)
	cdef np.ndarray[np.int_t, ndim=1] y = np.zeros( L * len( y_l), dtype = int)
	cdef int y_idx
	# cdef np.ndarray[np.int_t, ndim=1] ones_L = np.ones( L)

	for y_idx in range( len(y_l)):
		st = y_idx * L
		ed = (y_idx + 1) * L
		y[ st:ed] = y_l[ y_idx] # numpy propagation
		m_y_a = gen_m_a( y_l[ y_idx])
		X[ st:ed, :] = gen_r_with_m_L( m_y_a, L = L, N = N, SNRdB = SNRdB, f0T = f0T, isfading = isfading)

	return X, y	

def get_Xy_nonoise( int L = 500, y_l = range(5), float f0T = 0, int N = 250):
	"""
	Noise will not be added.
	"""

	cdef np.ndarray[np.complex_t, ndim=2] X = np.zeros( (L * len( y_l), N), dtype = complex)
	cdef np.ndarray[np.int_t, ndim=1] y = np.zeros( L * len( y_l), dtype = int)
	cdef int y_idx
	# cdef np.ndarray[np.int_t, ndim=1] ones_L = np.ones( L)

	for y_idx in range( len(y_l)):
		st = y_idx * L
		ed = (y_idx + 1) * L
		y[ st:ed] = y_l[ y_idx] # numpy propagation
		m_y_a = gen_m_a( y_l[ y_idx])
		X[ st:ed, :] = gen_r_with_m_L_nonoise( m_y_a, L = L, N = N, f0T = f0T)

	return X, y	

def get_Xy_fading( int L = 500, y_l = range(5), SNRdB = None, float f0T = 0, int N = 250, isfading = False):
	"""
	get_Xy( int L = 500, y_l = range(5), float SNRdB = 8, float f0T = 0, int N = 250):
	"""

	cdef np.ndarray[np.complex_t, ndim=2] X = np.zeros( (L * len( y_l), N), dtype = complex)
	cdef np.ndarray[np.int_t, ndim=1] y = np.zeros( L * len( y_l), dtype = int)
	cdef int y_idx
	# cdef np.ndarray[np.int_t, ndim=1] ones_L = np.ones( L)

	for y_idx in range( len(y_l)):
		st = y_idx * L
		ed = (y_idx + 1) * L
		y[ st:ed] = y_l[ y_idx] # numpy propagation
		m_y_a = gen_m_a( y_l[ y_idx])
		X[ st:ed, :] = gen_r_with_m_L( m_y_a, L = L, N = N, SNRdB = SNRdB, f0T = f0T, isfading = isfading)

	return X, y	
