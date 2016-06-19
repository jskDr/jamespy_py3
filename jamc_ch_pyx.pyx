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


class CH():
	def __init__(self, int L = 500, y_l = range(5), SNRdB = None, float f0T = 0, int N = 250, fading_type = False):
		"""
		Modeling fading channel

		Inputs
		--------
		L: int
		The number of frames (or blocks)

		N: int
		The number of symbols in a frame

		SNRdB: float
		dB scale SNR values. SNRdB becomes infinity if SNRdB = None

		f0T: float (0,1)
		Normalized frequency offset

		fading_type: a string or False
		one of "fast", "block", "slow" or for AWGN, "awgn", "no"

		Editorial
		-----------
		A set of related functions can be divided into class member functions. 
		"""
		self.L = L
		self.y_l = y_l
		self.SNRdB = SNRdB
		self.f0T = f0T
		self.N = N
		self.fading_type = fading_type

	def get_Xy(self):
		return self.X, self.y

	def get_Xy_CSS(self):
		X_org = self.X

		X_abs = np.abs( X_org)
		X_angle = np.mod( np.angle( X_org) / (2*np.pi), 1.0)

		X_abs.sort( axis = 1)
		X_angle.sort( axis = 1)

		X = np.concatenate( [X_abs, X_angle], axis = 1)
		print( X_abs.shape, X_abs.dtype)
		print( X_angle.shape, X_angle.dtype)
		print( X.shape)

		return X, self.y

	def get_Xy_CSS_norm_abs(self):
		"""
		1. The maganitude is normalized by std of it.
		2. The phase is normalized by the first one. 
		3. The concatenated vector is normalized by std of it. 
		"""
		X_org = self.X

		X_abs = np.abs( X_org)
		X_angle = np.mod( np.angle( X_org) / (2*np.pi), 1.0)

		# Normalization is performed for each frame 
		# STD must be performed for 
		X_abs = X_abs / np.std( X_org, axis = 1, keepdims=True)
		# Equivalent to X_abs = X_abs / np.norm( X_org, axis = 1, keepdims=True) / sqrt( X_abs.shape[1])
		# X_angle = np.mod( X_angle - X_angle[:, 0:1], 1.0)

		X_abs.sort( axis = 1)
		X_angle.sort( axis = 1)

		X = np.concatenate( [X_abs, X_angle], axis = 1)
		# X = X / np.std( X, axis = 0)
		print( X_abs.shape, X_abs.dtype)
		print( X_angle.shape, X_angle.dtype)
		print( X.shape)

		return X, self.y

	def get_Xy_CSS_norm_angle(self):
		"""
		1. The maganitude is normalized by std of it.
		2. The phase is normalized by the first one. 
		3. The concatenated vector is normalized by std of it. 
		"""
		X_org = self.X

		X_abs = np.abs( X_org)
		X_angle = np.mod( np.angle( X_org) / (2*np.pi), 1.0)

		# Normalization is performed for each frame 
		# STD must be performed for 
		# X_abs = X_abs / np.std( X_org, axis = 1, keepdims=True)
		# Equivalent to X_abs = X_abs / np.norm( X_org, axis = 1, keepdims=True) / sqrt( X_abs.shape[1])
		X_angle = np.mod( X_angle - X_angle[:, 0:1], 1.0)

		X_abs.sort( axis = 1)
		X_angle.sort( axis = 1)

		X = np.concatenate( [X_abs, X_angle], axis = 1)
		# X = X / np.std( X, axis = 0)
		print( X_abs.shape, X_abs.dtype)
		print( X_angle.shape, X_angle.dtype)
		print( X.shape)

		return X, self.y

	def get_Xy_CSS_norm(self):
		"""
		1. The maganitude is normalized by std of it.
		2. The phase is normalized by the first one. 
		3. The concatenated vector is normalized by std of it. 
		"""
		X_org = self.X

		X_abs = np.abs( X_org)
		X_angle = np.mod( np.angle( X_org) / (2*np.pi), 1.0)

		# Normalization is performed for each frame 
		# STD must be performed for 
		X_abs = X_abs / np.std( X_org, axis = 1, keepdims=True)
		# Equivalent to X_abs = X_abs / np.norm( X_org, axis = 1, keepdims=True) / sqrt( X_abs.shape[1])
		X_angle = np.mod( X_angle - X_angle[:, 0:1], 1.0)

		X_abs.sort( axis = 1)
		X_angle.sort( axis = 1)

		X = np.concatenate( [X_abs, X_angle], axis = 1)
		# X = X / np.std( X, axis = 0)
		print( X_abs.shape, X_abs.dtype)
		print( X_angle.shape, X_angle.dtype)
		print( X.shape)

		return X, self.y

	def gen_r_with_m( self, m_y_a):
		"""
		Apr 23, 2016
		------------
		m will be get as argument and gen_s(m, y) will be called. 
		"""
		N = self.N
		SNRdB = self.SNRdB
		f0T = self.f0T		
		fading_type = self.fading_type
		
		s_n = gen_s_fast( N, m_y_a)
		if fading_type == "fast":
			ch_n = (np.random.randn( N) + 1j*np.random.randn( N)) / np.sqrt(2)	
			if SNRdB is not None:
				alpha = np.sqrt( np.power( 10.0, SNRdB / 10.0))
				g_n = (np.random.randn( N) + 1j*np.random.randn( N)) / np.sqrt(2)
				r_n = np.exp( 1j*f0T*np.arange(N)) * (ch_n * s_n) + g_n / alpha
			else:
				r_n = np.exp( 1j*f0T*np.arange(N)) * (ch_n * s_n)	
		elif fading_type == "block":
			ch = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)	
			if SNRdB is not None:
				alpha = np.sqrt( np.power( 10.0, SNRdB / 10.0))
				g_n = (np.random.randn( N) + 1j*np.random.randn( N)) / np.sqrt(2)
				r_n = np.exp( 1j*f0T*np.arange(N)) * (ch * s_n) + g_n / alpha
			else:
				r_n = np.exp( 1j*f0T*np.arange(N)) * (ch * s_n)	
		elif fading_type in ["awgn", "no"]:
			if SNRdB is not None:
				alpha = np.sqrt( np.power( 10.0, SNRdB / 10.0))
				g_n = (np.random.randn( N) + 1j*np.random.randn( N)) / np.sqrt(2)
				r_n = np.exp( 1j*f0T*np.arange(N)) * s_n + g_n / alpha
			else:
				r_n = np.exp( 1j*f0T*np.arange(N)) * s_n	

		return r_n	

	def gen_r_with_m_L( self, m_y_a):
		L = self.L
		N = self.N 
		# SNRdB = self.SNRdB
		# f0T = self.f0T

		X_1 = np.zeros( (L, N), dtype = complex)
		for l in range(L):
			X_1[l, :] = self.gen_r_with_m( m_y_a)
		return X_1

	def run( self):
		L = self.L
		y_l = self.y_l
		# SNRdB = self.SNRdB
		# f0T = self.f0T
		N = self.N

		X = np.zeros( (L * len( y_l), N), dtype = complex)
		y = np.zeros( L * len( y_l), dtype = int)

		for y_idx in range( len(y_l)):
			st = y_idx * L
			ed = (y_idx + 1) * L
			y[ st:ed] = y_l[ y_idx] # numpy propagation
			m_y_a = gen_m_a( y_l[ y_idx])
			X[ st:ed, :] = self.gen_r_with_m_L( m_y_a)

		self.X = X
		self.y = y

		return self

class CH_noise_norm( CH):
	def __init__(self, int L = 500, y_l = range(5), SNRdB = None, float f0T = 0, int N = 250, fading_type = False):
		super().__init__( L = L, y_l = y_l, SNRdB = SNRdB, f0T = f0T, N = N, fading_type = fading_type)
		self.noise_norm = True

	def gen_r_with_m( self, m_y_a):
		"""
		Depending on self.noise_norm
		-----------------------------------
		var(noise) becomes 1 if True
		var(signal) becomes 1 if False
		"""
		N = self.N
		SNRdB = self.SNRdB
		f0T = self.f0T		
		fading_type = self.fading_type
		
		s_n = gen_s_fast( N, m_y_a)

		if fading_type == "fast":
			ch_n = (np.random.randn( N) + 1j*np.random.randn( N)) / np.sqrt(2)	
			ch = ch_n
		elif fading_type == "block": 
			ch = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)	
		else: #fading_type in ["awgn", "no"]:
			ch = 1.0

		if SNRdB is not None:
			g_n = (np.random.randn( N) + 1j*np.random.randn( N)) / np.sqrt(2)
			alpha = np.sqrt( np.power( 10.0, SNRdB / 10.0))
			if self.noise_norm:
				# The variance of the noise becomes to be fixed one
				s_n *= alpha
			else:
				# The variance of the signal becomes to be fixed one except fading variation
				g_n /= alpha

			r_n = np.exp( 1j*f0T*np.arange(N)) * (ch * s_n) + g_n
		else:
			r_n = np.exp( 1j*f0T*np.arange(N)) * (ch * s_n)	

		return r_n		