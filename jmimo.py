"""
Author
--------
Best regards, 
Sungjin (James) Kim, PhD
Postdoc, CCB in Harvard
sungjinkim@fas.harvard.edu

[Web] http://aspuru.chem.harvard.edu/james-sungjin-kim/ 
[Linkedin] https://www.linkedin.com/in/jamessungjinkim 
[Facebook] https://www.facebook.com/jamessungjin.kim 
[alternative email] jamessungjin.kim@gmail.com 

Licence
---------
MIT License

"""
from __future__ import print_function
# I started to use __future__ so as to be compatible with Python3

import numpy as np
from sklearn import linear_model
from sklearn import cross_validation
from sklearn import metrics
import pandas as pd
from collections import OrderedDict

# To improve the speed, I using pyx. 
import jpyx
import jutil
from jsklearn import codes

def mld( r_l, mod_l = [-0.70710678, 0.70710678]):
	"""
	maximum likelihood detection
	
	r_l: received signals after reception processing
	mod_l: list of all modulation signals
		BPSK: [-0.70710678, 0.70710678]
	
	return the demodulated signals (0, 1, ...)
	"""
	sd_l = list() # store demodulated signal
	for r in r_l:
		dist = list() #Store distance
		for m in mod_l:
			d = np.power( np.abs( r - m), 2)
			dist.append( d)
		sd = np.argmin( dist)
		sd_l.append( sd)
	return np.array( sd_l)

def calc_BER( r_l, x_l):
	"""
	calculate bit error rate (BER)
	r_l: demodulated signals (ndarray, 1D)
	x_l: transmitted signals (ndarray, 1D)
	"""
	err_l = r_l - x_l
	errs = np.where( err_l != 0)[0]
	# print 'err_l =', err_l 
	# print 'errs =', errs
	Nerr = len(np.where( err_l != 0)[0])
	return float( Nerr) / len( err_l), Nerr 

def db2var( SNRdB):
	return np.power( 10.0, SNRdB / 10.0)

def gen_BPSK(Nx, Nt):
	"""
	Generate BPSK modulated signals
	"""
	BPSK = np.array( [1, -1]) / np.sqrt( 2.0)
	s_a = np.random.randint( 0, 2, Nx * Nt)
	x_flat_a = BPSK[ s_a]
	x_a = np.reshape( x_flat_a, (Nx, Nt))

	return BPSK, s_a, x_flat_a, x_a

def gen_H( Nr, Nt):
	return np.random.randn( Nr, Nt)

def gen_Rx( Nr, Nx, SNR, H_a, x_a):
	"""
	The received signals are modeled.
	"""
	n_a = np.random.randn( Nr, Nx) / np.sqrt( SNR)
	y_a = np.dot( H_a, x_a.T) + n_a	

	return y_a

def normalize( W_a):
	"Weight is normalized."

	nW_a = np.linalg.norm( W_a, axis = 1)
	for a0 in range( W_a.shape[0]):
		W_a[a0,:] = np.divide( W_a[a0,:], nW_a[a0])

	return W_a

class MIMO(object):
	"""
	Modeling for a MIMO wireless communication system.
	"""
	def __init__(self, Nt = 2, Nr = 4, Nx = 10, SNRdB = 10, model = "Ridge", Npilot = 10, Nloop = 10):
		"""
		The parameter of 'model' determines the regression method.
		"""
		self.set_param( (Nt, Nr, Nx, SNRdB))
		self.model = model
		self.Npilot = Npilot
		self.Nloop = Nloop

		# The function of test_ridge_all() uses 3 cases for testing. 
		# self.N_test_ridge_all = 3

	def set_param( self, param_NtNrNxSNRdB):

		Nt, Nr, Nx, SNRdB	 = param_NtNrNxSNRdB
		
		# The antenna configuration is conducted.
		self.Nt = Nt
		self.Nr = Nr
		# No of streams is fixed.
		self.Nx = Nx

		# Initial SNR is defined
		self.SNRdB = SNRdB
		self.SNR = db2var(SNRdB)

	def _gen_BPSK_r0(self):
		"""
		Generate BPSK modulated signals
		"""
		self.BPSK = np.array( [1, -1]) / np.sqrt( 2.0)
		self.s_a = np.random.randint( 0, 2, self.Nx * self.Nt)
		self.x_flat_a = self.BPSK[ self.s_a]
		self.x_a = np.reshape( self.x_flat_a, (self.Nx, self.Nt))

	def gen_BPSK( self):
		"""
		Generate BPSK signals using global function gen_BPSK().
		This function will be used to generate pilot signal as well. 
		"""
		self.BPSK, self.s_a, self.x_flat_a, self.x_a = gen_BPSK( self.Nx, self.Nt)

	def gen_H(self):
		"""
		The MIMO channel is generated.
		"""
		self.H_a = gen_H( self.Nr, self.Nt)

	def _gen_Rx_r0(self):
		"""
		The received signals are modeled.
		"""
		self.n_a = np.random.randn( self.Nr, self.Nx) / np.sqrt( self.SNR)
		self.y_a = np.dot( self.H_a, self.x_a.T) + self.n_a

	def gen_Rx(self):
		"""
		The received signals are modeled.
		"""
		self.y_a = gen_Rx( self.Nr, self.Nx, self.SNR, self.H_a, self.x_a)

	def gen_WR_ideal(self): 
		"""
		The reception process with ideal channel estimation 
		is conducted.
		each reception vector of W_a should be noramlized to one.
		"""
		self.W_a = np.linalg.pinv( self.H_a)
		# The reception signal vector is transposed.

		self.gen_Decoding()

	def gen_WR_pilot(self, pilot_SNRdB):

		"""
		The reception process with pilot channel estimation
		is conducted.
		Pilot will be transmitted through random information channel.
		"""
		pilot_SNR = db2var(pilot_SNRdB)
		N_a = np.random.randn( *self.H_a.shape) / np.sqrt( pilot_SNR)
		Hp_a = self.H_a + N_a
		self.W_a = np.linalg.pinv( Hp_a)

		self.gen_Decoding()

	def gen_WR_pilot_channel(self, pilot_SNRdB):

		"""
		The reception process with pilot channel estimation
		is conducted.
		"""
		Npilot = self.Npilot
		SNRpilot = db2var( pilot_SNRdB)

		BPSK, s_a, x_flat_a, x_a = gen_BPSK( Npilot, self.Nt)
		# H_a = gen_H( self.Nr, self.Nt)
		# H_a = self.H_a
		y_a = gen_Rx( self.Nr, Npilot, SNRpilot, self.H_a, x_a)

		yT_a = y_a.T

		# print( x_a.shape, yT_a.shape)

		lm = linear_model.LinearRegression()
		lm.fit( yT_a, x_a)
		"""
		Power normalization should be considered 
		unless it is multiplied with both sinal and noise. 
		In this case, MMSE weight is calculated while
		pinv() obtain ZF filter.  
		"""
		self.W_a = lm.coef_

		# print( "np.dot( W_a, H_a) =", np.dot( self.W_a, self.H_a))

		self.gen_Decoding()

	def gs_pilot_reg_only(self, alpha_l):
		"""
		Grid search is applied for alpha_l.
		Later, the best alpha will be selected and decode data using it.
		"""

		pdo = pd.DataFrame()
		for alpha in alpha_l:
			pdi = self.cv_pilot_reg_only( alpha)
			pdo = pdo.append( pdi, ignore_index = True)

		return pdo

	def gs_pilot_reg_full(self, alpha_l):
		"""
		Full means data and pilot are both generated and processed including data decoding
		"""
		self.gen_BPSK()
		self.gen_H()
		self.gen_Rx()
		self.rx_pilot()

		return self.gs_pilot_reg_only( alpha_l)

	def gs_pilot_reg_best(self, alpha_l):
		"""
		Find the best alpha using Ridge regression.
		Return
		--------
		The best alpha is returned.
		"""
		pdi = self.gs_pilot_reg_only( alpha_l)
		# print( 'pdi["E[scores]"]', pdi["E[scores]"])
		i_max = np.argmin( pdi["E[scores]"])
		alpha_best = pdi["alpha"][i_max]

		return alpha_best

	def gs_pilot_reg_best_full(self, alpha_l):
		"""
		Full means data and pilot are both generated and processed including data decoding
		"""
		self.gen_BPSK()
		self.gen_H()
		self.gen_Rx()
		self.rx_pilot()

		return self.gs_pilot_reg_best( alpha_l)

	def rx_pilot(self):

		Npilot = self.Npilot
		SNRpilot = self.SNR

		BPSK, s_a, x_flat_a, x_a = gen_BPSK( Npilot, self.Nt)
		# H_a = gen_H( self.Nr, self.Nt)
		# H_a = self.H_a
		y_a = gen_Rx( self.Nr, Npilot, SNRpilot, self.H_a, x_a)

		yT_a = y_a.T

		self.rx_p = dict()
		self.rx_p["yT_a"] = yT_a
		self.rx_p["x_a"] = x_a

	def cv_pilot_only(self):

		"""
		Cross-validatin scores are evaluated using LOO. 
		SNRpilot is equal to SNR, which is SNRdata.		
		"""
		yT_a = self.rx_p["yT_a"]
		x_a = self.rx_p["x_a"]

		lm = linear_model.LinearRegression()
		scores = codes.cross_val_score_loo( lm, yT_a, x_a)

		# Output is stored with enviromental variables.
		pdi = pd.DataFrame()
		pdi["model"] = ["LinearRegression"]
		pdi["alpha"] = [0]
		pdi["metric"] = ["mean_squared_error"]
		pdi["E[scores]"] = [np.mean(scores)]
		pdi["std[scores]"] = [np.std(scores)]
		pdi["scores"] = [scores]

		return pdi

	def cv_pilot( self):
		self.rx_pilot()
		return self.cv_pilot_only()

	def _cv_pilot_reg_only_r0(self, alpha = 0):
		model = self.model
		yT_a = self.rx_p["yT_a"]
		x_a = self.rx_p["x_a"]

		# kf = KFold() 
		# loo = cross_validation.LeaveOneOut( x_a.shape[0])
		if alpha == 0:
			lm = linear_model.LinearRegression()
		else:
			lm = getattr( linear_model, model)(alpha)
		scores = codes.cross_val_score_loo( lm, yT_a, x_a)

		return scores

	def cv_pilot_reg_only(self, alpha = 0):
		model = self.model
		yT_a = self.rx_p["yT_a"]
		x_a = self.rx_p["x_a"]

		# kf = KFold() 
		# loo = cross_validation.LeaveOneOut( x_a.shape[0])
		if alpha == 0:
			lm = linear_model.LinearRegression()
		else:
			lm = getattr( linear_model, model)(alpha)
		scores = codes.cross_val_score_loo( lm, yT_a, x_a)

		# Output is stored with enviromental variables.
		pdi = pd.DataFrame()
		pdi["model"] = [model]
		pdi["alpha"] = [alpha]
		pdi["metric"] = ["mean_squared_error"]
		pdi["E[scores]"] = [np.mean(np.power(scores,2))] # MSE
		pdi["std[scores]"] = ["t.b.d."]
		pdi["scores"] = [scores]

		return pdi

	def cv_pilot_reg( self, alpha = 0):
		self.rx_pilot()
		return self.cv_pilot_reg_only( alpha)

	def _cv_pilot_reg_r0(self, alpha = 0):

		"""
		Cross-validatin scores are evaluated using LOO. 
		SNRpilot is equal to SNR, which is SNRdata.		
		"""
		Npilot = self.Npilot
		SNRpilot = self.SNR
		model = self.model

		BPSK, s_a, x_flat_a, x_a = gen_BPSK( Npilot, self.Nt)
		# H_a = gen_H( self.Nr, self.Nt)
		# H_a = self.H_a
		y_a = gen_Rx( self.Nr, Npilot, SNRpilot, self.H_a, x_a)

		yT_a = y_a.T

		# kf = KFold() 
		# loo = cross_validation.LeaveOneOut( x_a.shape[0])
		if alpha == 0:
			lm = linear_model.LinearRegression()
		else:
			lm = getattr( linear_model, model)(alpha)
		scores = codes.cross_val_score_loo( lm, yT_a, x_a)

		return scores

	def _gen_WR_pilot_ch_r0(self, pilot_SNRdB, alpha = 0):

		"""
		The reception process with pilot channel estimation
		is conducted.
		"""
		Npilot = 10
		SNRpilot = db2var( pilot_SNRdB)

		BPSK, s_a, x_flat_a, x_a = gen_BPSK( Npilot, self.Nt)
		# H_a = gen_H( self.Nr, self.Nt)
		# H_a = self.H_a
		y_a = gen_Rx( self.Nr, Npilot, SNRpilot, self.H_a, x_a)

		yT_a = y_a.T

		# print( x_a.shape, yT_a.shape)

		lm = linear_model.Ridge( alpha)
		lm.fit( yT_a, x_a)
		self.W_a = lm.coef_

		# print( "np.dot( W_a, H_a) =", np.dot( self.W_a, self.H_a))

		self.gen_Decoding()

	def _gen_WR_pilot_ch_r1(self, pilot_SNRdB, alpha = 0, model = "Ridge"):

		"""
		The reception process with pilot channel estimation
		is conducted.
		"""
		Npilot = 10
		SNRpilot = db2var( pilot_SNRdB)

		BPSK, s_a, x_flat_a, x_a = gen_BPSK( Npilot, self.Nt)
		# H_a = gen_H( self.Nr, self.Nt)
		# H_a = self.H_a
		y_a = gen_Rx( self.Nr, Npilot, SNRpilot, self.H_a, x_a)

		yT_a = y_a.T

		# print( x_a.shape, yT_a.shape)

		# Now you can use either Ridge or Lasso methods. 
		#lm = linear_model.Ridge( alpha)		
		lm = getattr( linear_model, model)(alpha)
		lm.fit( yT_a, x_a)
		self.W_a = lm.coef_

		# print( "np.dot( W_a, H_a) =", np.dot( self.W_a, self.H_a))

		self.gen_Decoding()		

	def gen_WR_pilot_ch(self, pilot_SNRdB, alpha_l1r = 0, model = "Ridge"):

		"""
		The reception process with pilot channel estimation
		is conducted.
		"""
		Npilot = self.Npilot

		SNRpilot = db2var( pilot_SNRdB)

		BPSK, s_a, x_flat_a, x_a = gen_BPSK( Npilot, self.Nt)
		# H_a = gen_H( self.Nr, self.Nt)
		# H_a = self.H_a
		y_a = gen_Rx( self.Nr, Npilot, SNRpilot, self.H_a, x_a)

		yT_a = y_a.T

		# print( x_a.shape, yT_a.shape)

		# Now you can use either Ridge or Lasso methods. 
		#lm = linear_model.Ridge( alpha)
		if model == "ElasticNet":
			lm = linear_model.ElasticNet( alpha_l1r[0], alpha_l1r[1])
		else:
			lm = getattr( linear_model, model)(alpha_l1r)

		lm.fit( yT_a, x_a)
		self.W_a = lm.coef_

		# print( "np.dot( W_a, H_a) =", np.dot( self.W_a, self.H_a))

		self.gen_Decoding()		

	def gen_WR_pilot_only(self, alpha_l1r = 0):
		"""
		yT_a and x_a was prepared already. 
		Now, W_a is calculated using alpha and then, 
		decode data. 
		For linear regression, alpha_l1r should not be specified except 0.
		"""

		yT_a = self.rx_p["yT_a"]
		x_a = self.rx_p["x_a"]

		# for alpha == 0, model is changed to linear regression.  
		if alpha_l1r == 0:
			model = "LinearRegression"
		else:
			model = self.model

		if model == "LinearRegression":
			lm = linear_model.LinearRegression()
		elif model == "ElasticNet":
			lm = linear_model.ElasticNet( alpha_l1r[0], alpha_l1r[1])
		else: # This is either Ridge or Lasso
			lm = getattr( linear_model, model)(alpha_l1r)

		lm.fit( yT_a, x_a)
		self.W_a = lm.coef_

		# print( "np.dot( W_a, H_a) =", np.dot( self.W_a, self.H_a))

		self.gen_Decoding()

	def gen_WR( self, pilot_SNRdB = None):
		if pilot_SNRdB:
			gen_WR_pilot( pilot_SNRdB)
		else:
			gen_WR_ideal()

	def gen_Decoding(self): 
		"""
		The reception process is conducted.
		"""
		self.W_a = normalize( self.W_a) # not important (useless at this moment)

		self.rT_a = np.dot( self.W_a, self.y_a)

		self.r_flat_a = self.rT_a.T.flatten()
		#print( "type( self.r_flat_a), type( self.BPSK)")
		#print( type( self.r_flat_a), type( self.BPSK))
		# self.sd_a = jpyx.mld( self.r_flat_a, self.BPSK)
		self.sd_a = jpyx.mld_fast( self.r_flat_a, self.BPSK)
		self.BER, self.Nerr = calc_BER( self.s_a, self.sd_a)


	def run_ideal( self, param_NtNrNxSNRdB = None, Nloop = 10, disp = False):
		"""
		A system is run from the transmitter to the receiver.

		"""
		return self.run_pilot( param_NtNrNxSNRdB = param_NtNrNxSNRdB, Nloop = Nloop, disp = disp)


	def run_pilot( self, pilot_SNRdB = None, param_NtNrNxSNRdB = None, Nloop = 10, disp = False):
		"""
		A system is run from the transmitter to the receiver.

		"""
		if param_NtNrNxSNRdB:
			self.set_param( param_NtNrNxSNRdB)

		self.gen_BPSK()

		BER_l = list()
		Nerr_total = 0
		for nloop in range( Nloop):
			self.gen_H()
			self.gen_Rx()

			if pilot_SNRdB is not None:
				self.gen_WR_pilot( pilot_SNRdB)
			else: 
				self.gen_WR_ideal()

			BER_l.append( self.BER)
			Nerr_total += self.Nerr

		self.BER = np.mean( BER_l)

		if disp:
			Ntot = self.Nt * self.Nx * Nloop
			print( "BER is {} with {}/{} errors at {} SNRdB ".format( self.BER, Nerr_total, Ntot, self.SNRdB))

		return self.BER

	def run_pilot_channel( self, pilot_SNRdB = None, param_NtNrNxSNRdB = None, Nloop = 10, disp = False):
		"""
		A system is run from the transmitter to the receiver.

		"""
		if param_NtNrNxSNRdB:
			self.set_param( param_NtNrNxSNRdB)

		self.gen_BPSK()

		BER_l = list()
		Nerr_total = 0
		for nloop in range( Nloop):
			self.gen_H()
			self.gen_Rx()

			if pilot_SNRdB is not None:
				# self.gen_WR_pilot( pilot_SNRdB)
				self.gen_WR_pilot_channel( pilot_SNRdB)
				# self.gen_WR_pilot_ch( pilot_SNRdB, alpha)
			else: 
				self.gen_WR_ideal()

			BER_l.append( self.BER)
			Nerr_total += self.Nerr

		self.BER = np.mean( BER_l)

		if disp:
			Ntot = self.Nt * self.Nx * Nloop
			print( "BER is {} with {}/{} errors at {} SNRdB ".format( self.BER, Nerr_total, Ntot, self.SNRdB))

		return self.BER

	def run_pilot_ch( self, pilot_SNRdB = None, param_NtNrNxSNRdB = None, Nloop = 10, alpha = 0, disp = False):
		"""
		A system is run from the transmitter to the receiver.

		"""
		if param_NtNrNxSNRdB:
			self.set_param( param_NtNrNxSNRdB)

		self.gen_BPSK()

		BER_l = list()
		Nerr_total = 0
		for nloop in range( Nloop):
			self.gen_H()
			self.gen_Rx()

			if pilot_SNRdB:
				# self.gen_WR_pilot( pilot_SNRdB)
				# self.gen_WR_pilot_channel( pilot_SNRdB)
				self.gen_WR_pilot_ch( pilot_SNRdB, alpha)
			else: 
				self.gen_WR_ideal()

			BER_l.append( self.BER)
			Nerr_total += self.Nerr

		self.BER = np.mean( BER_l)

		if disp:
			Ntot = self.Nt * self.Nx * Nloop
			print( "BER is {} with {}/{} errors at {} SNRdB ".format( self.BER, Nerr_total, Ntot, self.SNRdB))

		return self.BER

	def test_ridge_iter( self, alpha_l):

		# Ideal ZF(H)
		ID = 0
		self.method = "Ideal ZF(H)"
		self.model = "ZF"
		self.alpha = 0
		self.gen_WR_ideal()
		yield ID

		# Multiple Ridge regressions with alpha_l
		for alpha in alpha_l:
			ID += 1
			self.method = "Ridge each"
			self.model = "Ridge"
			self.alpha = alpha
			self.gen_WR_pilot_only( self.alpha)
			yield ID

		# Ridge regression with the best alpha among alpha_l
		ID += 1
		self.method = "Ridge best"
		self.model = "Ridge"
		self.alpha = self.gs_pilot_reg_best( alpha_l)
		self.gen_WR_pilot_only( self.alpha)
		yield ID

	def test_ridge_all( self, pdi_d_prev, alpha_l):
		"""
		1. LinearRegression
		2. multiple Ridge regression with each alpha in alpha_l
		3. Ridge regression with the best alpha among alpha_l
		"""

		# pdi_d is generated only once. 
		if pdi_d_prev is None:
			pdi_d = dict()
		else:
			pdi_d = pdi_d_prev

		for ID in self.test_ridge_iter(alpha_l):

			"""
			If pdi_l is not defined yet, 
			it will be generated first and initial values are stored.
			Otherwise, new data are added for the corresponding space.
			"""
			if pdi_d_prev is None:
				pdi = pd.DataFrame()
				pdi["Nerr_total"] = [0]
				pdi["BER_l"] = [[self.BER]]
			else: 
				pdi = pdi_d[ ID]
				pdi["Nerr_total"] = [ pdi["Nerr_total"][0] + self.Nerr]
				pdi["BER_l"] = [pdi["BER_l"][0] + [self.BER]]

			pdi["method"] = [self.method]
			pdi["model"] = [self.model]
			pdi["alpha"] = [self.alpha]
			# print( 'pdi["BER_l"]', pdi["BER_l"])
			pdi["BER"] = [np.mean( pdi["BER_l"][0])]

			pdi_d[ ID] = pdi

		return pdi_d

	def run_gs_pilot_Ridge( self, alpha_l):
		"""
		Search the best alpha using Ridge.
		I focus on Ridge for simplicity at this moment. 
		Other regularization modes will be used later on.
		"""

		Nloop = self.Nloop

		pdi_d = None
		for nloop in range( Nloop):
			self.gen_BPSK()
			self.gen_H()
			self.gen_Rx()
			# For fair comparision, pilot is also generated commonly for all methods.
			self.rx_pilot() 

			pdi_d = self.test_ridge_all( pdi_d, alpha_l)	

		pdo = pd.DataFrame()
		for pdi in pdi_d.values():
			pdo = pdo.append( pdi, ignore_index = True)

		return pdo

	def run_pilot_ch_model( self, pilot_SNRdB = None, param_NtNrNxSNRdB = None, Nloop = 10, alpha = 0, disp = False):
		"""
		A system is run from the transmitter to the receiver.
		self.model is used to determine the regression model such as Ridge and Lasso

		"""
		if param_NtNrNxSNRdB:
			self.set_param( param_NtNrNxSNRdB)

		self.gen_BPSK()

		BER_l = list()
		Nerr_total = 0
		for nloop in range( Nloop):
			self.gen_H()
			self.gen_Rx()

			if pilot_SNRdB is not None: # 'is' needed for checking None
				# self.gen_WR_pilot( pilot_SNRdB)
				# self.gen_WR_pilot_channel( pilot_SNRdB)
				self.gen_WR_pilot_ch( pilot_SNRdB, alpha, self.model)
			else: 
				self.gen_WR_ideal()

			BER_l.append( self.BER)
			Nerr_total += self.Nerr

		self.BER = np.mean( BER_l)

		if disp:
			Ntot = self.Nt * self.Nx * Nloop
			print( "BER is {} with {}/{} errors at {} SNRdB ".format( self.BER, Nerr_total, Ntot, self.SNRdB))

		return self.BER

	def get_BER_pilot_ch_model_eqsnr( 
		self,
		SNRdB_l = [5,6,7], 
		param_NtNrNx = (2,4,100), 
		Nloop = 1000, 
		pilot_ch = False, 
		alpha = 0, 
		model = "Ridge"):
		"""
		Ridge regression will be using to estimate channel.
		If alpha is zero, linear regression will be applied.
		If alpha is more than zero, Ridge regression will be applied.
		The default value of alpha is zero. 
		"""

		Nt, Nr, Nx = param_NtNrNx	

		BER_pilot = list()

		for SNRdB in SNRdB_l:
			# if pilot channel is used, SNRdB is given
			# Otherwise, ideal channel estimation is assumed.
			if pilot_ch:
				pilot_SNRdB = SNRdB
			else:
				pilot_SNRdB = None

			if alpha > 0:
				"""
				Ridge or Lasso is used.
				"""
				self.model = model
				ber = self.run_pilot_ch_model( pilot_SNRdB = pilot_SNRdB, 
					param_NtNrNxSNRdB =(Nt, Nr, Nx, SNRdB), Nloop = Nloop, alpha = alpha, disp = True)
				BER_pilot.append( ber)
			else:
				"""
				LinearRegression is used.
				"""
				ber = self.run_pilot_channel( pilot_SNRdB = pilot_SNRdB, 
					param_NtNrNxSNRdB =(Nt, Nr, Nx, SNRdB), Nloop = Nloop, disp = True)
				BER_pilot.append( ber)

		# print( "List of average BERs =", BER_pilot)

		return BER_pilot


	def get_BER_pilot_ch_model( self,
		SNRdB_l = [5,6,7], 
		param_NtNrNx = (2,4,100), 
		Nloop = 1000, 
		pilot_SNRdB = None, 
		alpha = 0, 
		model = "Ridge"):
		"""
		Ridge regression will be using to estimate channel.
		If alpha is zero, linear regression will be applied.
		If alpha is more than zero, Ridge regression will be applied.
		The default value of alpha is zero. 
		This function becomes a member function of class MIMO.
		"""

		BER_pilot = list()
		Nt, Nr, Nx = param_NtNrNx	
		if alpha > 0:
			"""
			Ridge or Lasso is used.
			"""
			for SNRdB in SNRdB_l:
				self.model = model
				ber = self.run_pilot_ch_model( pilot_SNRdB = pilot_SNRdB, 
					param_NtNrNxSNRdB =(Nt, Nr, Nx, SNRdB), Nloop = Nloop, alpha = alpha, disp = True)
				BER_pilot.append( ber)
		else:
			"""
			LinearRegression is used.
			"""
			for SNRdB in SNRdB_l:
				ber = self.run_pilot_channel( pilot_SNRdB = pilot_SNRdB, 
					param_NtNrNxSNRdB =(Nt, Nr, Nx, SNRdB), Nloop = Nloop, disp = True)
				BER_pilot.append( ber)

		# print( "List of average BERs =", BER_pilot)

		return BER_pilot

def get_BER( SNRdB_l = [5,6,7], param_NtNrNx = (2,4,100), Nloop = 1000, pilot_SNRdB = None):
	BER_pilot = list()

	Nt, Nr, Nx = param_NtNrNx	
	for SNRdB in SNRdB_l:
		ber = MIMO().run_pilot( pilot_SNRdB = pilot_SNRdB, 
			param_NtNrNxSNRdB =(Nt, Nr, Nx, SNRdB), Nloop = Nloop, disp = True)
		BER_pilot.append( ber)

	# print( "List of average BERs =", BER_pilot)

	return BER_pilot

def get_BER_pilot_ch( SNRdB_l = [5,6,7], param_NtNrNx = (2,4,100), Nloop = 1000, pilot_SNRdB = None, alpha = 0):
	"""
	Ridge regression will be using to estimate channel.
	If alpha is zero, linear regression will be applied.
	If alpha is more than zero, Ridge regression will be applied.
	The default value of alpha is zero. 
	"""
	BER_pilot = list()

	Nt, Nr, Nx = param_NtNrNx	
	if alpha > 0:
		"""
		LinearRegression is using.
		"""
		for SNRdB in SNRdB_l:
			ber = MIMO().run_pilot_ch( pilot_SNRdB = pilot_SNRdB, 
				param_NtNrNxSNRdB =(Nt, Nr, Nx, SNRdB), Nloop = Nloop, alpha = alpha, disp = True)
			BER_pilot.append( ber)
	else:
		"""
		Ridge is using.
		"""
		for SNRdB in SNRdB_l:
			ber = MIMO().run_pilot_channel( pilot_SNRdB = pilot_SNRdB, 
				param_NtNrNxSNRdB =(Nt, Nr, Nx, SNRdB), Nloop = Nloop, disp = True)
			BER_pilot.append( ber)

	# print( "List of average BERs =", BER_pilot)

	return BER_pilot

def get_BER_pilot_ch_model( 
	SNRdB_l = [5,6,7], 
	param_NtNrNx = (2,4,100), 
	Nloop = 1000, 
	pilot_SNRdB = None, 
	alpha = 0, 
	model = "Ridge"):
	"""
	Ridge regression will be using to estimate channel.
	If alpha is zero, linear regression will be applied.
	If alpha is more than zero, Ridge regression will be applied.
	The default value of alpha is zero. 
	"""

	BER_pilot = list()

	Nt, Nr, Nx = param_NtNrNx	
	if alpha > 0:
		"""
		Ridge or Lasso is used.
		"""
		for SNRdB in SNRdB_l:
			ber = MIMO( model = model).run_pilot_ch_model( pilot_SNRdB = pilot_SNRdB, 
				param_NtNrNxSNRdB =(Nt, Nr, Nx, SNRdB), Nloop = Nloop, alpha = alpha, disp = True)
			BER_pilot.append( ber)
	else:
		"""
		LinearRegression is used.
		"""
		for SNRdB in SNRdB_l:
			ber = MIMO().run_pilot_channel( pilot_SNRdB = pilot_SNRdB, 
				param_NtNrNxSNRdB =(Nt, Nr, Nx, SNRdB), Nloop = Nloop, disp = True)
			BER_pilot.append( ber)

	# print( "List of average BERs =", BER_pilot)

	return BER_pilot

def pd_gen_4_snr_pilot(Method, BER_l, alpha = None, Npilot = 10, 
						sim_task = "Fixed SNRpilot", pilot_SNRdB = 7, 
						param_NtNrNx = (2,10,100), SNRdB_l = range(-5, 5, 5)):
	"""
	This is a generalized pd_gen() which can be used for both 
	fixed_snr_pilot() and snr_snr_pilot().
	"""

	pdi = pd.DataFrame() 
	pdi["Simulation task"] = [ sim_task] * len( BER_l)
	pdi["Method"] = [ Method] * len( BER_l)

	if type(pilot_SNRdB) is list:
		pdi["SNRpilot"] = pilot_SNRdB
	else:
		pdi["SNRpilot"] = [pilot_SNRdB] * len( BER_l)

	pdi["#pilots"] = [Npilot] * len( BER_l)
	pdi["Nt,Nr,Nx"] = [param_NtNrNx] * len( BER_l)
	if alpha is None:
		pdi["alpha"] = ["Not defined"] * len( BER_l)
	else:
		pdi["alpha"] = [alpha] * len( BER_l)
	pdi["SNR"] = SNRdB_l
	pdi["BER"] = BER_l
	return pdi

def fixed_snr_pilot( SNRdB_l = range(-5, 5, 1), param_NtNrNx = (2,10,100), pilot_SNRdB = 7, 
					alpha_l = [0.01, 0.1, 1, 10, 100], Nloop = 5000):
	"""
	Simulate BER for fixed SNRpilot cases
	the results will be saved to pandas dataframe.
	The basic parameters are given from the input argements.
	"""

	def pd_gen(Method, BER_l, alpha = None, Npilot = 10):
		"""
		This is a meta-function of pd_gen_4_snr_pilot()
		"""
		return pd_gen_4_snr_pilot( Method = Method, BER_l = BER_l, Npilot = Npilot, alpha = alpha, 
			sim_task = "Fixed SNRpilot", pilot_SNRdB = pilot_SNRdB, 
			param_NtNrNx = param_NtNrNx, SNRdB_l = SNRdB_l)		

	pdi_l = list()
	
	BER_l = get_BER( SNRdB_l, param_NtNrNx = param_NtNrNx, Nloop = Nloop, pilot_SNRdB = None)
	pdi_l.append( pd_gen( "Ideal, ZF Rx", BER_l))
	
	BER_l = get_BER_pilot_ch( SNRdB_l, param_NtNrNx = param_NtNrNx, Nloop = Nloop, pilot_SNRdB = pilot_SNRdB)
	pdi_l.append( pd_gen( r"Pilot, $\alpha$=0 (MMSE)", BER_l, alpha = 0))
	
	for alpha in alpha_l:
		BER_l = get_BER_pilot_ch( SNRdB_l, param_NtNrNx = param_NtNrNx, Nloop = Nloop, 
												pilot_SNRdB = pilot_SNRdB, alpha = alpha)
		pdi_l.append( pd_gen( r"Pilot, $\alpha$={}".format(alpha),BER_l, alpha))
		
	pdo = pd.concat( pdi_l, ignore_index = True)
	
	return pdo

def snr_snr_pilot( SNRdB_l = range(-5, 5, 1), param_NtNrNx = (2,10,100), 
					alpha_l = [0.01, 0.1, 1, 10, 100], Npilot = 15, Nloop = 5000):
	"""
	Simulate BER for fixed SNRpilot cases
	the results will be saved to pandas dataframe.
	The basic parameters are given from the input argements.
	"""

	def pd_gen(Method, BER_l, alpha = None):
		"""
		This is a meta-function of pd_gen_4_snr_pilot()
		"""
		return pd_gen_4_snr_pilot( Method = Method, BER_l = BER_l, alpha = alpha, 
			Npilot = Npilot, sim_task = "SNRpilot = SNR", pilot_SNRdB = SNRdB_l, 
			param_NtNrNx = param_NtNrNx, SNRdB_l = SNRdB_l)		

	pdi_l = list()	

	mlm = MIMO( Npilot = Npilot)
	print( "Ideal channel estimation without considering noise: ZF decoding with perfect H")
	BER_l = mlm.get_BER_pilot_ch_model_eqsnr( SNRdB_l, param_NtNrNx = param_NtNrNx, 
		Nloop = Nloop, pilot_ch = False)
	pdi_l.append( pd_gen( "Ideal, ZF Rx", BER_l))

	print( "General channel estimation: MMSE decoding with H and noise")
	BER_l = mlm.get_BER_pilot_ch_model_eqsnr( SNRdB_l, param_NtNrNx = param_NtNrNx, 
		Nloop = Nloop, pilot_ch = True)
	pdi_l.append( pd_gen( r"Pilot, $\alpha$=0 (MMSE)", BER_l, alpha = 0))	

	print( "Ridge channel estimation: MMSE decoding with H and noise")
	for alpha in alpha_l:
	    print( "Ridge with alpha =", alpha)
	    BER_l = mlm.get_BER_pilot_ch_model_eqsnr( SNRdB_l, param_NtNrNx = param_NtNrNx, 
	    	Nloop = Nloop, pilot_ch = True, alpha = alpha, model = "Ridge")
	    pdi_l.append( pd_gen( r"Pilot, $\alpha$={}".format(alpha),BER_l, alpha))

	pdo = pd.concat( pdi_l, ignore_index = True)
	
	return pdo
 
