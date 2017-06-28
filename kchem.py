# Python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# This is James Sungjin Kim's library
# import jutil
import kutil

def gff_vec( smiles_vec, rad = 2, nBits = 1024):
	"It generates a fingerprint vector from a smiles code vector"
	return [gff(x, rad, nBits) for x in smiles_vec]

def gfb_vec( smiles_vec, rad = 4, nBits = 1024):
	"It generates a fingerprint vector from a smiles code vector"
	return [gfb(x, rad, nBits) for x in smiles_vec]

def gff_binlist( smiles_vec, rad = 2, nBits = 1024):
	"""
	It generates a binary list of fingerprint vector from a smiles code vector.
	Each string will be expanded to be the size of nBits such as 1024.
	- It shows error message when nBits < 1024 and len(x) > nBits.	
	- Now bits reduced to match input value of nBit eventhough the real output is large
	"""
	ff_vec = gff_vec( smiles_vec, rad, nBits)
	ff_bin = [ bin(int(x.ToBinary().encode("hex"), 16)) for x in ff_vec]

	#Show error message when nBits < 1024 and len(x) > nBits	
	"""
	for x in ff_bin:
		if len(x[2:]) > nBits:
			print 'The length of x is {0}, which is larger than {1}'.format(len(x[2:]), nBits)
			print 'So, the minimal value of nBits must be 1024 generally.'
	return [ map( int, list( '0'*(nBits - len(x[2:])) + x[2:])) for x in ff_bin]
	"""
	return [ list(map( int, list( kutil.sleast(x[2:], nBits)))) for x in ff_bin]

def gfb_binlist( smiles_vec, rad = 4, nBits = 1024):
	"""
	It generates a binary list of fingerprint vector from a smiles code vector.
	Each string will be expanded to be the size of nBits such as 1024.
	- It shows error message when nBits < 1024 and len(x) > nBits.	
	- Now bits reduced to match input value of nBit eventhough the real output is large
	- fp clean will be adopted.
	"""
	ff_vec = gfb_vec( smiles_vec, rad, nBits)
	ff_bin = [ bin(int(x.ToBinary().encode("hex"), 16)) for x in ff_vec]

	#Show error message when nBits < 1024 and len(x) > nBits	
	"""
	for x in ff_bin:
		if len(x[2:]) > nBits:
			print 'The length of x is {0}, which is larger than {1}'.format(len(x[2:]), nBits)
			print 'So, the minimal value of nBits must be 1024 generally.'
	return [ map( int, list( '0'*(nBits - len(x[2:])) + x[2:])) for x in ff_bin]
	"""
	return [ list(map( int, list( kutil.sleast(x[2:], nBits)))) for x in ff_bin]


def gfp_binlist( smiles_vec, rad = 4, nBits = 1024):
	gff_binlist( smiles_vec, rad = rad, nBits = nBits)

def gff_binlist_bnbp( smiles_vec, rad = 2, nBits = 1024, bnbp = 'bn'):
	"""
	It generates a binary list of fingerprint vector from a smiles code vector.
	Each string will be expanded to be the size of nBits such as 1024.
	- It shows error message when nBits < 1024 and len(x) > nBits.	
	- Now bits reduced to match input value of nBit eventhough the real output is large
	bnbp --> if binary input, bnbp = 'bn', else if bipolar input, bnbp = 'bp'
	"""
	ff_vec = gff_vec( smiles_vec, rad, nBits)
	ff_bin = [ bin(int(x.ToBinary().encode("hex"), 16)) for x in ff_vec]

	if bnbp == 'bp': #bipolar input generation
		return [ list(map( kutil.int_bp, list( kutil.sleast(x[2:], nBits)))) for x in ff_bin]
	else:
		return [ list(map( int, list( kutil.sleast(x[2:], nBits)))) for x in ff_bin]

def gff_M( smiles_vec, rad = 2, nBits = 1024):
	"It generated a binary matrix from a smiles code vecor."
	return np.mat(gff_binlist( smiles_vec, rad = rad, nBits = nBits))

def gfp_M( smiles_vec, rad = 4, nBits = 1024):
	"It generated a binary matrix from a smiles code vecor."
	xM = np.mat(gfb_binlist( smiles_vec, rad = rad, nBits = nBits))	
	#Now fingerprint matrix is cleaned if column is all the same value such as all 1, all 0
	return clean_fp_M( xM)

def gff_M_bnbp( smiles_vec, rad = 2, nBits = 1024, bnbp = 'bn'):
	"It generated a binary matrix from a smiles code vecor."
	return np.mat(gff_binlist_bnbp( smiles_vec, rad, nBits, bnbp))

def calc_tm_dist_int( A_int, B_int):
	"""
	Calculate tanimoto distance of A_int and B_int
	where X_int isinteger fingerprint vlaue of material A.
	"""
	C_int = A_int & B_int
	
	A_str = bin(A_int)[2:]
	B_str = bin(B_int)[2:]
	C_str = bin(C_int)[2:]

	lmax = max( [len( A_str), len( B_str), len( C_str)])

	""" this shows calculation process 
	print "A:", A_str.ljust( lmax, '0')
	print "B:", B_str.ljust( lmax, '0')
	print "C:", C_str.ljust( lmax, '0')
	"""

	a = A_str.count('1')
	b = B_str.count('1')
	c = C_str.count('1')

	# print a, b, c
	if a == 0 and b == 0:
		tm_dist = 1
	else:
		tm_dist = float(c) / float( a + b - c)

	return tm_dist

def calc_tm_dist( A_smiles, B_smiles):

	A_int = ff_int( A_smiles)
	B_int = ff_int( B_smiles)

	return calc_tm_dist_int( A_int, B_int)

def getw( Xs, Ys, N = 57, nBits = 400):
	"It calculate weight vector for specific N and nNBits."

	Xs50 = Xs[:N]
	Ys50 = Ys[:N]

	X = gff_M( Xs50, nBits=400)
	y = np.mat( Ys50).T

	print(X.shape)

	# Xw = y is assumed for Mutiple linear regression
	w = np.linalg.pinv( X) * y
	#print w

	plt.plot( w)
	plt.show()

	return w

def getw_clean( Xs, Ys, N = None, rad = 2, nBits = 1024):
	"Take only 50, each of which has safe smile code."
	nXs, nYs = clean_smiles_vec_io( Xs, Ys)

	# print len(nXs), len(nYs)

	if N is None:
		N = len( nXs)

	X = gff_M( nXs[:N], rad = rad, nBits = nBits)
	y = np.mat( nYs[:N]).T

	w = np.linalg.pinv( X) * y

	plt.plot( w)
	plt.title('Weight Vector')
	plt.show()

	y_calc = X*w
	e = y - y_calc
	se = (e.T * e)
	mse = (e.T * e) / len(e)
	print("SE =", se)
	print("MSE =", mse)
	print("RMSE =", np.sqrt( mse))
	
	plt.plot(e)
	plt.title("Error Vector: y - y_{calc}")
	plt.show()

	plt.plot(y, label='original')
	plt.plot(y_calc, label='predicted')
	plt.legend()
	plt.title("Output values: org vs. pred")
	plt.show()

	return w

def getw_clean_bnbp( Xs, Ys, N = None, rad = 2, nBits = 1024, bnbp = 'bn'):
	"""
	Take only 50, each of which has safe smile code.
	Translate the input into bipolar values.
	"""
	nXs, nYs = clean_smiles_vec_io( Xs, Ys)

	# print len(nXs), len(nYs)

	if N is None:
		N = len( nXs)

	X = gff_M_bnbp( nXs[:N], rad = rad, nBits = nBits, bnbp = bnbp)
	y = np.mat( nYs[:N]).T

	w = np.linalg.pinv( X) * y

	plt.plot( w)
	plt.title('Weight Vector')
	plt.show()

	y_calc = X*w
	e = y - y_calc
	se = (e.T * e)
	mse = (e.T * e) / len(e)
	print("SE =", se)
	print("MSE =", mse)
	print("RMSE =", np.sqrt( mse))
	
	plt.plot(e)
	plt.title("Error Vector: y - y_{calc}")
	plt.show()

	plt.plot(y, label='original')
	plt.plot(y_calc, label='predicted')
	plt.legend()
	plt.title("Output values: org vs. pred")
	plt.show()

	return w

def fpM_pat( xM):
	#%matplotlib qt
	xM_sum = np.sum( xM, axis = 0)

	plt.plot( xM_sum)
	plt.xlabel('fingerprint bit')
	plt.ylabel('Aggreation number')
	plt.show()

def gen_input_files( A, yV, fname_common = 'ann'):
	"""
	Input files of ann_in.data and ann_run.dat are gerneated.
	ann_in.data and ann_run.data are training and testing data, respectively
	where ann_run.data does not includes output values.
	The files can be used in ann_aq.c (./ann_aq) 
	* Input: A is a matrix, yV is a vector with numpy.mat form.
	"""
	# in file
	no_of_set = A.shape[0]
	no_of_input = A.shape[1]
	const_no_of_output = 1 # Now, only 1 output is considerd.
	with open("{}_in.data".format( fname_common), "w") as f:
		f.write( "%d %d %d\n" % (no_of_set, no_of_input, const_no_of_output))
		for ix in range( no_of_set):
			for iy in range( no_of_input):
				f.write( "{} ".format(A[ix,iy]))
			f.write( "\n{}\n".format( yV[ix,0]))
		print(("{}_in.data is saved for trainig.".format( fname_common)))

	# run file 
	with open("{}_run.data".format( fname_common), "w") as f:
		#In 2015-4-9, the following line is modified since it should not be 
		#the same to the associated line in ann_in data but it does not include the output length. 
		f.write( "%d %d\n" % (no_of_set, no_of_input))
		for ix in range( no_of_set):
			for iy in range( no_of_input):
				f.write( "{} ".format(A[ix,iy]))
			f.write( "\n") 
		print(("{}_run.data is saved for testing.".format( fname_common)))

def gen_input_files_valid( At, yt, Av):
	"""
	Validation is also considerd.
	At and yt are for training while Av, yv are for validation.
	Input files of ann_in.data and ann_run.dat are gerneated.
	The files are used in ann_aq.c (./ann_aq) 
	* Input: At, Av is matrix, yt, yv is vector
	"""

	const_no_of_output = 1 # Now, only 1 output is considerd.

	# in file
	no_of_set = At.shape[0]
	no_of_input = At.shape[1]
	with open("ann_in.data", "w") as f:
		f.write( "%d %d %d\n" % (no_of_set, no_of_input, const_no_of_output))
		for ix in range( no_of_set):
			for iy in range( no_of_input):
				f.write( "{} ".format(At[ix,iy]))
			f.write( "\n{}\n".format( yt[ix,0]))
		print(("ann_in.data with {0} sets, {1} inputs is saved".format( no_of_set, no_of_input)))

	# run file 
	no_of_set = Av.shape[0]
	no_of_input = Av.shape[1]
	with open("ann_run.data", "w") as f:
		f.write( "%d %d\n" % (no_of_set, no_of_input))
		for ix in range( no_of_set):
			for iy in range( no_of_input):
				f.write( "{} ".format(Av[ix,iy]))
			f.write( "\n") 
		print(("ann_run.data with {0} sets, {1} inputs is saved".format( no_of_set, no_of_input)))


def get_valid_mode_output( aV, yV, rate = 3, more_train = True, center = None):
	"""
	Data is organized for validation. The part of them becomes training and the other becomes validation.
	The flag of 'more_train' represents tranin data is bigger than validation data, and vice versa.
	"""
	ix = list(range( len( yV)))
	if center == None:
		center = int(rate/2)
	if more_train:
		ix_t = [x for x in ix if x%rate != center]
		ix_v = [x for x in ix if x%rate == center]
	else:
		ix_t = [x for x in ix if x%rate == center]
		ix_v = [x for x in ix if x%rate != center]

	aM_t, yV_t = aV[ix_t, 0], yV[ix_t, 0]
	aM_v, yV_v = aV[ix_v, 0], yV[ix_v, 0]

	return aM_t, yV_t, aM_v, yV_v	

def get_valid_mode_data( aM, yV, rate = 3, more_train = True, center = None):
	"""
	Data is organized for validation. The part of them becomes training and the other becomes validation.
	The flag of 'more_train' represents tranin data is bigger than validation data, and vice versa.
	"""
	ix = list(range( len( yV)))
	if center == None:
		center = int(rate/2)
	if more_train:
		ix_t = [x for x in ix if x%rate != center]
		ix_v = [x for x in ix if x%rate == center]
	else:
		ix_t = [x for x in ix if x%rate == center]
		ix_v = [x for x in ix if x%rate != center]

	aM_t, yV_t = aM[ix_t, :], yV[ix_t, 0]
	aM_v, yV_v = aM[ix_v, :], yV[ix_v, 0]

	return aM_t, yV_t, aM_v, yV_v	

def _estimate_accuracy_r0( yv, yv_ann, disp = False):
	"""
	The two column matrix is compared in this function and 
	It calculates RMSE and r_sqr.
	"""
	e = yv - yv_ann
	se = e.T * e
	aae = np.average( np.abs( e))
	RMSE = np.sqrt( se / len(e))

	# print "RMSE =", RMSE
	y_unbias = yv - np.mean( yv)
	s_y_unbias = y_unbias.T * y_unbias
	r_sqr = 1.0 - se/s_y_unbias

	if disp:
		print("r_sqr = {0:.3e}, RMSE = {1:.3e}, AAE = {2:.3e}".format( r_sqr[0,0], RMSE[0,0], aae))

	return r_sqr[0,0], RMSE[0,0]

def estimate_accuracy( yv, yv_ann, disp = False):
	"""
	The two column matrix is compared in this function and 
	It calculates RMSE and r_sqr.
	"""

	print(yv.shape, yv_ann.shape)
	if not( yv.shape[0] > 0 and yv.shape[1] == 1 and yv.shape == yv_ann.shape):
		raise TypeError( 'Both input data matrices must be column vectors.')

	e = yv - yv_ann
	se = e.T * e
	aae = np.average( np.abs( e))
	RMSE = np.sqrt( se / len(e))

	# print "RMSE =", RMSE
	y_unbias = yv - np.mean( yv)
	s_y_unbias = y_unbias.T * y_unbias
	r_sqr = 1.0 - se/s_y_unbias

	if disp:
		print("r_sqr = {0:.3e}, RMSE = {1:.3e}, AAE = {2:.3e}".format( r_sqr[0,0], RMSE[0,0], aae))

		#print "len(e) = ", len(e)
		#print "se = ", se
		#print "s_y_unbias =", s_y_unbias

	return r_sqr[0,0], RMSE[0,0]

def estimate_accuracy3( yv, yv_ann, disp = False):
	"""
	The two column matrix is compared in this function and 
	It calculates RMSE and r_sqr.
	"""

	print(yv.shape, yv_ann.shape)
	if not( yv.shape[0] > 0 and yv.shape[1] == 1 and yv.shape == yv_ann.shape):
		raise TypeError( 'Both input data matrices must be column vectors.')

	e = yv - yv_ann
	se = e.T * e
	aae = np.average( np.abs( e))
	RMSE = np.sqrt( se / len(e))

	# print "RMSE =", RMSE
	y_unbias = yv - np.mean( yv)
	s_y_unbias = y_unbias.T * y_unbias
	r_sqr = 1.0 - se/s_y_unbias

	if disp:
		print("r_sqr = {0:.3e}, RMSE = {1:.3e}, AAE = {2:.3e}".format( r_sqr[0,0], RMSE[0,0], aae))

		#print "len(e) = ", len(e)
		#print "se = ", se
		#print "s_y_unbias =", s_y_unbias

	return r_sqr[0,0], RMSE[0,0], aae	

def to1D( A):
	"""
	Regardless of a type of A is array or matrix, 
	to1D() return 1D numpy array.	
	"""
	return np.array(A).flatten()


def estimate_score3( yv, yv_ann, disp = False):
	"""
	The two column matrix is compared in this function and 
	It calculates RMSE and r_sqr.
	"""
	yv = to1D( yv)
	yv_ann = to1D( yv_ann)

	if disp:
		print("The shape values of yv and yv_ann are", yv.shape, yv_ann.shape)
		
	if not( yv.shape[0] > 0 and yv.shape[0] == yv_ann.shape[0]):
		raise TypeError("The length of the input vectors should be equal and more than zero.")

	e = yv - yv_ann
	MAE = np.average( np.abs( e))
	RMSE = np.sqrt( np.average( np.power( e, 2)))
	r_sqr = 1.0 - np.average( np.power( e, 2)) / np.average( np.power( yv - np.mean( yv), 2))

	if disp:
		print("r_sqr = {0:.3e}, RMSE = {1:.3e}, MAE = {2:.3e}".format( r_sqr, RMSE, MAE))

	return r_sqr, RMSE, MAE

def clean_fp_M_bias( xM):
	iy_list = []
	xM_sum = np.sum( xM, 0)
	for iy in range( xM.shape[1]):
		if xM_sum[0,iy] == 0 or xM_sum[0,iy] == xM.shape[0]:
			#print 'deleted: ', iy
			iy_list.append( iy)	
	
	xM = np.delete(xM, iy_list, 1)
	
	return xM

def clean_fp_M_pattern( xM):
	iy_list = []
	for iy in range( xM.shape[1]):
		if iy not in iy_list:
			pat = xM[:, iy]
			# print pat
			for iy2 in range( iy+1, xM.shape[1]):
				if iy2 not in iy_list:
					if not np.any( pat - xM[:, iy2]):
						iy_list.append( iy2)			

	#print iy_list
	xM = np.delete(xM, iy_list, 1)
	
	return xM

def clean_fp_M( xM):
	"""
	1. Zero sum column vectors will be removed.
	2. All one column vectors wiil be also removed.
	3. The same patterns for different position will be merged to one.
	* np.all() and np.any() should be understand clearly.
	"""
	
	xM = clean_fp_M_bias( xM)
	xM = clean_fp_M_pattern( xM)	
	
	return xM

def check_fp_M_row_pattern( xM):
	"""
	If the pattern in row is the same, 
	it will give the number of the same pattern rows.
	"""
	ix_list = []
	ix_pair_list = []
	for ix in range( xM.shape[0]):
		if ix not in ix_list:
			pat = xM[ix, :]
			# print pat
			for ix2 in range( ix+1, xM.shape[0]):
				if ix2 not in ix_list:
					if not np.any( pat - xM[ix2, :]):
						ix_list.append( ix2)
						ix_pair_list.append( (ix, ix2))

	#if len( ix_list):
	#	print 'The same row pair list is', ix_list
		
	return ix_pair_list


def gfpM_c( smiles_vec, rad = 4, nBits = 1024):
	
	xM = gfpM( smiles_vec, rad = rad, nBits = nBits)

	return clean_fp_M( xM)

def list_indices( l, target):
	return [i for i,val in enumerate(l) if val == target]


def pd_check_mol2smiles( pd_smiles):

	smiles_l = pd_smiles.tolist()
	fail_l = check_mol2smiles( smiles_l)
	
	# since siles_l is changed, pd values are also changed.
	pd_smiles = smiles_l

	return fail_l


def check_dup_list( x_list):
	"""
	Duplication indices are returned.
	"""
	# print 'Duplication if false', x_smiles == set( x_smiles)
	x_list_dup_count = np.array( [ x_list.count( x) for x in x_list])
	
	return np.where( x_list_dup_count > 1)


def get_duplist( x_list, disp = True):
	"""
	Duplication indices are returned.
	"""
	duplist = []
	for x in set( x_list):
		if x_list.count( x) > 1:
			duplist.append( list_indices( x_list, x))

	if disp:
		print(duplist)
		for d in duplist:
			print([x_list[x] for x in d])

	return duplist

def pd_remove_no_mol2smiles( pdr, smiles_id = 'SMILES'):
	"""
	Find not working smiles codes
	"""
	s = pdr[ smiles_id].tolist()
	fail_list = get_mol2smiles( s)

	pdr = kutil.pd_remove_faillist_ID( pdr, fail_list)

	return pdr

def add_new_descriptor( xM, desc_list):
	xMT_l = xM.T.tolist()
	#print np.shape(xMT_l)

	xMT_l.append( desc_list)
	#print np.shape(xMT_l)

	xM_add = np.mat( xMT_l).T
	print(xM_add.shape)

	return xM_add

def get_xM_molw( s_l):
	
	molw_l = rdkit_molwt( s_l)

	return np.mat( molw_l).T

def get_xM_lasa( s_l):
	
	molw_l = rdkit_LabuteASA( s_l)

	return np.mat( molw_l).T

def get_xM_ensemble( s_l, ds_l = ['molw', 'lsas']):
	xM_l = list()
	for ds in ds_l:
		xM_l.append( eval( 'get_xM_{}( s_l)'.format( ds)))

	return np.concatenate( xM_l, axis = 1)


def matches( s_l, p, disp = False):
	"""
	Find matches in list of molecules.
	c_l, r_l = matches( s_l, p, ...)
	where c_l is the number of matching points and r_l is a index of matching positions.
	"""
	c_l = list()
	r_l = list()

	r_l = [ matches_each(s, p, disp) for s in s_l]
	c_l = list(map( len, r_l))

	return c_l, r_l

def estimate_accuracy4(yEv, yEv_calc, disp = False):
	r_sqr = metrics.r2_score( yEv, yEv_calc)
	RMSE = np.sqrt( metrics.mean_squared_error( yEv, yEv_calc))
	MAE = metrics.mean_absolute_error( yEv, yEv_calc)
	DAE = metrics.median_absolute_error( yEv, yEv_calc)

	if disp:
		print("r^2={0:.2e}, RMSE={1:.2e}, MAE={2:.2e}, DAE={3:.2e}".format( r_sqr, RMSE, MAE, DAE))

	return r_sqr, RMSE, MAE, DAE
