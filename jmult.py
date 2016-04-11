"""
jmult
multiprocessing 
"""
from multiprocessing import Process
from multiprocessing import Pool

import numpy as np

def corr_fp( xM):
	lx = xM.shape[0]
	AC_X = np.zeros( [lx, lx])
	for l1 in xrange(lx):
		for l2 in xrange(lx):
			AC_X[l1, l2] = xM[l1, :] * xM[l2, :].T

	return AC_X

def corr_fp_inner( AC_X_inner, xM, l1, lx):
	"""
	Inner part of correlation is perforemd. 
	"""
	
	for l2 in xrange(lx):
		AC_X_inner[l2] = xM[l1, :] * xM[l2, :].T


def fast_corr_fp( xM):
	lx = xM.shape[0]
	AC_X = np.zeros( [lx, lx])
	for l1 in xrange(lx):
		# the first argument is reference pointer array
		corr_fp_inner( AC_X[l1,:], xM, l1, lx)
		#for l2 in xrange(lx):
			#AC_X[l1, l2] = xM[l1, :] * xM[l2, :].T

	return AC_X

def _fast_corr_fp_r1_fail( xM):
	lx = xM.shape[0]
	AC_X = np.zeros( [lx, lx])
	p = []
	for l1 in xrange(lx):
		p.append( Process( target = corr_fp_inner, args = ( AC_X[l1,:], xM, l1, lx,)))
		p[-1].start()

	for l1 in xrange(lx):
		p[l1].join()
		print sum( AC_X[l1,:])

	return AC_X


def map_corr_fp( xM):
	"""
	map based correlation
	"""
	lx = xM.shape[0]
	AC_X_list = []
	for l1 in xrange(lx):
		in_list = []
		for l2 in xrange(lx):
			in_list.append( [xM[l1, :], xM[l2, :]])
			# AC_X[l1, l2] = xM[l1, :] * xM[l2, :].T
		out_list = map( lambda x: (x[0] * x[1].T)[0,0], in_list)
		AC_X_list.append( out_list)

	#print np.shape( AC_X_list)
	AC_X = np.mat( AC_X_list)

	return AC_X

def pmap_f(x):
	return (x[0] * x[1].T)[0,0]

def pmap_corr_fp( xM):
	"""
	map based correlation
	"""
	p = Pool(10)

	lx = xM.shape[0]
	AC_X_list = []
	for l1 in xrange(lx):
		in_list = []
		for l2 in xrange(lx):
			in_list.append( [xM[l1, :], xM[l2, :]])
		out_list = p.map( pmap_f, in_list)
		AC_X_list.append( out_list)

	#print np.shape( AC_X_list)
	AC_X = np.mat( AC_X_list)

	return AC_X
