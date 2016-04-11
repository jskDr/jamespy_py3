"""
This is python code compatible with cython code.
I will write this code in order to check the speed update of cython codes.
"""
import numpy as np


"""
part I. testing codes for cython
"""
def jsum(N):
	ss = 0.0;
	for ii in range(N):
		for jj in range(N):
			for kk in range(N):
				ss = ss + 0.1*ii*jj*kk

	return ss

def jsum_float_full( C, A, B):
	C = A + B

def f_test(A):
	A[0] = 1

def jsum_float( C, A, B, N):
	#C = A + B and C[ii] = A[ii] + B[ii] 
	#are different. The former generates new C. 
	
	for jj in range( N):
		for ii in range(C.shape[0]):
			C[ii] = A[ii] + B[ii]

"""
part II. Useful codes for cython
"""
