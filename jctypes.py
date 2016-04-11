"""
This program calls c functions directly using ctypes.
"""

import numpy as np
import ctypes

jclib = ctypes.cdll.LoadLibrary("./jctypes_c.so")

def square_inplace( arr):
	arr_n = len( arr)
	arr_ptr = arr.ctypes.data_as( ctypes.POINTER( ctypes.c_double))

	jclib.square( arr_ptr, arr_n)

	return arr

def square( arr_org):
	arr = arr_org.copy()
	arr_n = len( arr)
	arr_ptr = arr.ctypes.data_as( ctypes.POINTER( ctypes.c_double))

	jclib.square( arr_ptr, arr_n)

	return arr

print "I. square_inplace() is called."
arr = np.arange( 5, dtype = np.double)
print "The square of", arr, "is"
arr2 = square_inplace( arr)
print arr2
print "Then, arr becomes", arr

print "II. square() is called."
arr = np.arange( 5, dtype = np.double)
print "The square of", arr, "is"
arr2 = square( arr)
print arr2
print "Then, arr becomes", arr
