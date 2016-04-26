# Python3
import numpy as np

def long_to_int64_array( val, ln):
    sz = ln / 64 + 1
    ar = np.zeros( sz, dtype=int)
    i64 = 2**64 - 1
    for ii in range( sz):
        ar[ ii] = int(val & i64)
        val = val >> 64
    return ar

def int64_array_ro_long( ar):
    val = long(0)
    for ii in range( ar.shape[0]):
        val = val | ar[-ii-1]
        print( val)
        if ii < ar.shape[0] - 1:
            val = val << 64
        print( val)
    return val

def count( a_l, a, inverse = False):
    """
    It returns the number of elements which are equal to 
    the target value. 
    In order to resolve when x is an array with more than
    one dimensions, converstion from array to list is used. 
    """
    if inverse == False:
        x = np.where( np.array( a_l) == a)
    else: 
        x = np.where( np.array( a_l) != a)

    return len(x[0].tolist())