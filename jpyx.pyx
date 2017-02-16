"""
jpyx.pyx
=========
Fab. 12, 'import jchem' is removed with an assoficated function.
To use calc_corr, please import jpyx_org.pyx after compiling it or
directly copy the function to the code. 

Now, jpyx.pyx can be used independently from rdkik or 
any other libaries except numpy.  
"""
import numpy as np
cimport numpy as np

import math
#import jchem

cdef extern from "jc.h":
    int prt()
    int prt_str( char*)
    unsigned int bin_sum( unsigned int, unsigned int)
    int sumup( int)

def prt_c():
    prt()

def prt_str_c( str):
    prt_str( str)

def sumup_c( int N):
    return sumup( N)

#=================================================

def type_checkable( int x):
    """
    This is cython code, which return x+1
    arg: int x
    """
    return x+1

#=================================================


# def calc_corr( smiles_l, radius = 6, nBits = 4096):
#   """
#   It emulate calc_corr in jchem using cython.
#   """
#   xM = jchem.get_xM( smiles_l, radius = radius, nBits = nBits)
#   A = calc_tm_sim_M( xM)

#   return A

def calc_bin_sim_M( np.ndarray[np.long_t, ndim=2] xM, gamma = 1.0):
    """
    Ex. A = calc_tm_sim_M( np.array( xM, dtype = long)
    """

    cdef int ln = xM.shape[0]
    cdef int lm = xM.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros( (ln,ln))
    cdef int ix, iy, ii
    cdef np.ndarray[np.long_t, ndim=1] a = np.zeros( ln, dtype = long)
    cdef int a_ix = 0
    cdef int c
    
    for ix in range( ln):
        a_ix = 0
        for ii in range( lm):
            a_ix += xM[ix, ii]
        #print ix, a_ix
        a[ix] = a_ix
    
    for ix in range( ln):
        for iy in range( ln):
            c = 0
            for ii in range( lm):
                c += xM[ix, ii] & xM[iy, ii]   
                
            if a[ix] == 0 and a[iy] == 0:
                A[ ix, iy] = 0.0
            else:
                A[ ix, iy] = gamma * float( c) / ( gamma * float( c) + a[ix] + a[iy] - 2*c)
            
    return A

def calc_tm_sim_M( np.ndarray[np.long_t, ndim=2] xM):
    """
    Ex. A = calc_tm_sim_M( np.array( xM, dtype = long)
    """

    cdef int ln = xM.shape[0]
    cdef int lm = xM.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros( (ln,ln))
    cdef int ix, iy, ii
    cdef np.ndarray[np.long_t, ndim=1] a = np.zeros( ln, dtype = long)
    cdef int a_ix = 0
    cdef int c, d
    
    for ix in range( ln):
        a_ix = 0
        for ii in range( lm):
            a_ix += xM[ix, ii]
        #print ix, a_ix
        a[ix] = a_ix
    
    for ix in range( ln):
        for iy in range( ln):
            c = 0
            for ii in range( lm):
                c += xM[ix, ii] & xM[iy, ii]   
                
            if a[ix] == 0 and a[iy] == 0:
                A[ ix, iy] = 0.0
            else:
                d = a[ix] + a[iy] - c
                A[ ix, iy] = float( c) / d
            
    return A

def calc_tm_sim_MM( np.ndarray[np.long_t, ndim=2] xM_data, np.ndarray[np.long_t, ndim=2] xM_db):
    """
    calc_tm_sim_MM : Calculate similarity between new and training binary vectors
    Ex. A = calc_tm_sim_M( np.array( xM, dtype = long), np.array( xM, dtype = long))
    """

    cdef int ln_data = xM_data.shape[0]
    cdef int ln_db = xM_db.shape[0]
    cdef int lm = xM_data.shape[1]  
    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros( (ln_data,ln_db))
    cdef int ix, iy, ii
    cdef np.ndarray[np.long_t, ndim=1] a_data = np.zeros( ln_data, dtype = long)
    cdef np.ndarray[np.long_t, ndim=1] a_db = np.zeros( ln_db, dtype = long)
    cdef int a_ix = 0
    cdef int c, d
    
    # a represents the number of on bits in a binary vector. 
    # Therefore, it should be divided to two variables such a_db and a_data
    for ix in range( ln_data):
        a_ix = 0
        for ii in range( lm):
            a_ix += xM_data[ix, ii]
        #print ix, a_ix
        a_data[ix] = a_ix

    for ix in range( ln_db):
        a_ix = 0
        for ii in range( lm):
            a_ix += xM_db[ix, ii]
        #print ix, a_ix
        a_db[ix] = a_ix

    
    # The maximum bound of ix is changed from ln to ln_data since
    # it is associated with the number of calculation molecules.
    # The maximum bound of iy is changed from ln to ln_data since
    # it is associated with the number of training molecules. 
    # lm is not changed since it should be the same for both new and training vectors. 
    # If lasso is used in regularization, the computation time for prediction can be
    # further reduced by the use of sparsity in the training molecules where
    # only part of training molecules will be used in the prediction process.  
    for ix in range( ln_data):
        for iy in range( ln_db):
            c = 0
            for ii in range( lm):
                c += xM_data[ix, ii] & xM_db[iy, ii]   
                
            if a_data[ix] == 0 and a_db[iy] == 0:
                A[ ix, iy] = 0.0
            else:
                d = a_data[ix] + a_db[iy] - c
                A[ ix, iy] = float( c) / d
            
    return A    

def calc_RBF( np.ndarray[np.long_t, ndim=2] xM, float epsilon):
    """
    calculate Radial basis function for xM with epsilon
    in terms of direct distance
    where distance is the number of different bits between the two vectors.
    """
    d = calc_atm_dist_M( xM)
    return RBF(d, epsilon) 

def calc_rRBF( np.ndarray[np.long_t, ndim=2] xM, float epsilon):
    """
    calculate Radial basis function for xM with epsilon
    in terms of relative distance
    where distance is the number of different bits between the two vectors.
    """
    rd = calc_atm_rdist_M( xM)
    return RBF(rd, epsilon) 


def RBF( d, e = 1):
    return np.exp( - e*np.power(d,2))

def calc_atm_dist_M( np.ndarray[np.long_t, ndim=2] xM):
    """
    Ex. A = calc_tm_sim_M( np.array( xM, dtype = long)
    """

    cdef int ln = xM.shape[0]
    cdef int lm = xM.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros( (ln,ln))
    cdef int ix, iy, ii
    cdef np.ndarray[np.long_t, ndim=1] a = np.zeros( ln, dtype = long)
    cdef int a_ix = 0
    cdef int c, d
    
    for ix in range( ln):
        a_ix = 0
        for ii in range( lm):
            a_ix += xM[ix, ii]
        #print ix, a_ix
        a[ix] = a_ix
    
    for ix in range( ln):
        for iy in range( ln):
            c = 0
            for ii in range( lm):
                c += xM[ix, ii] & xM[iy, ii]   
                
            if a[ix] == 0 and a[iy] == 0:
                A[ ix, iy] = 0.0
            else:
                d = a[ix] + a[iy] - 2*c
                A[ ix, iy] = d
            
    return A

def calc_atm_rdist_M( np.ndarray[np.long_t, ndim=2] xM):
    """
    Ex. A = calc_tm_sim_M( np.array( xM, dtype = long)
    Relative Tanimoto distance is calculated
    """

    cdef int ln = xM.shape[0]
    cdef int lm = xM.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros( (ln,ln))
    cdef int ix, iy, ii
    cdef np.ndarray[np.long_t, ndim=1] a = np.zeros( ln, dtype = long)
    cdef int a_ix = 0
    cdef int c, d
    
    for ix in range( ln):
        a_ix = 0
        for ii in range( lm):
            a_ix += xM[ix, ii]
        #print ix, a_ix
        a[ix] = a_ix
    
    for ix in range( ln):
        for iy in range( ln):
            c = 0
            for ii in range( lm):
                c += xM[ix, ii] & xM[iy, ii]   
                
            if a[ix] == 0 and a[iy] == 0:
                A[ ix, iy] = 0.0
            else:
                d = a[ix] + a[iy] - c # A or B
                A[ ix, iy] = (float) (d-c) / d # A or B - A and B / A or B (relative distance)
            
    return A



def calc_tm_dist_M( np.ndarray[np.long_t, ndim=2] xM):
    """
    Ex. A = calc_tm_sim_M( np.array( xM, dtype = long)
    """

    cdef int ln = xM.shape[0]
    cdef int lm = xM.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros( (ln,ln))
    cdef int ix, iy, ii
    cdef np.ndarray[np.long_t, ndim=1] a = np.zeros( ln, dtype = long)
    cdef int a_ix = 0
    cdef int c, d
    
    for ix in range( ln):
        a_ix = 0
        for ii in range( lm):
            a_ix += xM[ix, ii]
        #print ix, a_ix
        a[ix] = a_ix
    
    for ix in range( ln):
        for iy in range( ln):
            c = 0
            for ii in range( lm):
                c += xM[ix, ii] & xM[iy, ii]   
                
            if a[ix] == 0 and a[iy] == 0:
                A[ ix, iy] = 0.0
            else:
                d = a[ix] + a[iy] - c
                A[ ix, iy] = float( d - c) / d
            
    return A

def _calc_ec_sim_M_r0( np.ndarray[np.long_t, ndim=2] xM):
    """
    Euclidean-tanimoto distance
    """

    cdef int ln = xM.shape[0]
    cdef int lm = xM.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros( (ln,ln))
    cdef int ix, iy, ii
    cdef np.ndarray[np.long_t, ndim=1] a = np.zeros( ln, dtype = long)
    cdef int a_ix = 0
    cdef int c, d
    
    for ix in range( ln):
        a_ix = 0
        for ii in range( lm):
            a_ix += xM[ix, ii]
        #print ix, a_ix
        a[ix] = a_ix
    
    for ix in range( ln):
        for iy in range( ln):
            c = 0
            for ii in range( lm):
                c += xM[ix, ii] & xM[iy, ii]   
                
            if a[ix] == 0 and a[iy] == 0:
                A[ ix, iy] = 0.0
            else:
                d = a[ix] + a[iy] - c
                A[ ix, iy] = float(lm - d + c) / lm
            
    return A        

def _calc_ec_dist_M_r0( np.ndarray[np.long_t, ndim=2] xM):
    """
    Euclidean-tanimoto distance
    """

    cdef int ln = xM.shape[0]
    cdef int lm = xM.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros( (ln,ln))
    cdef int ix, iy, ii
    cdef np.ndarray[np.long_t, ndim=1] a = np.zeros( ln, dtype = long)
    cdef int a_ix = 0
    cdef int c, d
    
    for ix in range( ln):
        a_ix = 0
        for ii in range( lm):
            a_ix += xM[ix, ii]
        #print ix, a_ix
        a[ix] = a_ix
    
    for ix in range( ln):
        for iy in range( ln):
            c = 0
            for ii in range( lm):
                c += xM[ix, ii] & xM[iy, ii]   
                
            if a[ix] == 0 and a[iy] == 0:
                A[ ix, iy] = 0.0
            else:
                d = a[ix] + a[iy] - c
                A[ ix, iy] = float( d - c) / lm
            
    return A    

def calc_ec_sim_M( np.ndarray[np.long_t, ndim=2] xM):
    """
    Euclidean-tanimoto distance
    """

    cdef int ln = xM.shape[0]
    cdef int lm = xM.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros( (ln,ln), dtype = float)
    cdef int ix, iy, ii
    cdef np.ndarray[np.long_t, ndim=1] a = np.zeros( ln, dtype = long)
    cdef int a_ix = 0
    cdef int c
    
    for ix in range( ln):
        a_ix = 0
        for ii in range( lm):
            a_ix += xM[ix, ii]
        #print ix, a_ix
        a[ix] = a_ix
    
    for ix in range( ln):
        for iy in range( ln):
            c = 0
            for ii in range( lm):
                c += xM[ix, ii] ^ xM[iy, ii]   
                
            if a[ix] == 0 and a[iy] == 0:
                A[ ix, iy] = 1.0
            else:
                #d = a[ix] + a[iy] - c
                A[ ix, iy] = float( lm - c) / lm
            
    return A    

def calc_ec_dist_M( np.ndarray[np.long_t, ndim=2] xM):
    """
    Euclidean-tanimoto distance
    """

    cdef int ln = xM.shape[0]
    cdef int lm = xM.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros( (ln,ln))
    cdef int ix, iy, ii
    cdef np.ndarray[np.long_t, ndim=1] a = np.zeros( ln, dtype = long)
    cdef int a_ix = 0
    cdef int c
    
    for ix in range( ln):
        a_ix = 0
        for ii in range( lm):
            a_ix += xM[ix, ii]
        #print ix, a_ix
        a[ix] = a_ix
    
    for ix in range( ln):
        for iy in range( ln):
            c = 0
            for ii in range( lm):
                c += xM[ix, ii] ^ xM[iy, ii]   
                
            if a[ix] == 0 and a[iy] == 0:
                A[ ix, iy] = 0.0
            else:
                #d = a[ix] + a[iy] - c
                A[ ix, iy] = float( c) / lm
            
    return A    

def bcalc_tm_sim_vec(int a, int b, int ln):
    cdef int ii
    cdef int a_and_b = a & b
    cdef int a_or_b = a | b
    cdef int a_and_b_sum = 0
    cdef int a_or_b_sum = 0
    
    for ii in range(ln):
        a_and_b_sum += a_and_b & 1
        a_and_b = a_and_b >> 1
        a_or_b_sum += a_or_b & 1
        a_or_b = a_or_b >> 1
    return float(a_and_b_sum) / float(a_or_b_sum)

def calc_tm_sim_vec(np.ndarray[np.long_t, ndim=1] a, np.ndarray[np.long_t, ndim=1] b):
    cdef int ii
    cdef int a_and_b_sum = 0
    cdef int a_or_b_sum = 0
    cdef int ln = a.shape[0]
    
    for ii in range( ln):
        a_and_b_sum += a[ii] & b[ii]
        a_or_b_sum += a[ii] | b[ii]
    return float(a_and_b_sum) / float(a_or_b_sum)


"""
################################################################################
fast tm function is developed.  
################################################################################
"""
def fast_calc_tm_sim_M( np.ndarray[np.long_t, ndim=2] xM):
    """
    Ex. A = calc_tm_sim_M( np.array( xM, dtype = long))
    """

    cdef int ln = xM.shape[0]
    cdef int lm = xM.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros( (ln,ln))
    cdef int ix, iy, ii
    cdef np.ndarray[np.long_t, ndim=1] a = np.zeros( ln, dtype = long)
    cdef int a_ix = 0
    cdef int c, d
    
    for ix in range( ln):
        a_ix = 0
        for ii in range( lm):
            a_ix += xM[ix, ii]
        #print ix, a_ix
        a[ix] = a_ix
    
    for ix in range( ln):
        for iy in range( ln):
            c = 0
            for ii in range( lm):
                c += xM[ix, ii] & xM[iy, ii]   
                
            if a[ix] == 0 and a[iy] == 0:
                A[ ix, iy] = 0.0
            else:
                d = a[ix] + a[iy] - c
                A[ ix, iy] = float( c) / d
            
    return A

def bin_sum_pyx( int a, int b):
    """
    Sum of two binary integer variables.
    bin_sum( int a, int b)
    a: integer variable
    b: integer variable 
    return a + b 

    """
    c = a + b

    return c

def bin_sum_c( unsigned int a, unsigned int b):
    return bin_sum( a, b)


"""
################################################################################
Communication routines are developed since 12th Jan., 2016
################################################################################
"""

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

def mld_list_fast( r_list, mod_list = [-0.70710678, 0.70710678]):
    """
    maximum likelihood detection
    
    r_l: received signals after reception processing
    mod_l: list of all modulation signals
        BPSK: [-0.70710678, 0.70710678]
    
    return the demodulated signals (0, 1, ...)
    """

    assert type(r_list) == list
    assert type(mod_list) == list

    cdef N_r = len( r_list)
    cdef N_mod = len( mod_list)

    cdef np.ndarray[np.float64_t, ndim=1] m_l = np.array( mod_list)
    cdef np.ndarray[np.float64_t, ndim=1] r_l = np.array( r_list)
    cdef np.ndarray[np.float64_t, ndim=1] dist = np.zeros( N_mod, dtype = float)
    cdef np.ndarray[np.float64_t, ndim=1] sd_l = np.zeros( N_r, dtype = float)

    cdef int i_m 

    # sd_l.fill(0) # this is not needed since sd_l is initiated by zeros()
    for i_r in range( N_r):
        dist.fill( 0) #Store distance
        for i_m in range( N_mod):
            dist[ i_m] = np.power( np.abs( r_l[ i_r] - m_l[ i_m]), 2)
        sd_l[ i_r] = np.argmin( dist)
    
    return sd_l
    
def _mld_fast_r0( np.ndarray[np.float64_t, ndim=1] r_a, 
            np.ndarray[np.float64_t, ndim=1] m_a):
    """
    maximum likelihood detection
    
    r_l: received signals after reception processing
    mod_l: list of all modulation signals
        BPSK: [-0.70710678, 0.70710678]
    
    return the demodulated signals (0, 1, ...)
    """

    assert type(r_a) == np.ndarray
    assert type(m_a) == np.ndarray

    cdef int N_r = r_a.shape[0]
    cdef int N_mod = m_a.shape[0]

    #cdef np.ndarray[np.float64_t, ndim=1] m_l = np.array( mod_list)
    #cdef np.ndarray[np.float64_t, ndim=1] r_l = np.array( r_list)
    cdef np.ndarray[np.float64_t, ndim=1] sd_a = np.zeros( N_r, dtype = float)
    cdef np.ndarray[np.float64_t, ndim=1] dist_a = np.zeros( N_mod, dtype = float)

    cdef int i_m
    cdef int i_r 

    # sd_l.fill(0) # this is not needed since sd_l is initiated by zeros()
    for i_r in range( N_r):
        dist_a.fill( 0) #Store distance
        for i_m in range( N_mod):
            # dist_a[ i_m] = np.power( np.abs( r_a[ i_r] - m_a[ i_m]), 2)            
            dist_a[ i_m] = (r_a[ i_r] - m_a[ i_m]) * (r_a[ i_r] - m_a[ i_m])
        sd_a[ i_r] = np.argmin( dist_a)
        
    return sd_a

def mld_fast( np.ndarray[np.float64_t, ndim=1] r_a, 
            np.ndarray[np.float64_t, ndim=1] m_a):
    """
    maximum likelihood detection
    
    r_l: received signals after reception processing
    mod_l: list of all modulation signals
        BPSK: [-0.70710678, 0.70710678]
    
    return the demodulated signals (0, 1, ...)
    """

    assert type(r_a) == np.ndarray
    assert type(m_a) == np.ndarray

    cdef int N_r = r_a.shape[0]
    cdef int N_mod = m_a.shape[0]

    #cdef np.ndarray[np.float64_t, ndim=1] m_l = np.array( mod_list)
    #cdef np.ndarray[np.float64_t, ndim=1] r_l = np.array( r_list)
    cdef np.ndarray[np.float64_t, ndim=1] sd_a = np.zeros( N_r, dtype = float)
    cdef np.ndarray[np.float64_t, ndim=1] dist_a = np.zeros( N_mod, dtype = float)

    cdef int i_m
    cdef int i_r 
    cdef int argmin_d
    cdef float min_d, d

    # sd_l.fill(0) # this is not needed since sd_l is initiated by zeros()
    for i_r in range( N_r):
        #dist_a.fill( 0) #Store distance        
        argmin_d = 0
        min_d = (r_a[ i_r] - m_a[ 0]) * (r_a[ i_r] - m_a[ 0])
        for i_m in range( 1, N_mod):
            # dist_a[ i_m] = np.power( np.abs( r_a[ i_r] - m_a[ i_m]), 2)            
            # dist_a[ i_m] = (r_a[ i_r] - m_a[ i_m]) * (r_a[ i_r] - m_a[ i_m])
            d = (r_a[ i_r] - m_a[ i_m]) * (r_a[ i_r] - m_a[ i_m])
            if d < min_d:
                argmin_d = i_m
                min_d = d               
        sd_a[ i_r] = argmin_d
        
    return sd_a


# ==================================================================
# jamespyx_ext/mgh.recon
# ==================================================================
def update_recon_pyx(
        Recon1,  # For complex, direct use is the best approach
        np.ndarray[np.int_t, ndim=2] support):
    cdef float err1 = 1.0
    cdef np.ndarray[np.float_t, ndim=2] Constraint = np.ones(Recon1.shape)
    cdef int R1y, R1x
    cdef np.ndarray[np.float_t, ndim=2] Recon1_abs = np.abs(Recon1)
    cdef np.ndarray[np.float_t, ndim=2] Recon1_pwr2 = np.power(Recon1_abs, 2)
    
    R1y, R1x = Recon1.shape
    
    for p in range(R1y):
        for q in range(R1x):
            if support[p, q] == 1:
                Constraint[p, q] = Recon1_abs[p, q]
                err1 += Recon1_pwr2[p, q]
            if Recon1_abs[p, q] > 1:
                Constraint[p, q] = 1
                
    Recon1_update = Constraint * np.exp(1j * np.angle(Recon1))

    return Recon1_update, err1
