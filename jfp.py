"""
jfp.py is updated in order to match tap character.
"""
import numpy as np
import jchem
import jutil

def find_cluster( fa_list, thr = 0.5):
	"""
	find similar pattern with 
	the first element: fa0
	"""
	fa0 = fa_list[0]
	fa0_group = [fa0]
	fa_other = fa_list[1:]
	for fa_o in fa_list[1:]:
		tm_d = jchem.calc_tm_dist_int( fa0, fa_o)
		if tm_d > thr:
			fa0_group.append( fa_o)
			fa_other.remove( fa_o)

	return fa0_group, fa_other

def find_cluster_all( fa_list, thr = 0.5):
	"""
	all cluster are founded based on threshold of 
	fingerprint similarity using greedy methods
	"""
	fa_o = fa_list
	fa0_g_all = []

	while len( fa_o) > 0:
		fa0_g, fa_o = find_cluster( fa_o, thr)
		fa0_g_all.append( fa0_g)

	return fa0_g_all

def _calc_tm_sim_int_r0( A_int, B_int):
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


def calc_tm_sim_int( A_int, B_int):
	"""
	Calculate tanimoto distance of A_int and B_int
	where X_int isinteger fingerprint vlaue of material A.
	"""
	A_str_org = bin(A_int)
	B_str_org = bin(B_int)

	ln_max = max( map(len, [A_str_org, B_str_org]))

	A_str_ext = jutil.sleast( A_str_org, ln_max)
	B_str_ext = jutil.sleast( B_str_org, ln_max)

	A_int_ext = int( A_str_ext, 2)
	B_int_ext = int( B_str_ext, 2)
	C_int_ext = A_int_ext & B_int_ext
	
	C_str_ext = bin( C_int_ext)

	A_str = A_str_ext[2:]
	B_str = B_str_ext[2:]
	C_str = C_str_ext[2:]

	a = A_str.count('1')
	b = B_str.count('1')
	c = C_str.count('1')

	# print a, b, c
	if a == 0 and b == 0:
		tm_dist = 1
	else:
		tm_dist = float(c) / float( a + b - c)

	return tm_dist	


def calc_tm_sim( A_smiles, B_smiles):

	A_int = jchem.ff_int( A_smiles)
	B_int = jchem.ff_int( B_smiles)

	return calc_tm_sim_int( A_int, B_int)	

def _calc_tm_sim_V_r0( x1, x2):
	a = np.sum( x1)
	b = np.sum( x2)
	c = np.shape(np.argwhere( x1+x2 == 2))[0]

	if a == 0 and b == 0:
		return 1
	else: 
		return float(c) / (a+b-c)

def calc_tm_sim_V( x1, x2):
	a = np.sum( x1)
	b = np.sum( x2)
	c = np.sum( np.logical_and(x1, x2))

	if a == 0 and b == 0:
		return 1
	else: 
		return float(c) / (a+b-c)


def calc_tm_sim_M( xM):

	A = np.zeros( (xM.shape[0], xM.shape[0]))
	
	for ix in range( xM.shape[0]):
		for iy in range( xM.shape[0]):
			A[ ix, iy] = calc_tm_sim_V( xM[ix, :], xM[iy, :])

	return A

def calc_tm_sim_M_pyx( xM):
	"""
	Ex. A = calc_tm_sim_M( np.array( xM, dtype = long)
	"""

	ln = xM.shape[0]
	lm = xM.shape[1]
	A = np.zeros( (ln,ln))
	#ix, iy, ii
	a = np.zeros( ln, dtype = long)
	a_ix = 0
	#c
	
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
				A[ ix, iy] = float( c) / (a[ix] + a[iy] - c)
			
	return A

def get_babel( fname, disp = True):
	"""
	Open babel output file such as aqds.fp_int
	and read fp values.
	"""

	fp_list = []
	with open( fname, 'r') as f:
		ss = f.readlines()

	fp = []
	fp_count = 6
	for s in ss:
		if s[0] == '>' or s[0] == 'P':
			if disp: print s
		else:
			fp.append( s)
			fp_count -= 1
			if fp_count == 0:
				fp_list.append( babel_transform( fp))
				fp = []
				fp_count = 6

	return fp_list

def babel_transform( fp):
	"""
	The separated values are merged into one string and 
	then, it is transformed to 1024 binary integer list.
	"""

	ff_m1 = fp[0][:-2]
	for f in fp[1:]:
		ff_m1 += ' ' + f[:-2]

	ff_m2 = ''.join( ff_m1.split())

	ff_m3 = '0x' + ff_m2

	ff_m4 = int( ff_m2, 16)

	ff_m5 = bin( ff_m4)

	ff_m6 = ff_m5[2:].zfill( 1024)

	ff_m7 = map( int, ff_m6)

	return ff_m7

def hexlist2binlist( fp, fpl = 4096):
	"""
	fp is list of hex strings
	output is the matrix with binary bit arrays
	"""
	fp_1 = []
	for x in fp:
		x_1 = bin(int(x, 16))
		fp_1.append( x_1[2:].rjust( fpl, '0'))
		
	# fp_1 = [bin(int(x, 16)[2:].rjust( fpl, '0') ) for x in fp]
	fp_2 = [map( int, x) for x in fp_1]		
	
	return fp_2

def binlist2hex( bl, fpl_hex = 1024):

	bl_1 = map( str, bl)
	bl_2 = ''.join( bl_1)
	bl_3 = int( bl_2, 2)
	bl_4 = hex( bl_3)
	bl_5 = bl_4[2:-1].rjust( fpl_hex, '0')
	
	return bl_5
	
	
def get_chemfp_rdkit_eg( fname = 'aqds_morgan.fpt', fpl = 4096, disp = True):
	"""
	Read rdkit2fps data file for special cases
	Later, it will be extended for general cases. 
	"""		
	
	fp = []
	with open( fname, 'r') as f:
		ss = f.readlines()

	for s in ss:
		if s[0] == '#':
			if disp: 
				print s
		else:
			s2 = s.split()
			fp.append( s2[0])
			if disp:
				print "Molecule: ", s2[1]
	
	fp_bl = hexlist2binlist( fp, fpl = fpl)
		
	return np.mat( fp_bl)
	