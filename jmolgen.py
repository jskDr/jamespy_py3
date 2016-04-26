import pandas as pd
import numpy as np
import re #regular expression
import itertools

import matplotlib.pyplot as plt
from mpltools import color

from collections import OrderedDict

import jmath
from poodle import frame

"""
ox_s_base = "[H]C1(C)[N+](CCCS([O-])(=O)=O)=C2C(=C{0}C(=O)C{1}=C2{2})C1(C)C"
rd_s_base = "[H]C1(C)N(CCCS([O-])(=O)=O)C2=C{2}C{1}=C(O)C{0}=C2C1(C)C"

"""


def get_r_list( N_Rgroup = 4, so3h = '(S(O)(=O)=O)', disp = False, pdForm = True):
	pdr_id, pdr_index, pdr_rgroups, pdr_no_r = [], [], [], []
	
	N_max_bin = '0b' + '1' * N_Rgroup
	for pos in range( int(N_max_bin, 2) + 1):
		pos_bin = bin( pos)[2:].rjust( N_Rgroup, '0')
		so_int_l = [int(x) for x in pos_bin]
		so_l = [so3h if x == 1 else '' for x in so_int_l ]
		no_r = sum( so_int_l)
		
		pdr_id.append( pos + 1)
		pdr_no_r.append( no_r)
		pdr_index.append( so_int_l)
		pdr_rgroups.append( so_l)
		
		if disp: print(pos, no_r, so_int_l, '==>', so_l)
		
	if pdForm:
		pdr = pd.DataFrame()
		pdr['ID'] = pdr_id
		pdr['Rgroup'] = [so3h] * len( pdr_id)   
		pdr['NoOfR'] = pdr_no_r
		pdr['Index'] = pdr_index
		pdr['Rgroups'] = pdr_rgroups
		return pdr
	else:
		return so_l

def get_multi_r_list( N_positions = 4, r_l = ['', '(S(O)(=O)=O)', '(O)'], disp = False, pdForm = True):
	"""
	Multiple R-groups will be attached. 
	The '' attachment should be involved in the list of R-groups since 
	it is also one of the possible case.
	"""
	pdr_id, pdr_index, pdr_rgroups, pdr_no_r = [], [], [], []
	
	# The number of the possible elements in product operation is length of R-groups    
	# N_positions reprensents the number of attachment positions. 
	Nr = len( r_l)
	so_int_l_all = itertools.product( list(range( Nr)), repeat = N_positions)
	for pos, so_int_l in enumerate(so_int_l_all):
		so_l = [ r_l[x] for x in so_int_l]
		no_r = jmath.count( so_int_l, 0, inverse = True)

		pdr_id.append( pos + 1)
		pdr_no_r.append( no_r)
		pdr_index.append( so_int_l)
		pdr_rgroups.append( so_l)

		if disp: print(pos, no_r, so_int_l, '==>', so_l)

	if pdForm:
		pdr = pd.DataFrame()
		pdr['ID'] = pdr_id
		if len( r_l) == 2 and '' in r_l:
			"""
			If r_l consists of one R-group and blank, 
			R-group becomes the one R-group. 
			The empty position can be 0 or 1, which is support for 
			generalization although it usually located in 0. 
			"""
			if r_l.index( '') == 0:
				pdr['Rgroup'] = [ r_l[1]] * len( pdr_id)   
			else:
				pdr['Rgroup'] = [ r_l[0]] * len( pdr_id)   
		else:
			pdr['Rgroup'] = ['Mix'] * len( pdr_id)   
		pdr['NoOfR'] = pdr_no_r
		pdr['Index'] = pdr_index
		pdr['Rgroups'] = pdr_rgroups
		return pdr
	else:
		return so_l  

def gen_r_attach( mol = 'Oc1nc(O)c2nc3c{0}c{1}c{2}c{3}c3nc2n1', so3h = '(S(O)(=O)=O)', disp = False):
	"""
	generate molecules with R group fragment
	"""
	N_group = len( re.findall( '{[0-9]*}', mol)) # find number of R group positions

	pdr = get_r_list( N_group, so3h, disp = disp, pdForm = True)
	so_l = pdr['Rgroups'].tolist()

	aso_l = []
	for so in so_l:        
		aso = mol.format(*so)
		aso_l.append( aso)
		if disp: print(so, aso)

	pdr['SMILES'] = aso_l
	pdr['BaseMol'] = [aso_l[0]] * len( aso_l)
	pdr['BaseStr'] = [mol] * len( aso_l)

	return pdr

def gen_rl_attach( mol = 'Oc1nc(O)c2nc3c{0}c{1}c{2}c{3}c3nc2n1', r_l = ['', '(S(O)(=O)=O)'], disp = False):
	"""
	generate molecules with R group fragment
	get_r_list becomes get_multi_r_list so as to generate molecules with multiple R-group attached.
	"""
	N_group = len( re.findall( '{[0-9]*}', mol)) # find number of R group positions

	pdr = get_multi_r_list( N_group, r_l, disp = disp, pdForm = True)
	so_l = pdr['Rgroups'].tolist()

	aso_l = []
	for so in so_l:        
		aso = mol.format(*so)
		aso_l.append( aso)
		if disp: print(so, aso)

	pdr['SMILES'] = aso_l
	pdr['BaseMol'] = [aso_l[0]] * len( aso_l)
	pdr['BaseStr'] = [mol] * len( aso_l)

	return pdr  

def gen_rl_2attach( mol, mol_nH, r_l = ['', '(S(O)(=O)=O)'], disp = False):
	"""
	generate molecules with R group fragment
	get_r_list becomes get_multi_r_list so as to generate molecules with multiple R-group attached.
	Reduced (or hydrated) SMILES strings will be generated as well. 
	"""
	N_group = len( re.findall( '{[0-9]*}', mol)) # find number of R group positions

	pdr = get_multi_r_list( N_group, r_l, disp = disp, pdForm = True)
	so_l = pdr['Rgroups'].tolist()

	aso_l = []
	aso_nH_l = []
	for so in so_l:        
		aso = mol.format(*so)
		aso_l.append( aso)

		aso_nH = mol_nH.format(*so)
		aso_nH_l.append( aso_nH)

		if disp: print(so, aso, aso_nH)

	# Storing canonical smiles strings
	#pdr['SMILES'] = jchem.csmiles_l( aso_l)
	pdr['SMILES'] = aso_l
	#pdr['R-SMILES'] = jchem.csmiles_l( aso_nH_l)
	pdr['R-SMILES'] = aso_nH_l
	pdr['BaseMol'] = [aso_l[0]] * len( aso_l)
	pdr['BaseStr'] = [mol] * len( aso_l)
	pdr['BaseR-Mol'] = [aso_nH_l[0]] * len( aso_nH_l)
	pdr['BaseR-Str'] = [mol_nH] * len( aso_nH_l)

	return pdr  

def get_R_od( Nitro = False):
	if Nitro == True:
		R_od = OrderedDict( [
				('no group', '[H]'), 
				('amine', 'N'),
				('hydroxyl', 'O'), 
				('methyl', 'C'),
				('fluoro', 'F'),
				('phosphonic acid', 'P(O)(O)(=O)'),
				('sulfonic acid', 'S(=O)(=O)(O)'), 
				('carboxylic acid', 'C(O)(=O)'),
				('nitro', '[N+]([O-])(=O)')])
	else:
		R_od = OrderedDict( [
				('no group', '[H]'), 
				('amine', 'N'),
				('hydroxyl', 'O'), 
				('methyl', 'C'),
				('fluoro', 'F'),
				('phosphonic acid', 'P(O)(O)(=O)'),
				('sulfonic acid', 'S(=O)(=O)(O)'), 
				('carboxylic acid', 'C(O)(=O)')])

	return R_od

class InfoFrame( pd.DataFrame):
	def __init__(self, info_d = {}, **kwargs):
		"""
		DataFrame + Dict
		Parameters
		----------
		The parameters of Pandas DataFrame are used 
		*args : any type
			 all arguments without a keyword
		**kwargs: any type
			 all arguments without a keyword
		"""
		super().__init__(**kwargs)
		self.info_d = info_d

class MolGen_IF( frame.InfoFrame):
	def __init__(self, Base_d, RGroup_l = ['[H]', 'S(=O)(=O)(O)'], 
			data=None, index=None, columns=None, dtype=None, copy=False):
		"""
		Input example
		-----
		Base_d, RGroup_l = ['[H]', 'S(=O)(=O)(O)']

		Parameters
		-----
		Base_d can be a dict of Ox and Rd or a scalr string
		RGroup_l can be a list of strings or a scalr string.
		"""
		info_d = dict()
		info_d["Base_d"] = Base_d
		info_d["RGroup_l"] = RGroup_l

		super().__init__( info_d=info_d, 
			data=data, index=index, columns=columns, dtype=dtype, copy=copy)
	
	def enum_r( self):        
		# Input variables are restored.
		Base = self.info_d["Base_d"]
		RGroup = self.info_d["RGroup_l"]
		assert type(Base) is not dict and type(RGroup) is not list
		
		tmp_df = gen_r_attach( Base, RGroup)    
			
		# The constant information are saved to dictionary.
		self.info_d["BaseSMILES"] = tmp_df["BaseMol"][0]
		
		# The result sequences are stored to self dataframe.
		self["NoOfR"]  = tmp_df["NoOfR"]
		self["RIndex"] = tmp_df["Index"]
		self["RGroups"]= tmp_df["Rgroups"]
		self["SMILES"] = tmp_df["SMILES"]
		
	def enum_rl( self):        
		# Input variables are restored.
		Base = self.info_d["Base_d"]
		RGroup_l = self.info_d["RGroup_l"]
		assert type(Base) is not dict and type(RGroup_l) is list
		
		tmp_df = gen_rl_attach( Base, RGroup_l)    
			
		# The constant information are saved to dictionary.
		self.info_d["BaseSMILES"] = tmp_df["BaseMol"][0]
		
		# The result sequences are stored to self dataframe.
		self["NoOfR"]  = tmp_df["NoOfR"]
		self["RIndex"] = tmp_df["Index"]
		self["RGroups"]= tmp_df["Rgroups"]
		self["SMILES"] = tmp_df["SMILES"]

	def enum_rl_redox( self):
		# Input variables are restored.
		Base_d = self.info_d["Base_d"] # {'rd': str, 'ox': str}
		RGroup_l = self.info_d["RGroup_l"]
		assert type(Base_d) is dict and type(RGroup_l) is list
		assert Base_d['ox'] and Base_d['rd']
		# print( Base_d['ox'], Base_d['rd'])
		
		tmp_df = gen_rl_2attach( Base_d["ox"], Base_d["rd"], RGroup_l)    
			
		# The constant information are saved to dictionary.
		self.info_d["BaseSMILES"] = tmp_df["BaseMol"][0]
		self.info_d["BaseR-SMILES"] = tmp_df["BaseR-Mol"][0]    
		
		# The result sequences are stored to self dataframe.
		self["NoOfR"]  = tmp_df["NoOfR"]
		self["RIndex"] = tmp_df["Index"]
		self["RGroups"]= tmp_df["Rgroups"]
		self["SMILES"] = tmp_df["SMILES"]
		self["R-SMILES"] = tmp_df["R-SMILES"]

class MolGen_DF(pd.DataFrame):
	def __init__(self, Base_d, RGroup_l = ['[H]', 'S(=O)(=O)(O)']):
		"""
		Input
		-----
		Base_d can be a dict of Ox and Rd or a scalr string
		RGroup_l can be a list of strings or a scalr string.
		"""
		super().__init__()
		self.__info_d = dict()
		self.__info_d["Base_d"] = Base_d
		self.__info_d["RGroup_l"] = RGroup_l
		
	def info( self):
		return self.__info_d
	
	def enum_r( self):        
		# Input variables are restored.
		Base = self.__info_d["Base_d"]
		RGroup = self.__info_d["RGroup_l"]
		assert type(Base) is not dict and type(RGroup) is not list
		
		tmp_df = gen_r_attach( Base, RGroup)    
			
		# The constant information are saved to dictionary.
		self.__info_d["BaseSMILES"] = tmp_df["BaseMol"][0]
		
		# The result sequences are stored to self dataframe.
		self["NoOfR"]  = tmp_df["NoOfR"]
		self["RIndex"] = tmp_df["Index"]
		self["RGroups"]= tmp_df["Rgroups"]
		self["SMILES"] = tmp_df["SMILES"]
		
	def enum_rl( self):        
		# Input variables are restored.
		Base = self.__info_d["Base_d"]
		RGroup_l = self.__info_d["RGroup_l"]
		assert type(Base) is not dict and type(RGroup_l) is list
		
		tmp_df = gen_rl_attach( Base, RGroup_l)    
			
		# The constant information are saved to dictionary.
		self.__info_d["BaseSMILES"] = tmp_df["BaseMol"][0]
		
		# The result sequences are stored to self dataframe.
		self["NoOfR"]  = tmp_df["NoOfR"]
		self["RIndex"] = tmp_df["Index"]
		self["RGroups"]= tmp_df["Rgroups"]
		self["SMILES"] = tmp_df["SMILES"]

	def enum_rl_redox( self):
		# Input variables are restored.
		Base_d = self.__info_d["Base_d"] # {'rd': str, 'ox': str}
		RGroup_l = self.__info_d["RGroup_l"]
		assert type(Base_d) is dict and type(RGroup_l) is list
		assert Base_d['ox'] and Base_d['rd']
		# print( Base_d['ox'], Base_d['rd'])
		
		tmp_df = gen_rl_2attach( Base_d["ox"], Base_d["rd"], RGroup_l)    
			
		# The constant information are saved to dictionary.
		self.__info_d["BaseSMILES"] = tmp_df["BaseMol"][0]
		self.__info_d["BaseR-SMILES"] = tmp_df["BaseR-Mol"][0]    
		
		# The result sequences are stored to self dataframe.
		self["NoOfR"]  = tmp_df["NoOfR"]
		self["RIndex"] = tmp_df["Index"]
		self["RGroups"]= tmp_df["Rgroups"]
		self["SMILES"] = tmp_df["SMILES"]
		self["R-SMILES"] = tmp_df["R-SMILES"]