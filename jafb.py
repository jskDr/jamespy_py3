"""
my library for aquaos flow battery
"""
import pandas as pd
import numpy as np
from rdkit import Chem

class AQDS():
	"""
	data flow is controlled outside, while
	flags are constrolled inside by self variables.
	"""

	def __init__(self, csv_fname = 'oh_subs_csv.csv'):
		"read data file"
		self.dfr = pd.read_csv( csv_fname)

	def get_dfr( self):
		"""
		dfr can be accessed either directly or indirectly:
		dfr = self.dfr
		dfr = self.get_dfr()
		"""
		return self.dfr

	def add_dfr( self, alist, name = 'smiles'):
		self.dfr[ name] = alist

		return self.dfr

	def trans_x_binary(self, dfr):
		#Input is generated. It is 36 x 6 matrix
		#The character should be changed as H -> 0, OH -> 1
		ri_vec = [1, 3, 4, 5, 6, 8]
		R = []
		HOH201 = {'H': 0, 'OH': 1}

		for ri in ri_vec:
			s = 'R{}'.format( ri)
			rv = dfr[s].tolist()
			rv_01 = map( lambda x: HOH201[x], rv)
			R.append( rv_01)
		RM = np.mat( R).T

		return RM

	def trans_x_frag6(self, dfr):
		Frag6_L = []   
		OHSYMB = {'H': '', 'OH': '(O)'}
		ri_vec = [1, 3, 4, 5, 6, 8]
		for ri in ri_vec:
			s = 'R{}'.format( ri)
			rv = dfr[s].tolist()
			fr_01 = map( lambda x: OHSYMB[x], rv)
			Frag6_L.append( fr_01)

		Frag6_D = []
		for ii in range( len(Frag6_L[0])):
			Frag6_D.append({})

		for ii, frag in enumerate(Frag6_L):
			for ix, fr in enumerate(frag):
				dict_key = '{B%d}' % ii
				Frag6_D[ix][dict_key] = fr

		return Frag6_D

	def trans_y_EG( self, dfr):
		yE = np.mat( dfr['E(V)'].tolist()).T
		yG = np.mat( dfr['G(kJ/Mol)'].tolist()).T

		return yE, yG

	def trans_smiles( self, Frag6_D):
		mol_smiles_list = []
		base_smiles = 'C1(=O)c2c{B3}c{B4}c(S(=O)(=O)O)c{B5}c2C(=O)c2c{B0}c(S(=O)(=O)O)c{B1}c{B2}c21'
		for ix, mol_symb in enumerate(Frag6_D):
			mol = bq14_oh2 = Chem.MolFromSmiles( base_smiles, replacements=mol_symb)
			mol_smiles = Chem.MolToSmiles( mol)
			mol_smiles_list.append( mol_smiles)

		return mol_smiles_list