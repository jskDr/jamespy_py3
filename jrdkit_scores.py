# python 3.0

from rdkit import Chem
import jchem

from SA_Score import sascorer
from NP_Score import npscorer

class SANP():
	"""
	NP: Natural likeness point
	SA: Sythetic accesbility
	"""
	def __init__(self, disp = False, graph = False):
		self.fscore = npscorer.readNPModel()
		self.disp = disp
		self.graph = graph
		
	def get_np( self, s):
		fscore = self.fscore
		m = Chem.MolFromSmiles( s)
		np = npscorer.scoreMol(m,fscore)

		if self.graph: jchem.show_mol( s)
		if self.disp: print('NP Score is', np)
		return np

	def get_sa( self, s):
		m = Chem.MolFromSmiles( s)
		sa = sascorer.calculateScore( m)

		if self.graph: jchem.show_mol( s)
		if self.disp: print('NP Score is', sa)        
		return sa