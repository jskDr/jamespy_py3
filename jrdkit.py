"""
RDKit related files are included.
"""
from jchem import *
from chemspipy import ChemSpider
cs_global = ChemSpider('9fc1f087-d349-49ef-bdfb-f725fd6b81b5')

def find( sm, sg, disp = False, graph = False):
	m = Chem.MolFromSmiles( sm)
	results = m.GetSubstructMatches( Chem.MolFromSmarts( sg))

	if graph:
		show_mol( sm)

	if disp: 
		print('Results:', results)
		print('Canonical SMILES:', Chem.MolToSmiles( m))

	return results


def prt_sg( m, sg = '[C]', info = ''):
	matched_sg_l = m.GetSubstructMatches( Chem.MolFromSmarts( sg))
	print(sg, ':', len(matched_sg_l), '-->', matched_sg_l, info)

	return len(matched_sg_l)

def find_subgroups( sm = 'CNN', info = ''):
	"""
	return subgroups for various functional groups
	"""

	# sm = 'CNN' # Acetamide --> 1.58
	show_mol( sm)
	print('SMILES: {},'.format(sm), info)

	m = Chem.MolFromSmiles( sm)

	prt_sg( m, '[a]')
	prt_sg( m, '[A]')
	prt_sg( m, '[#6]')
	prt_sg( m, '[R1]')
	prt_sg( m, '[R2]')
	prt_sg( m, '[r5]')
	prt_sg( m, '[v4]')
	prt_sg( m, '[v3]')
	prt_sg( m, '[X2]')
	prt_sg( m, '[H]')
	prt_sg( m, '[H1]')
	prt_sg( m, '*')
	prt_sg( m, 'CC')
	prt_sg( m, '[#6]~[#6]') # any bond
	prt_sg( m, '[#6]@[#6]') # ring bond
	#prt_sg( m, 'F/?[#6]=C/[Cl]')
	prt_sg( m, 'F/[#6]=C/Cl')
	prt_sg( m, '[!c]')
	prt_sg( m, '[N,#8]')
	prt_sg( m, '[#7,C&+0,+1]')
	print('============================================')    


	sgn_d = dict()
	prt_sg( m, '[C]',   info = 'no subgroup')
	print('')

	prt_sg( m, '[CH3]',   info = 'no subgroup')
	sgn_d[1] = prt_sg( m,  '[*CH3]',  info = '#01, sol += -1.7081') 
	print('')

	prt_sg( m, '[CH2]',   info = 'no subgroup') 
	sgn_d[2] = prt_sg( m,  '[*CH2*]', info = '#02, sol += -0.4991')
	sgn_d[5] = prt_sg( m, '*=[CH2]', info = '#02, sol += -0.4991')
	print('')
	
	prt_sg( m, '[CH1]',   info = 'no subgroup') 
	sgn_d[3] = prt_sg( m,  '*[CH1]*(*)', info = '#02, sol += -0.4991')
	sgn_d[6] = prt_sg( m, '*=[CH1]*', info = '#02, sol += -0.4991')
	print('')

	prt_sg( m, '[CH0]',   info = 'no subgroup') 
	prt_sg( m, '*[CH0]',   info = 'no subgroup')    
	prt_sg( m, '[*CH0]',   info = 'no subgroup')    
	prt_sg( m, '[*CH0*](*)',   info = 'no subgroup')    
	prt_sg( m, '*=[CH0*](*)',   info = 'no subgroup')   
	sgn_d[4] = prt_sg( m,  '*[CH0]*(*)(*)', info = '#02, sol += -0.4991')
	sgn_d[7] = prt_sg( m, '*=[CH0]*(*)', info = '#02, sol += -0.4991')
	sgn_d[8] = prt_sg( m, '*=[CH0]=*', info = '#02, sol += -0.4991')
	sgn_d[9] = prt_sg( m,  '*[CH0]#[CH1]', info = '#02, sol += -0.4991')
	print('')
	
	sgn_d[8] = prt_sg( m, '[NH2]', info = '#49, sol += 0') 
	sgn_d[8] = prt_sg( m, '[NH]',  info = '#50, sol += 0') 
	sgn_d[8] = prt_sg( m, '[OH]') #26 (primary) 0.2711

def get_atoms( s_org='Oc1ccccc1', sub_s = 'ccO', detail = False, number = False):
	print("Original SMILES :", s_org)
	s = Chem.MolToSmiles( Chem.MolFromSmiles( s_org))
	print("Canonical SMILES:", s)
	print("Chemspider")
	for result in cs_global.search( s):
		print(result, result.common_name, result.csid)

	show_mol(s)
	m = Chem.MolFromSmiles(s)
	for ii, atom in enumerate(m.GetAtoms()):
		if number: print( ii, ':', sep = '', end = ' ')
		print(atom.GetAtomicNum(), atom.GetSymbol(), atom.GetSmarts(), end=' ')
		for nb in atom.GetNeighbors():
			nb_idx = nb.GetIdx()
			print( m.GetBondBetweenAtoms(ii,nb_idx).GetBondType(), nb_idx, end=', ')
		print()

	print("===========================")

	if detail:
		print('------------')
		print("Atom status")
		print('------------')   
		for atom in m.GetAtoms():
			ii = atom.GetIdx()
			print("Atom", ii, ":", atom.GetAtomicNum(), atom.GetSymbol(), 
				atom.GetSmarts(), atom.GetAtomicNum())

			print("Neighbor atom indices and link bond types:", end = ' ')
			for nb in atom.GetNeighbors():
				nb_idx = nb.GetIdx()
				print( m.GetBondBetweenAtoms(ii,nb_idx).GetBondType(), nb_idx, end=', ')
			print()

			print("atom.IsInRing:", atom.IsInRing())
			print("atom.IsInRingSize(3):", atom.IsInRingSize(3))
			print("atom.IsInRingSize(4):", atom.IsInRingSize(4))
			print("atom.IsInRingSize(5):", atom.IsInRingSize(5))
			print("atom.IsInRingSize(6):", atom.IsInRingSize(6))
			for bond in atom.GetBonds():
				print(bond.GetBondType())
		
		print() 
		print('------------')
		print("Bond status")
		print('------------')   
		for bond in m.GetBonds():
			print(bond.GetBondType())

		print()
		print('------------')
		print("Ring status")
		print('------------')
		print('Chem.GetSSSR:', Chem.GetSSSR( m))    

		print()
		print('------------')
		print("Substring matching - Smart")
		print('------------')
		print('Substring SMILES:', sub_s)
		patt = Chem.MolFromSmarts( sub_s)
		print("Sets of matching indices:", m.GetSubstructMatches( patt))

def get_atoms_mol(m, sm = Chem.MolFromSmarts('ccO'), detail = False, number = False):
	s_org = Chem.MolToSmiles( m)
	sub_s = Chem.MolToSmarts( sm)
	return get_atoms( s_org, sub_s = sub_s, detail=detail, number=number)

def get_AQ_sd():
	"""
	Return SMILES strings of AQ family members. 
	"""
	AQ_sd = {
	"AQ": 'O=C1c2ccccc2C(=O)c2ccccc12',
	"AQS": 'OS(=O)(=O)c1ccc2C(=O)c3ccccc3C(=O)c2c1',
	"AQDS": 'OS(=O)(=O)c1ccc2C(=O)c3ccc(cc3C(=O)c2c1)S(O)(=O)=O'}

	return AQ_sd

def get_AQ_md():
	"""
	return mols of AQ family members
	"""
	AQ_md = {}
	AQ_sd = get_AQ_sd()
	for k in AQ_sd:
		AQ_md[ k] = Chem.MolFromSmiles( AQ_sd[k])

	return AQ_md

def generate_CSMILES( s_l, disp = False):
	cs_l = []
	for i, s in enumerate(s_l):
		if disp: print( i, s)
		cs = Chem.MolToSmiles(Chem.MolFromSmiles( s))
		cs_l.append( cs)
	return cs_l