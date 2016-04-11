from IPython.display import display

def pd_find_SMILES( pdr, s, disp = 'False', smiles_id = 'SMILES'):
	pdw = pdr[ pdr[ smiles_id] == s]
	if disp:
		display( pdw)

	return pdw
