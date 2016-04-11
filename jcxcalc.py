"""
This file includes a function to predict pH dependent solubility
using the original file and cxcalc result file.
Since the cxcalc result file does not include SMILES string information, 
two files should be merged for final reporting. 
Now, I will use R-SMILES strings which represent molecules of oxidized form. 

1. Merge two files
2. Predict intrinsic solubility using the merged file 
3. Calculate pH dependent solubility
"""

# These are python libraries
import pandas as pd
import numpy as np

# These are my libraries
import jpandas as jpd

"""
def get_logSpH( pdw, pdr_logPlogD):
	logD_all_list = list()
	for pH in range(15):
		logD = pdr_logPlogD['pH={}.00'.format(pH)].values
		logD_all_list.append( logD)
	logD_all = np.array( logD_all_list).T
	#print logD_all.shape

	MaxClogD = np.max( logD_all, axis = 1)
	#print MaxClogD .shape

	pdw['logDmax'] = MaxClogD 
	pdw['Delta'] = pdr_logPlogD['logP'].values - MaxClogD
	pdw['logSmin'] = pdw['PlogS'] + pdw['Delta'] 
	
	for pH in range(15):
		#logD = pdr_logPlogD['pH={}.00'.format(0)].values
		pdw['PlogS({})'.format( pH)] = pdw['PlogS'] + pdr_logPlogD['logP'] - pdr_logPlogD['pH={}.00'.format(pH)]
	
	return pdw

def gen_save_pdw( fname, pdr_Flavin256, PlogS, pdr_logPlogD):
	pdw = pdr_Flavin256.copy()
	pdw['PlogS'] = PlogS
	get_logSpH( pdw, pdr_logPlogD)
	pdw.to_csv( fname, index = False)
	return pdw

gen_save_pdw( 'sheet/cxcalc/Alloxazine2_64_PlogSpH.csv', pdr_enum, pdr_PlogS.Solubility, pdr_logPlogD)
"""

def get_logSpH( pdr):
	"""
	From logS (intrinsic log solubility and logP/logD(pH)),
	this function calculates logS(pH).
	"""
	# The final results will be stored into pwd together with the original information. 
	pdw = pdr.copy()
	
	#It extracts logD data into a numpy array.
	logD_all_list = list()
	for pH in range(15):
		logD = pdw['pH={}.00'.format(pH)].values
		logD_all_list.append( logD)
	logD_all = np.array( logD_all_list).T
	#print logD_all.shape

	MaxClogD = np.max( logD_all, axis = 1)
	#print MaxClogD .shape

	pdw['logDmax'] = MaxClogD 
	pdw['Delta'] = pdw['logP'].values - MaxClogD
	pdw['logSmin'] = pdw['PlogS'] + pdw['Delta'] 
	
	for pH in range(15):
		#logD = pdr_logPlogD['pH={}.00'.format(0)].values
		pdw['PlogS({})'.format( pH)] = pdw['PlogS'] + pdw['logP'] - pdw['pH={}.00'.format(pH)]
	
	return pdw


def pd_merge_molgen_cxcalc(fname_molgen = 'sheet/alloxazine/Alloxazine2_64_molgen.csv', 
					fname_cxcalc = 'sheet/alloxazine/Alloxazine2_64_RSMILES_cxcalc.csv',
					ID_name = 'id', 
					smiles_id = 'SMILES'):
	"""
	Two files will be merged. The cxcalc file data items are separated by tap. So, 
	it should be noticed in read_csv(). 
	"""
	pdr_molgen = pd.read_csv( fname_molgen)

	# A special column is generated. The column is aggregation of the two columns as follows:
	R_Index = list()
	R_map = {'(O)': 'OH', '(S(O)(=O)=O)': 'SO3H', '(OC)': 'OCH3', '(C)': 'CH3'}
	for a, b in zip( pdr_molgen['Rgroup'], pdr_molgen['Index']):

		R_Index.append( R_map[a] + b)
	pdr_molgen['R-Index'] = R_Index

	pdr_cxcalc = pd.read_csv( fname_cxcalc, sep = '\t')
	pdr_merge = pd.merge( pdr_molgen, pdr_cxcalc, on = 'id')
	if smiles_id != 'SMILES':
		pdr_merge[ 'SMILES'] = pdr_merge[ smiles_id]

	return pdr_merge

def get_solpred_path():
	return "/home/jamessungjinkim/Dropbox/Aspuru-Guzik/030 Labs/solpred_lab/"

def predict_logS( fname_merge):
	# logS prediction part
	solpred_path = get_solpred_path()
	jmbr = jpd.PD_MBR_Solubility( fname_model = solpred_path + 'sheet/wang1708_smiles.pkl', 
								mode = 'offline', 
								fname_db= solpred_path + 'sheet/wang1708_smiles.csv', 
								y_id = 'exp')
	jmbr.predict_fname( fname_merge)

	return jmbr.PDR_SOL


def predict_logSpH( fname = None,
					fname_molgen = 'sheet/alloxazine/Alloxazine2_64_RSMILES_molgen.csv', 
					fname_cxcalc = 'sheet/alloxazine/Alloxazine2_64_RSMILES_cxcalc.csv',
					ID_name = 'id',
					smiles_id = 'R-SMILES'):
	"""
	1. Load a molgen file and a cxcalc file and merge them using 'id',
	   which is now 'ID' in molgen file and so, should be updated to save as 'id'.
	1.1 if SMILES_id is not SMILES, a new column of SMILES will be generated with 
	the same content with the given SMILES_id column. 
	2. predict logS
	"""

	# Merging part
	pdr_merge = pd_merge_molgen_cxcalc( fname_molgen = fname_molgen, fname_cxcalc = fname_cxcalc, ID_name = ID_name, smiles_id = smiles_id)

	# Predict solubility (logS), to do it the output pdr should be saved to a file. 	
	fname_merge = fname_cxcalc[:-4] + '+molgen' + '.csv'
	pdr_merge.to_csv( fname_merge, index = False)
	pdr_sol = predict_logS(fname_merge)

	# Since logSpH() uses logS as Solubility, the name of the column is changed. 
	pdr_sol.rename( columns = {'Solubility': 'PlogS'}, inplace = True)
	pdr = get_logSpH( pdr_sol)

	if fname is not None:
		pdr.to_csv( fname, index = False)

	return pdr


def pd_cxcalc_logSpH( 
					fname_molgen = 'sheet/alloxazine/Alloxazine2_64_RSMILES_molgen.csv', 
					fname_cxcalc = 'sheet/alloxazine/Alloxazine2_64_RSMILES_cxcalc.csv',
					fname = None,
					ID_name = 'id',
					smiles_id = 'SMILES'):
	"""
	1. Load a molgen file and a cxcalc file and merge them using 'id',
	   which is now 'ID' in molgen file and so, should be updated to save as 'id'.
	1.1 if SMILES_id is not SMILES, a new column of SMILES will be generated with 
	the same content with the given SMILES_id column. 
	2. predict logS

	- The argument position of fname is changed from the first to the third. 
	"""

	# Merging part
	pdr_molgen = pd.read_csv( fname_molgen)
	pdr_cxcalc = pd.read_csv( fname_cxcalc, sep = '\t')
	pdr_merge = pd.merge(pdr_molgen, pdr_cxcalc, on='id')

	# Predict solubility (logS), to do it the output pdr should be saved to a file. 	
	fname_merge = fname_cxcalc[:-4] + '+molgen' + '.csv'
	pdr_merge.to_csv( fname_merge, index = False)
	pdr_sol = predict_logS(fname_merge)

	# Since logSpH() uses logS as Solubility, the name of the column is changed. 
	pdr_sol.rename( columns = {'Solubility': 'PlogS'}, inplace = True)
	pdw = get_logSpH( pdr_sol)

	if fname is not None:
		pdw.to_csv( fname, index = False)

	return pdw
