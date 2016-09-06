# Sungjin (James) Kim, Jul 5, 2016
"""
jlogs.py
--------
Solubility prediction with respect to pH as well as intrinsic values. 
"""

import numpy as np
import pandas as pd
import subprocess

# Internal libraries
import jpandas as jpd

def _save_smiles( smiles_l, fname_smiles):
	#print("Saving smiles")
	s_df = pd.DataFrame()
	s_df["SMILES"] = smiles_l
	s_df["id"] = range( 1, len( smiles_l) + 1) 
	s_df.to_csv( fname_smiles, index = False, header = False, sep = '\t')

def _get_from_cxcalc( fname_smiles):
	# print("Get from cxcalc")
	# Get logP_a
	with open("tmp_logp.tsv", "w") as f:
		cmd = ['cxcalc logp {}'.format( fname_smiles)]
		p = subprocess.Popen(cmd, shell=True, universal_newlines=True, stdout=f)
		p.wait()

	logP_df = pd.read_csv( "tmp_logp.tsv", index_col=0, sep = '\t')
	logP_a = logP_df.logP.values

	# Get logD_pH_aa
	with open("tmp_logd_ph.tsv", "w") as f:
		cmd = ['cxcalc logd tmp.smiles']
		p = subprocess.Popen(cmd, shell=True, universal_newlines=True, stdout=f)
		p.wait()

	logD_pH_df = pd.read_csv( "tmp_logd_ph.tsv", index_col=0, sep = '\t')
	logD_pH_aa = logD_pH_df.values

	return logP_a, logD_pH_aa

def _predict_logS_in( smiles_l):
	# print("predict logS_in")
	jmbr = jpd.PD_MBR_Solubility_Fast( fname_model = 'sheet/wang1708_smiles.pkl', mode = 'offline',
				fname_db='sheet/wang1708_smiles.csv',
				y_id = 'exp', graph = False, disp = False)
	jmbr.predict_pdr_pri()
	yV = jmbr.predict_smiles( smiles_l)
	logS_in_a = np.array( yV)[:,0]

	return logS_in_a

def _predict_logS_pH( logS_in_a, logP_a, logD_pH_aa):
	"""
	calculate logS_pH_aa
	"""
	logS_pH_aa = logS_in_a.reshape(-1,1) + logP_a.reshape(-1,1) - logD_pH_aa
	return( logS_pH_aa)

def predict_logS( smiles_l):
	"""
	predict_logs
	------------
	Prediction intrinsic and pH-depedent logS

	Input
	-----
	smiles_l: list of SMILES strings

	Return - dictionary form
	------
	logs_in (Ns ndarray), logs_pH_l (Ns x 15 ndarray)
	where Ns is length of smiles_l

	Example
	-------
	smiles_l = ['c1cc2c(cc1O)C(=O)c3cc(ccc3C2=O)', 'c1cc2c(cc1O)C(=O)c3cc(ccc3C2=O)O']
	d = jlogs.predict_logS( smiles_l)
	print( d)

	Output
	------
	{'logS_pH': array([[ 1.35501318,  1.35501318,  1.35501318,  1.35501318,  1.34501318,
		 1.34501318,  1.32501318,  1.17501318,  0.58501318, -0.22498682,
		-0.68498682, -0.78498682, -0.79498682, -0.79498682, -0.79498682],
	   [ 1.27128874,  1.27128874,  1.27128874,  1.27128874,  1.27128874,
		 1.27128874,  1.23128874,  0.92128874, -0.24871126, -1.86871126,
		-2.79871126, -2.99871126, -3.01871126, -3.01871126, -3.01871126]]), 'logS_in': array([-3.88498682, -3.34871126])}
	"""

	fname_smiles = "tmp.smiles"
	_save_smiles( smiles_l, fname_smiles)
	logP_a, logD_pH_aa = _get_from_cxcalc( fname_smiles)
	logS_in_a = _predict_logS_in( smiles_l)
	logS_pH_aa = _predict_logS_pH( logS_in_a, logP_a, logD_pH_aa)

	return {"logS_in": logS_in_a, "logS_pH": logS_pH_aa}