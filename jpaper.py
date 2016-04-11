"""
I will generate automatic codes for drawing data used in my papers. 
Three cases will be shown at the same time after completion of this coding. 
Currently, BIKE is only considered while later, MLR will be performed as well. 
"""

import pandas as pd
import numpy as np

import jpandas as jpd
import jutil
import jpyx
import jchem
import jgrid

def grid_BIKE_A(pdr, alphas_log, y_id = 'Solubility_log_mol_l'):
	print "BIKE with A"

	xM1 = jpd.pd_get_xM( pdr, radius=6, nBits=4096)
	xM2 = jpd.pd_get_xM_MACCSkeys( pdr)

	yV = jpd.pd_get_yV( pdr, y_id = y_id)

	A1 = jpyx.calc_tm_sim_M( xM1)
	A2 = jpyx.calc_tm_sim_M( xM2)
	A = np.concatenate( ( A1, A2), axis = 1)
	print A.shape

	molw_l = jchem.rdkit_molwt( pdr.SMILES.tolist())
	print np.shape( molw_l)
	A_molw = jchem.add_new_descriptor( A, molw_l)
	print A_molw.shape

	A_molw = A1
	gs = jgrid.gs_Ridge( A_molw, yV, alphas_log=alphas_log)
	jutil.show_gs_alpha( gs.grid_scores_)
	
	jgrid.cv( 'Ridge', A_molw, yV, alpha = gs.best_params_['alpha'])
	
	return gs

def grid_BIKE_B(pdr, alphas_log, y_id = 'Solubility_log_mol_l'):
	print "BIKE with B"

	xM1 = jpd.pd_get_xM( pdr, radius=6, nBits=4096)
	xM2 = jpd.pd_get_xM_MACCSkeys( pdr)

	yV = jpd.pd_get_yV( pdr, y_id = y_id)

	A1 = jpyx.calc_tm_sim_M( xM1)
	A2 = jpyx.calc_tm_sim_M( xM2)
	A = np.concatenate( ( A1, A2), axis = 1)
	print A.shape

	molw_l = jchem.rdkit_molwt( pdr.SMILES.tolist())
	print np.shape( molw_l)
	A_molw = jchem.add_new_descriptor( A, molw_l)
	print A_molw.shape

	A_molw = A2
	gs = jgrid.gs_Ridge( A_molw, yV, alphas_log=alphas_log)
	jutil.show_gs_alpha( gs.grid_scores_)
	
	jgrid.cv( 'Ridge', A_molw, yV, alpha = gs.best_params_['alpha'])
	
	return gs

def grid_BIKE_W(pdr, alphas_log, y_id = 'Solubility_log_mol_l'):
	print "BIKE with W"

	#xM1 = jpd.pd_get_xM( pdr, radius=6, nBits=4096)
	#xM2 = jpd.pd_get_xM_MACCSkeys( pdr)

	yV = jpd.pd_get_yV( pdr, y_id = y_id)

	#A1 = jpyx.calc_tm_sim_M( xM1)
	#A2 = jpyx.calc_tm_sim_M( xM2)
	#A = np.concatenate( ( A1, A2), axis = 1)
	#print A.shape

	molw_l = jchem.rdkit_molwt( pdr.SMILES.tolist())
	#print np.shape( molw_l)
	#A_molw = jchem.add_new_descriptor( A, molw_l)
	#print A_molw.shape

	A_molw = np.mat(molw_l).T
	gs = jgrid.gs_Ridge( A_molw, yV, alphas_log=alphas_log)
	jutil.show_gs_alpha( gs.grid_scores_)
	
	jgrid.cv( 'Ridge', A_molw, yV, alpha = gs.best_params_['alpha'])
	
	return gs	

def grid_BIKE2(pdr, alphas_log, y_id = 'Solubility_log_mol_l'):
	print "BIKE with (A+B)+W"

	xM1 = jpd.pd_get_xM( pdr, radius=6, nBits=4096)
	xM2 = jpd.pd_get_xM_MACCSkeys( pdr)

	yV = jpd.pd_get_yV( pdr, y_id = y_id)

	#A1 = jpyx.calc_tm_sim_M( xM1)
	#A2 = jpyx.calc_tm_sim_M( xM2)
	#A = np.concatenate( ( A1, A2), axis = 1)
	xM = np.concatenate( ( xM1, xM2), axis = 1)
	A = jpyx.calc_tm_sim_M( xM1)
	print A.shape

	molw_l = jchem.rdkit_molwt( pdr.SMILES.tolist())
	print np.shape( molw_l)
	A_molw = jchem.add_new_descriptor( A, molw_l)
	print A_molw.shape

	gs = jgrid.gs_Ridge( A_molw, yV, alphas_log=alphas_log)
	jutil.show_gs_alpha( gs.grid_scores_)
	
	jgrid.cv( 'Ridge', A_molw, yV, alpha = gs.best_params_['alpha'])
	
	return gs

def grid_BIKE3(pdr, alphas_log, y_id = 'Solubility_log_mol_l'):
	print "BIKE with A+B+W"

	xM1 = jpd.pd_get_xM( pdr, radius=6, nBits=4096)
	xM2 = jpd.pd_get_xM_MACCSkeys( pdr)

	yV = jpd.pd_get_yV( pdr, y_id = y_id)

	A1 = jpyx.calc_tm_sim_M( xM1)
	A2 = jpyx.calc_tm_sim_M( xM2)
	A = np.concatenate( ( A1, A2), axis = 1)
	print A.shape

	molw_l = jchem.rdkit_molwt( pdr.SMILES.tolist())
	print np.shape( molw_l)
	A_molw = jchem.add_new_descriptor( A, molw_l)
	print A_molw.shape

	gs = jgrid.gs_Ridge( A_molw, yV, alphas_log=alphas_log)
	jutil.show_gs_alpha( gs.grid_scores_)
	
	jgrid.cv( 'Ridge', A_molw, yV, alpha = gs.best_params_['alpha'])
	
	return gs

"""
Now the MLR version will be implemented.
"""
def grid_MLR3(pdr, alphas_log, y_id = 'Solubility_log_mol_l'):
	print "MLR with A+B+W"

	xM1 = jpd.pd_get_xM( pdr, radius=6, nBits=4096)
	xM2 = jpd.pd_get_xM_MACCSkeys( pdr)

	yV = jpd.pd_get_yV( pdr, y_id = y_id)

	#A1 = jpyx.calc_tm_sim_M( xM1)
	#A2 = jpyx.calc_tm_sim_M( xM2)

	xM = np.concatenate( ( xM1, xM2), axis = 1)
	print xM.shape

	molw_l = jchem.rdkit_molwt( pdr.SMILES.tolist())
	print np.shape( molw_l)
	xM_molw = jchem.add_new_descriptor( xM, molw_l)
	print xM_molw.shape

	gs = jgrid.gs_Ridge( xM_molw, yV, alphas_log=alphas_log)
	jutil.show_gs_alpha( gs.grid_scores_)
	
	jgrid.cv( 'Ridge', xM_molw, yV, alpha = gs.best_params_['alpha'])
	
	return gs

"""MLR with (A+B)+W is equivalent to MLR with A+B+W"""
grid_MLR2 = grid_MLR3

def grid_MLR_A(pdr, alphas_log, y_id = 'Solubility_log_mol_l'):
	print "MLR with A"

	xM1 = jpd.pd_get_xM( pdr, radius=6, nBits=4096)

	xM_molw = xM1
	yV = jpd.pd_get_yV( pdr, y_id = y_id)

	gs = jgrid.gs_Ridge( xM_molw, yV, alphas_log=alphas_log)
	jutil.show_gs_alpha( gs.grid_scores_)
	
	jgrid.cv( 'Ridge', xM_molw, yV, alpha = gs.best_params_['alpha'])
	
	return gs

def grid_MLR_B(pdr, alphas_log, y_id = 'Solubility_log_mol_l'):
	print "MLR with B"

	xM2 = jpd.pd_get_xM_MACCSkeys( pdr)

	xM_molw = xM2
	yV = jpd.pd_get_yV( pdr, y_id = y_id)

	gs = jgrid.gs_Ridge( xM_molw, yV, alphas_log=alphas_log)
	jutil.show_gs_alpha( gs.grid_scores_)
	
	jgrid.cv( 'Ridge', xM_molw, yV, alpha = gs.best_params_['alpha'])
	
	return gs


class PaperSol:
	def __init__(self, fname = 'sheet/ws_all_smiles_496.csv', y_id = 'Solubility_log_mol_l'):
		print "The title of this paper is as follows:"
		print "Binary Kernel Ensemble Machine For Solubility Prediction " + \
			"With Its Application Of Discovering Aqueous Flow Battery Electrolytes"
		
		self.pdr = pd.read_csv( fname)
		self.y_id = y_id

	def grid_BIKE3( self, alphas_log = (-2,2,5)):
		gs = grid_BIKE3( self.pdr, alphas_log=alphas_log, y_id = self.y_id)
		return gs

	def grid_BIKE2( self, alphas_log = (-2,2,5)):
		gs = grid_BIKE2( self.pdr, alphas_log=alphas_log, y_id = self.y_id)
		return gs

	def grid_BIKE_A( self, alphas_log = (-2,2,5)):
		gs = grid_BIKE_A( self.pdr, alphas_log=alphas_log, y_id = self.y_id)
		return gs

	def grid_BIKE_B( self, alphas_log = (-2,2,5)):
		gs = grid_BIKE_B( self.pdr, alphas_log=alphas_log, y_id = self.y_id)
		return gs

	def grid_BIKE_W( self, alphas_log = (-2,2,5)):
		gs = grid_BIKE_W( self.pdr, alphas_log=alphas_log, y_id = self.y_id)
		return gs

	def grid_BIKE( self, alphas_log = (-3,1,5)):
		grid_BIKE3( self.pdr, alphas_log=alphas_log, y_id = self.y_id)
		grid_BIKE2( self.pdr, alphas_log=alphas_log, y_id = self.y_id)
		grid_BIKE_A( self.pdr, alphas_log=alphas_log, y_id = self.y_id)
		grid_BIKE_B( self.pdr, alphas_log=alphas_log, y_id = self.y_id)
		grid_BIKE_W( self.pdr, alphas_log=alphas_log, y_id = self.y_id)

	def grid_MLR( self, alphas_log = (-3,1,5)):
		"""
		For MLR, only one version will be implemented at the moment. 
		Later, each case will be implemented. 
		"""
		grid_MLR3( self.pdr, alphas_log=alphas_log, y_id = self.y_id)
		#grid_MLR2( self.pdr, alphas_log=alphas_log, y_id = self.y_id)
		grid_MLR_A( self.pdr, alphas_log=alphas_log, y_id = self.y_id)
		grid_MLR_B( self.pdr, alphas_log=alphas_log, y_id = self.y_id)
		#grid_MLR_W( self.pdr, alphas_log=alphas_log, y_id = self.y_id)



