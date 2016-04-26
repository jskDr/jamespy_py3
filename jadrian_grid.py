# Python 3
"""
grid search is included for optimizing adrian molecules
"""

from collections import OrderedDict
import pandas as pd

# import jadrian
#import jgrid
import kgrid
from jadrian import MedianMeanStd, iter_od_base

class MedianMeanStd_Run( MedianMeanStd):
	def __init__(self, test_flag = False, default_d = {'CV Mode': ['LOO']}, fname_out_tag = ""):
		"""
		Multiple cases are invoked here.
		Now save for each mode. Later, I will save all mode results at the same time.
		"""

		self_od = OrderedDict()
		for od, aod in iter_od_base( default_d):
			"""
			All iteration results will be save to self_od
			"""
			op_mode = aod['op_mode']
			print( 'op_mode is', op_mode)
			if test_flag:
				super().__init__( od, aod)
				self.run()
				self.test()
				self_od[op_mode] = self.pdo.copy()
			else:
				super().__init__( od, aod)
				self.run()
				self_od[op_mode] = self.pdo.copy()
	
			#print( 'QC Models (Family ID) is', self_od[op_mode].pdo['QC Models (Family ID)'])
		"""
		# array will be saved as pointers not values, which should be kept in mind.
		print( 'self_od.keys() are', self_od.keys())
		for op_mode in list(self_od.keys()):
			print( 'op_mode is', op_mode)
			s_pdo = self_od[ op_mode]
			print( 'QC Models (Family ID) is', s_pdo['QC Models (Family ID)'])	
		"""

		# All pdo are collected and saved to out_x.csv where x is number of op_mode(s).
		pdo_all = pd.DataFrame()
		for s_pdo in list(self_od.values()):
			#print( 'QC Models (Family ID) is', s.pdo['QC Models (Family ID)'])			
			pdo_all = pdo_all.append( s_pdo, ignore_index = True)
		all_out_file = "sheet/out_mms_{0}{1}.csv".format( len(self_od), fname_out_tag)
		print('The collected dataframe is saved to', all_out_file)
		pdo_all.to_csv( all_out_file, index = False)
		

class MedianMeanStd_Expand_LOO( MedianMeanStd_Run):
	"""
	The results will be expanded for each molecule assuming LOO is used for 
	Crossvalidation. 
	"""
	def __init__( self, test_flag = False, default_d = {'CV Mode': ['LOO']}):
		"""
		Grid testing based on MedianMeanStd_Run().
		"""
		self.fname_out_tag = '_expand_loo'
		super().__init__( test_flag = test_flag, default_d = default_d, 
						fname_out_tag = self.fname_out_tag)


	def init_od( self, od): # list up all elements in od for ordering
		# od['CV Mode'] = ['No regress']	

		# This is input_od list
		# od = OrderedDict()
		# od['QC Models (Family ID)'] = [["B3LYP"]]
		# od['H + S'] = [False]
		# od['CV Mode'] = ['10*5KF/LOO']
		# od['Em type'] = ['Chemical potential']
		# od['Regularization'] = ['None']
		# od['Bounds/Constraints'] = ['None']

		od['Regression'] = []
		od['Group mode'] = []
		od['Group(s)'] = []
		od['Unit'] = []
		od['Target'] = []
		od['Result'] = []
		od['ABS_err'] = []
		#od['Median_ABS_err'] = []
		#od['Mean_ABS_err'] = []
		#od['STD_ABS_err'] = []
		#od['(coef_,intercept_)'] = []

		return od
		
	def each_base( self, pdr, type_id = 0): # pdr = pdr[ pdr.Type == type_id]:
		"""
		if type_id is not defined, it becomes 0.
		"""
		od = self.od.copy() 
		xM, yV = self.get_xM_yV( pdr)
		
		# shuffle is False since it consider LOO and will be compared with 
		# original molecule index. 
		o_d = kgrid.cv_LOO( xM, yV, ldisp = self.Disp)
		
		# print( "yVp = ", o_d['yVp'])

		# Original value vector is extended to have repeated value vector
		for key in list( self.od):
			od[ key] = self.od[ key] * pdr.shape[0]

		od['CV Mode'] = ['LOO'] * pdr.shape[0]
		od['Regression'] = ['Linear'] * pdr.shape[0]
		od['Group mode'] = ['Independent'] * pdr.shape[0]
		od['Group(s)'] = [type_id] * pdr.shape[0]

		od['Unit'] = range( pdr.shape[0])
		od['Target'] = yV.A1.tolist()
		od['Result'] = o_d['yVp']
		od['ABS_err'] = o_d['list']
		#od['Median_ABS_err'] = [o_d['median_abs_err']] * pdr.shape[0]
		#od['Mean_ABS_err'] = [o_d['mean_abs_err']] * pdr.shape[0]
		#od['STD_ABS_err'] = [o_d['std_abs_err']] * pdr.shape[0]

		#od['(coef_,intercept_)'] = [o_d['ci']] * pdr.shape[0]
	
		#for key in list( od):
		#	print( 'length of od[{}] of od is'.format(key), len(od[key]))

		return od

	def run(self):
		self.each()

		fname_out = self.out_file[:-4] + self.fname_out_tag + '.csv'
		print("The result dataframe is saved to", fname_out)
		self.pdo.to_csv( self.out_file, index = False)
		return self

class LOO_Grid( MedianMeanStd_Expand_LOO):
	def __init__(self, alpha_l = [0]):
		"""
		Ridge based grid search is applied with alpha_l
		"""
		self.alpha_l = alpha_l
		super().__init__()

	def each_base( self, pdr, type_id = 0): # pdr = pdr[ pdr.Type == type_id]:
		"""
		if type_id is not defined, it becomes 0.
		"""
		od = self.od.copy() 
		xM, yV = self.get_xM_yV( pdr)
		
		# shuffle is False since it consider LOO and will be compared with 
		# original molecule index. 
		o_d = kgrid.cv_LOO_Ridge( xM, yV, od['alpha'], ldisp = self.Disp)

		# print( "yVp = ", o_d['yVp'])

		# Original value vector is extended to have repeated value vector
		for key in list( self.od):
			od[ key] = self.od[ key] * pdr.shape[0]

		od['CV Mode'] = ['LOO'] * pdr.shape[0]
		od['Regression'] = ['Linear'] * pdr.shape[0]
		od['Group mode'] = ['Independent'] * pdr.shape[0]
		od['Group(s)'] = [type_id] * pdr.shape[0]

		od['Unit'] = range( pdr.shape[0])
		od['Target'] = yV.A1.tolist()
		od['Result'] = o_d['yVp']
		od['ABS_err'] = o_d['list']
		#od['Median_ABS_err'] = [o_d['median_abs_err']] * pdr.shape[0]
		#od['Mean_ABS_err'] = [o_d['mean_abs_err']] * pdr.shape[0]
		#od['STD_ABS_err'] = [o_d['std_abs_err']] * pdr.shape[0]

		#od['(coef_,intercept_)'] = [o_d['ci']] * pdr.shape[0]
	
		#for key in list( od):
		#	print( 'length of od[{}] of od is'.format(key), len(od[key]))

		return od

	def run(self):
		self.od['Regularization'] = ['Ridge']
		for alpha in self.alpha_l:
			self.od['alpha'] = [alpha]
			self.each()

		fname_out = self.out_file[:-4] + self.fname_out_tag + '.csv'
		print("The result dataframe is saved to", fname_out)
		self.pdo.to_csv( self.out_file, index = False)
		return self

class LOO_3Regression( MedianMeanStd_Expand_LOO):
	def __init__(self, alpha_l = [0]):
		"""
		Ridge based grid search is applied with alpha_l
		"""
		self.alpha_l = alpha_l
		super().__init__()

	def each_base( self, pdr, type_id = 0): # pdr = pdr[ pdr.Type == type_id]:
		"""
		if type_id is not defined, it becomes 0.
		"""
		od = self.od.copy() 
		xM, yV = self.get_xM_yV( pdr)
		
		# shuffle is False since it consider LOO and will be compared with 
		# original molecule index. 
		o_d = kgrid.cv_LOO_mode( od['Regression'][0], xM, yV, ldisp = self.Disp)

		# Original value vector is extended to have repeated value vector
		for key in list( self.od):
			od[ key] = self.od[ key] * pdr.shape[0]

		od['CV Mode'] = ['LOO'] * pdr.shape[0]
		# od['Regression'] = ['Linear'] * pdr.shape[0]
		od['Group mode'] = ['Independent'] * pdr.shape[0]
		od['Group(s)'] = [type_id] * pdr.shape[0]

		od['Unit'] = range( pdr.shape[0])
		od['Target'] = yV.A1.tolist()
		od['Result'] = o_d['yVp']
		od['ABS_err'] = o_d['list']

		return od

	def run(self):
		for r_type in ["None", "Bias", "Linear"]: 
			print( "Regression type:", r_type)
			self.od['Regression'] = [r_type]
			self.each()

		fname_out = self.out_file[:-4] + self.fname_out_tag + '.csv'
		print("The result dataframe is saved to", fname_out)
		self.pdo.to_csv( self.out_file, index = False)
		return self
