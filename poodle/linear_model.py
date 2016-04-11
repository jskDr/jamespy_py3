# poodle
# Sung-Jin Kim, April 8, 2016

from sklearn import linear_model 
import pandas as pd
import numpy as np

from sklearn import grid_search # Will be removed soon

def read_csv( *args, index_col=0, header=[0,1], **kwargs):
	"""
	Emulation for pandas.DataFrame() 
	Parameters
	----------
	The parameters of Pandas DataFrame are used 
	*args : any type
		 all arguments without a keyword
	**kwargs: any type
		 all arguments without a keyword
	"""
	return pd.read_csv( *args, index_col=index_col, header=header, **kwargs)

class LinearRegression( linear_model.LinearRegression):
	def __init__( self, **kwargs):
		super().__init__(**kwargs)

	def fit( self, xy_file):
		df = read_csv( xy_file)
		X = df['X'].values
		y = df['y'].values
		return super().fit( X, y)

	def predict( self, x_file, fname_out):
		# Predict from data in x_file
		x_df = read_csv( x_file)
		X = x_df['X'].values
		yp = super().predict( X)

		# Save the prediction results into fname_out
		yi_s = list(map( lambda i: 'y{}'.format(i), range( yp.shape[1])))
		idx = pd.MultiIndex.from_product( [['yp'], yi_s])
		yp_df = pd.DataFrame( yp, columns = idx, index = x_df.index)
		xyp_df = pd.concat( [x_df, yp_df], axis = 1)
		xyp_df.to_csv( fname_out)

class GridSearchCV( grid_search.GridSearchCV):
	def __init__(self, fname, estimator = None, param_grid = None, **kwargs):	
		"""
		estimator and param_grid can be values defined in csv. 
		"""
		if estimator is None:
			estimator = linear_model.LinearRegression()
		if param_grid is None:
			param_grid = {}

		super().__init__( estimator, param_grid, **kwargs)

		self.fname = fname

		# 1. Read all data
		self.df = read_csv( fname)

		# 2. Separate data and parameters
		df_data = self.df[["X", "y"]]
		self.df_param = self.df["param"].set_index("index")
		# Later, Nan will be removed
		# Later, comments are separated. 

		# make data to be useful for regression
		self.X = df_data['X'].values
		self.y = df_data['y'].values

		# Later, the parameter part will be used.

	def fit(self):
		return super().fit( self.X, self.y)

	def cross_val_predict(self, fname_out = None):
		"""
		This function is added to save the result of the predicted values. 
		"""
		yp = cross_validation.cross_val_predict( self.best_estimator_, self.X, self.y)

		idx = pd.MultiIndex.from_product([['yp'], self.df['y'].columns])
		yp_df = pd.DataFrame( yp, index = self.df.index, columns=idx)
		df_out_org = self.df.merge( yp_df, left_index = True, right_index = True)
		self.df_out = DataFrame( df_out_org[["X", "y", "yp", "param"]])
		# df_out = pd.concat([self.df, yp_df], axis = 1)

		self.df_out.to_csv_excel( '_out', self.fname, fname_out)		

		return yp