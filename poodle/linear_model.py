# poodle
# Sung-Jin Kim, April 8, 2016

from sklearn import linear_model 
import pandas as pd
import numpy as np

from sklearn import grid_search, cross_validation

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
		yi_s = list(range( yp.shape[1]))
		idx = pd.MultiIndex.from_product( [['yp'], yi_s])
		yp_df = pd.DataFrame( yp, columns = idx, index = x_df.index)
		xyp_df = pd.concat( [x_df, yp_df], axis = 1)
		xyp_df.to_csv( fname_out)

class GridSearchCV( grid_search.GridSearchCV):
	def __init__(self, estimator = None, param_grid = None, **kwargs):	
		"""
		estimator and param_grid can be values defined in csv. 
		"""
		if estimator is None:
			estimator = linear_model.LinearRegression()
		if param_grid is None:
			param_grid = {}

		super().__init__( estimator, param_grid, **kwargs)

	def fit(self, xy_file, fname_out):
		"""
		All grid results will be saved later,
		although only the best result is saved.
		"""

		df = read_csv( xy_file)
		X = df['X'].values
		y = df['y'].values
		
		super().fit( X, y)

		yp = cross_validation.cross_val_predict( self.best_estimator_, X, y)

		m_idx = pd.MultiIndex.from_product([['yp'], df['y'].columns])
		yp_df = pd.DataFrame( yp, index = df.index, columns=m_idx)
		df_out = pd.concat([df, yp_df], axis = 1)

		df_out.to_csv( fname_out)

		return self