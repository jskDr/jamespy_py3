# kcell.py
# python3

import numpy as np
import pandas as pd

from sklearn import cross_validation, svm, metrics, cluster, tree, ensemble
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import kkeras

import klstm

param_d = {"n_cv_flt": 2, "n_cv_ln": 3, "cv_activation": "relu"}

def setparam( p_d):
	"""
	setup global parameters

	Input
	------
	p_d: dictionary
	This includes global parameters and values. 
	
	Parameters
	-------------
	n_cv_flt, int: number of CNN filters
	n_cv_ln, int: length of a CNN filter    
	"""

	global param_d

	for k in p_d:
		param_d[ k] = p_d[ k]


def clst( X_train, y_train, X_test, y_test, nb_classes):
	model = tree.DecisionTreeClassifier()
	model.fit( X_train, y_train)
	dt_score = model.score( X_test, y_test)
	print( "DT-C:", dt_score)

	model = svm.SVC( kernel = 'linear')
	model.fit( X_train, y_train)
	sv_score = model.score( X_test, y_test)
	print( "SVC:", sv_score)

	model = kkeras.MLPC( [X_train.shape[1], 30, 10, nb_classes])
	model.fit( X_train, y_train, X_test, y_test, nb_classes)
	mlp_score = model.score( X_test, y_test)
	print( "DNN:", mlp_score)

	model = kkeras.CNNC( param_d["n_cv_flt"], param_d["n_cv_ln"], param_d["cv_activation"],
		l=[X_train.shape[1], 30, 10, nb_classes])
	model.fit( X_train, y_train, X_test, y_test, nb_classes)
	cnn_score = model.score( X_test, y_test)
	print( "DCNN:", cnn_score)

	model = ensemble.RandomForestClassifier( n_estimators=10)
	model.fit( X_train, y_train)
	rf_score = model.score( X_test, y_test)
	print( "RF:", rf_score)

	return dt_score, sv_score, mlp_score, cnn_score, rf_score


def GET_clsf2_by_clst( nb_classes):
	def clsf2_by_clst( Xpart_cf, Xpart_ct):
		"""
		Clustering is performed and then, classification performed by clustered indices. 
		"""
		cl_model = cluster.KMeans(n_clusters=nb_classes)
		cl_model.fit(Xpart_ct)
		yint = cl_model.predict( Xpart_ct)

		X_train, X_test, y_train, y_test = cross_validation.train_test_split( Xpart_cf, yint, test_size = 0.2)

		return clst(X_train, y_train, X_test, y_test, nb_classes)

	return clsf2_by_clst

def GET_clsf2_by_yint( nb_classes):
	def clsf2_by_yint( X1part, yint):
		"""
		classification is performed by yint
		"""
		X_train, X_test, y_train, y_test = cross_validation.train_test_split( X1part, yint, test_size = 0.2)

		return clst(X_train, y_train, X_test, y_test, nb_classes)

	return clsf2_by_yint

def pd_df( ix, s_l, ms):
	VI = {1:"Velocity", 2:"Intensity", 12:"Combined"}
	ln = len( s_l)

	df_i = pd.DataFrame()
	df_i["Type"] = ["{}: ".format(ms) + str( ix)] * ln
	df_i["Clustering"] = [ VI[ix[0]]] * ln
	df_i["Classification"] = [ VI[ix[1]]] * ln
	df_i["Clustering method"] = [ "KMeans"] * ln
	df_i["Classification method"] = [ "DT", "SVC", "DNN", "DCNN", "RF"]
	df_i["Pc"] = s_l

	return df_i 

def pd_clsf2_by_clst( ix, Xpart_ct, Xpart_cf, nb_classes):
	print( "Type", ix, "- Clustering:", ix[1], "Classification:", ix[0])
	s_l = GET_clsf2_by_clst(nb_classes)(Xpart_cf, Xpart_ct)
	return pd_df( ix, s_l, "KMeans")

def pd_clsf2_by_yint( ix, yint, Xpart_cf, nb_classes):
	print( "Type", ix, "- Clustering:", ix[1], "Classification:", ix[0])
	s_l = GET_clsf2_by_yint(nb_classes)(Xpart_cf, yint)
	return pd_df( ix, s_l, "Science")

class Cell_Mat_Data:
	def __init__(self, ofname = None, time_range = None, norm_flag = False, 
					dir_name = '../data/all-3x9', 
					disp = False):
		"""
		Input params
		------------
		time_range = (s,e), 1D list
		Cut time of a range from s to e for every sequence 

		Usage
		-----
		Cell_Mat_Data( '../data/all-3x9-norm.csv', time_range=(199,251), norm_flag = True, disp = False)
		"""
		self.mode_l = ['Velocity', 'Intensity', 'Distance']
		self.time_range = time_range
		self.norm_flag = norm_flag
		self.dir_name = dir_name
		self.disp = disp
		
		if ofname is not None:
			cell_df = self.get_df()
			cell_df.to_csv( ofname, index=False)

	def preprocessing(self, val_2d_l):
		time_range = self.time_range
		norm_flag = self.norm_flag
		
		if time_range is not None:
			val_2d_l = val_2d_l[:, time_range[0]:time_range[1]]
			
		if norm_flag:
			assert(time_range is not None)
			val_2d_l /= np.linalg.norm( val_2d_l, axis=1, keepdims=True)           
		return val_2d_l
	
	def get_df_i(self, pname, idx):
		mode_l = self.mode_l
		dir_name = self.dir_name
		disp = self.disp
		
		csv_fname = '{0}/{1}-{2}.csv'.format(dir_name,pname,idx+1)
		csv_df = pd.read_csv( csv_fname, header=None)
		val_2d_l = self.preprocessing( csv_df.values)
		val = val_2d_l.reshape( -1)

		if disp:
			print(pname, csv_fname, csv_df.shape)
			#print( val[:10])      

		# generate a new dataframe
		df_i = pd.DataFrame()
		df_i['Protein'] = [pname] * np.prod(val_2d_l.shape)
		df_i['Mode'] = [mode_l[ int(idx%3)]] * np.prod(val_2d_l.shape)
		df_i['Cluster'] = [int(idx/3)] * np.prod(val_2d_l.shape)
		df_i['sample'] = np.repeat( list(range( val_2d_l.shape[0])), val_2d_l.shape[1])
		df_i['time'] = list(range( val_2d_l.shape[1])) * val_2d_l.shape[0]
		df_i['value'] = val

		# print( df_i.shape, df_i.keys())
		
		return df_i
		
	def get_df(self):
		df_l = list()
		for pname in [ 'arp23', 'cofilin1', 'vasp']:
			for idx in range(9):
				df_i = self.get_df_i( pname, idx)
				df_l.append( df_i)
				
		df = pd.concat( df_l, ignore_index=True)
		return df

class Cell_Mat_Data4( Cell_Mat_Data):
	def __init__(self, ofname = None, time_range = None, norm_flag = False, 
					dir_name = '../data/raw-4x9',
					pname_l = [ 'arp23', 'cofilin1', 'VASP', 'Actin'],
					disp = False):
		"""
		Input params
		------------
		time_range = (s,e), 1D list
		Cut time of a range from s to e for every sequence 

		pname_l, 1D string list
		list of candidate protein names
		e.g. pname_l = [ 'arp23', 'cofilin1', 'VASP', 'Actin']

		Usage
		-----
		Cell_Mat_Data( '../data/all-3x9-norm.csv', time_range=(199,251), norm_flag = True, disp = False)        
		"""
		self.pname_l = [ 'arp23', 'cofilin1', 'VASP', 'Actin']
		super().__init__( ofname=ofname, time_range=time_range, norm_flag=norm_flag, 
							dir_name=dir_name, disp=disp)

	def get_df(self):
		pname_l = self.pname_l

		df_l = list()
		for pname in pname_l:
			for idx in range(9):
				df_i = self.get_df_i( pname, idx)
				df_l.append( df_i)
				
		df = pd.concat( df_l, ignore_index=True)
		return df       

def _cell_get_X_r0( cell_df, Protein = 'VASP', Mode = 'Velocity', Cluster_l = [0]):
	"""
	Input
	-----
	cell_df, pd.DataFrame
	key() = ['Protein', 'Mode', 'Cluster', 'sample', 'time', 'value']
	e.g., cell_df = pd.read_csv( '../data/raw-4x9-norm.csv')

	"""
	cell_df_vasp_velocity_c0 = cell_df[ 
		(cell_df.Protein == Protein) & (cell_df.Mode == Mode) & (cell_df.Cluster.isin( Cluster_l))]

	x_vec = cell_df_vasp_velocity_c0.value.values
	# x_vec.shape

	# search ln
	df_i = cell_df[ (cell_df.Protein == Protein) & (cell_df.Mode == Mode) 
				& (cell_df.Cluster == 0) & (cell_df['sample'] == 0)]
	l_time = df_i.shape[0]
	#print( l_time)

	X = x_vec.reshape( -1, l_time)
	#print( X.shape)
	
	return X

def cell_get_X( cell_df, Protein = 'VASP', Mode = 'Velocity', Cluster_l = [0]):
	"""
	Input
	-----
	cell_df, pd.DataFrame
	key() = ['Protein', 'Mode', 'Cluster', 'sample', 'time', 'value']
	e.g., cell_df = pd.read_csv( '../data/raw-4x9-norm.csv')

	"""
	cell_df_vasp_velocity_c0 = cell_df[ 
		(cell_df.Protein == Protein) & (cell_df.Mode == Mode) & (cell_df.Cluster.isin( Cluster_l))]

	x_vec = cell_df_vasp_velocity_c0.value.values
	# x_vec.shape

	# search ln
	df_i0 = cell_df_vasp_velocity_c0[ cell_df_vasp_velocity_c0.Cluster == Cluster_l[0]]
	df_i = df_i0[ df_i0["sample"] == df_i0["sample"].values[0]]

	l_time = df_i.shape[0]
	#print( l_time)

	X = x_vec.reshape( -1, l_time)
	#print( X.shape)
	
	return X

def cell_get_cluster( cell_df, Protein = 'VASP', Mode = 'Velocity'):
	"""
	Input
	-----
	cell_df, pd.DataFrame
	key() = ['Protein', 'Mode', 'Cluster', 'sample', 'time', 'value']
	e.g., cell_df = pd.read_csv( '../data/raw-4x9-norm.csv')

	"""
	# To pick up cluster indices, consider only one time in a sequence
	# since all time samples in a sequence are associated to the same cluster.
	time_set = set( cell_df['time'])
	a_time = list(time_set)[0]

	cell_df_vasp_velocity_c0 = cell_df[ 
		(cell_df.Protein == Protein) & (cell_df.Mode == Mode) & (cell_df.time == a_time)]

	clusters_a = cell_df_vasp_velocity_c0["Cluster"].values
	
	return clusters_a

def cell_show_XcYc( cell_df, Protein = 'arp23'):
	X_VASP_Velocity = kcellml.cell_get_X(cell_df, Protein, 'Velocity')
	X_VASP_Intensity = kcellml.cell_get_X(cell_df, Protein, 'Intensity')

	X1 = X_VASP_Velocity
	X2 = X_VASP_Intensity
	#yint_org = np.loadtxt( 'sheet/vasp_y.gz').astype( int)
	#X1.shape

	#X1part = X1 
	X1part = X1 / np.linalg.norm( X1, axis = 1, keepdims=True)
	X2part = X2 / np.linalg.norm( X2, axis = 1, keepdims=True)

	cca = cross_decomposition.CCA(n_components=1)
	cca.fit(X1part.T, X2part.T)
	X_c, Y_c = cca.transform(X1part.T, X2part.T)

	line, = plt.plot( X_c, label = 'Velocity({})'.format(Protein))
	c = plt.getp( line, 'color')
	plt.plot( Y_c, '.-', color=c, label = 'Intensity({})'.format(Protein))
	plt.title('Canonical Correlation Analysis: {}'.format(Protein))
	plt.legend(loc = 0, fontsize='small')

def cell_show_avg_XY(cell_df, Protein = 'arp23'):
	X_VASP_Velocity = kcellml.cell_get_X(cell_df, Protein, 'Velocity')
	X_VASP_Intensity = kcellml.cell_get_X(cell_df, Protein, 'Intensity')

	X1 = X_VASP_Velocity
	X2 = X_VASP_Intensity
	#yint_org = np.loadtxt( 'sheet/vasp_y.gz').astype( int)
	#X1.shape

	#X1part = X1 
	X1part = X1 / np.linalg.norm( X1, axis = 1, keepdims=True)
	X2part = X2 / np.linalg.norm( X2, axis = 1, keepdims=True)

	X_c, Y_c = np.average( X1part, axis = 0), np.average( X2part, axis =0)    

	line, = plt.plot( X_c, label = 'Velocity({})'.format(Protein))
	c = plt.getp( line, 'color')
	plt.plot( Y_c, '.-', color=c, label = 'Intensity({})'.format(Protein))
	plt.title('Average: {}'.format(Protein))
	plt.legend(loc = 0, fontsize='small')

def get_nonan(x, disp = False):
	st_in = np.where( np.isnan(x)== False)[0][0]
	ed_ex = np.where( np.isnan(x[st_in:]) == True)[0][0] + st_in
	x_nonan = x[st_in: ed_ex]
	if disp: 
		print( st_in, ed_ex)
	return x_nonan

def conv_circ( signal, ker ):
	'''
		signal: real 1D array
		ker: real 1D array
		signal and ker must have same shape
	'''
	return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) ))

def plot_nonan_xy( x, y):
	x = X_VASP_Intensity[k,:]
	y = X_VASP_Velocity[k,:]

	x_nonan = get_nonan(x)
	y_nonan = get_nonan(y)

	plot_xy( x_nonan, y_nonan)
	
def plot_xy( ac_x_nonan, ac_y_nonan):
	ac_x_nonan -= np.mean( ac_x_nonan)
	ac_y_nonan -= np.mean( ac_y_nonan)
	
	ac_x_nonan /= np.max( ac_x_nonan) / np.max( ac_y_nonan)

	plt.plot( ac_x_nonan)
	plt.plot( ac_y_nonan)

def plot_ac_xy( k):
	x = X_VASP_Intensity[k,:]
	y = X_VASP_Velocity[k,:]

	x_nonan = get_nonan(x)
	y_nonan = get_nonan(y)
	x_nonan.shape, y_nonan.shape

	ac_x_nonan = conv_circ( x_nonan, x_nonan)
	ac_y_nonan = conv_circ( y_nonan, y_nonan)

	ac_x_nonan -= np.mean( ac_x_nonan)
	ac_y_nonan -= np.mean( ac_y_nonan)
	
	ac_x_nonan /= np.max( ac_x_nonan) / np.max( ac_y_nonan)

	plt.plot( ac_x_nonan)
	plt.plot( ac_y_nonan)

def get_k( train_k, look_back):
	trainX, trainY = klstm.create_dataset(train_k.reshape(-1,1), look_back)
	trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	return trainX, trainY

def get_train_test( X, pca_order = 10):
	X = X.astype('float32')

	scaler = MinMaxScaler(feature_range=(0, 1))
	X = scaler.fit_transform(X.reshape(-1,1)).reshape( X.shape)

	if pca_order > 0:
		pca = PCA(3)
		X = pca.fit_transform(X)
		X = pca.inverse_transform(X)   
		
	n_samples = X.shape[0]
	train_size = int(n_samples * 0.67)
	test_size = n_samples - train_size
	train, test = X[0:train_size,:], X[train_size:n_samples,:]
	return train, test, scaler   

def get_VI( fname = '../data/raw-4x9-195to251.csv', time_range = (195, 251), clusters_l=[0,1,2]):
	cell_df = pd.read_csv( fname)

	pname_l=['arp23', 'cofilin1', 'VASP', 'Actin']

	l_t = time_range[1] - time_range[0]

	V, I = {}, {}
	for pname in pname_l:
		print( "Reading:", pname)
		V[ pname] = cell_get_X(cell_df, pname, 'Velocity', clusters_l)
		I[ pname] = cell_get_X(cell_df, pname, 'Intensity', clusters_l)
		print( V[ pname].shape, I[pname].shape)
		
	return V, I

def get_VIC( fname = '../data/raw-4x9-195to251.csv', time_range = (195, 251), 
				clusters_l=[0,1,2],
				pname_l=['arp23', 'cofilin1', 'VASP', 'Actin']):
	# cluster_l is fixed. Later, this information should be obtained automatically using set() and list()
	cell_df = pd.read_csv( fname)

	#l_t = time_range[1] - time_range[0]

	V, I, C = {}, {}, {}
	for pname in pname_l:
		print( "Reading:", pname)
		V[ pname] = cell_get_X(cell_df, pname, 'Velocity', clusters_l)
		I[ pname] = cell_get_X(cell_df, pname, 'Intensity', clusters_l)
		C[ pname] = cell_get_cluster(cell_df, pname)
		print( V[ pname].shape, I[pname].shape, C[pname].shape)
		
	return V, I, C