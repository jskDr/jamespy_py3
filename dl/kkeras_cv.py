# Keras for cv

from sklearn import model_selection
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import kkeras

def cv_deep_learning_with_w1( X, y, nb_classes = 3, mode = "CNNC_Name", 
							 l = None, param_d = None):
	"""
	l =[X.shape[1], 3, 30, nb_classes]
	param_d = {"n_cv_flt": 3, "n_cv_ln": 50, "cv_activation": "relu"}
	"""
	
	KF = model_selection.KFold(5, shuffle=True)
	kf = KF.split(X)

	dcnn_score_l = []
	c_wb_l = []
	for tr, te in kf:
		X_train, y_train = X[tr,:], y[tr]
		X_test, y_test = X[te,:], y[te]
		
		if mode == "CNNC_Name":
			model = kkeras.CNNC_Name( param_d["n_cv_flt"], param_d["n_cv_ln"], param_d["cv_activation"],
					l=l)
			n_flt = param_d["n_cv_flt"]
		elif mode == "CNNC_Name_Border":
			model = kkeras.CNNC_Name_Border( param_d["n_cv_flt"], param_d["n_cv_ln"], param_d["cv_activation"],
					l=l, border_mode = 'valid')
			n_flt = param_d["n_cv_flt"]           
		elif mode == "MLPC_Name":
			model = kkeras.MLPC_Name(l=l)
			n_flt = l[1]
		else:
			raise ValueError("The given mode is not supported: mode={}".format(mode))
			
		model.fit( X_train, y_train, X_test, y_test, nb_classes, batch_size=6, nb_epoch = 5)
		dcnn_score = model.score( X_test, y_test)
		dcnn_score_l.append(dcnn_score)
		print( "Accuracy:", dcnn_score)
		
		c_w, c_b = model.get_c_wb()
		print( "c_w.shape=", c_w.shape)
		c_w = c_w.reshape(-1, n_flt)
		c_wb_l.append( (c_w, c_b))

	print( dcnn_score_l)
	print( "Mean:{0}, Std:{1}".format( np.mean( dcnn_score_l), np.std( dcnn_score_l)))

	# One of weight vectors are drawn.
	c_w = c_wb_l[0][0] # 0 = c_w, 1 = c_b
	for ll in range(n_flt):
		#plt.figure()
		plt.plot( c_w[:,ll], label="Filter #{}".format(ll))
	plt.legend()  
	
	return dcnn_score_l

def cv_deep_learning_conv_out( X, y, nb_classes = 3, mode = "CNNC_Name", 
							 l = None, param_d = None, graph = True):
	"""
	l =[X.shape[1], 3, 30, nb_classes]
	param_d = {"n_cv_flt": 3, "n_cv_ln": 50, "cv_activation": "relu"}
	"""
	
	KF = model_selection.KFold(5, shuffle=True)
	kf = KF.split(X)

	dcnn_score_l = []
	c_wb_l = []
	for tr, te in kf:
		X_train, y_train = X[tr,:], y[tr]
		X_test, y_test = X[te,:], y[te]
		
		if mode == "CNNC_Name":
			model = kkeras.CNNC_Name( param_d["n_cv_flt"], param_d["n_cv_ln"], param_d["cv_activation"],
					l=l)
			n_flt = param_d["n_cv_flt"]
		elif mode == "CNNC_Name_Border":
			model = kkeras.CNNC_Name_Border( param_d["n_cv_flt"], param_d["n_cv_ln"], param_d["cv_activation"],
					l=l, border_mode = 'valid')
			n_flt = param_d["n_cv_flt"]           
		elif mode == "MLPC_Name":
			model = kkeras.MLPC_Name(l=l)
			n_flt = l[1]
		else:
			raise ValueError("The given mode is not supported: mode={}".format(mode))
			
		model.fit( X_train, y_train, X_test, y_test, nb_classes, batch_size=6, nb_epoch = 5)
		dcnn_score = model.score( X_test, y_test)
		dcnn_score_l.append(dcnn_score)
		print( "Accuracy:", dcnn_score)
		
		c_w, c_b = model.get_c_wb()
		print( "c_w.shape=", c_w.shape)
		c_w = c_w.reshape(-1, n_flt)
		c_wb_l.append( (c_w, c_b))

	print( dcnn_score_l)
	print( "Mean:{0}, Std:{1}".format( np.mean( dcnn_score_l), np.std( dcnn_score_l)))

	if graph:
		# One of weight vectors are drawn.
		c_w = c_wb_l[0][0] # 0 = c_w, 1 = c_b
		for ll in range(n_flt):
			#plt.figure()
			plt.plot( c_w[:,ll], label="Filter #{}".format(ll))
		plt.legend()  
	
	return dcnn_score_l, c_wb_l

def cv_deep_learning_pred( X, y, nb_classes = 3, mode = "CNNC_Name", 
							 l = None, param_d = None, graph = True, n_splits = 5):
	"""
	l =[X.shape[1], 3, 30, nb_classes]
	param_d = {"n_cv_flt": 3, "n_cv_ln": 50, "cv_activation": "relu"}
	"""
	
	KF = model_selection.KFold(n_splits, shuffle=True)
	kf = KF.split(X)

	dcnn_score_l = []
	c_wb_l = []
	y_cv = np.copy( y)
	for tr, te in kf:
		X_train, y_train = X[tr,:], y[tr]
		X_test, y_test = X[te,:], y[te]
		
		if mode == "CNNC_Name":
			model = kkeras.CNNC_Name( param_d["n_cv_flt"], param_d["n_cv_ln"], param_d["cv_activation"],
					l=l)
			n_flt = param_d["n_cv_flt"]
		elif mode == "CNNC_Name_Border":
			model = kkeras.CNNC_Name_Border( param_d["n_cv_flt"], param_d["n_cv_ln"], param_d["cv_activation"],
					l=l, border_mode = 'valid')
			n_flt = param_d["n_cv_flt"]           
		elif mode == "MLPC_Name":
			model = kkeras.MLPC_Name(l=l)
			n_flt = l[1]
		else:
			raise ValueError("The given mode is not supported: mode={}".format(mode))
			
		model.fit( X_train, y_train, X_test, y_test, nb_classes, batch_size=6, nb_epoch = 5)
		tr_score = model.score( X_train, y_train)
		#dcnn_score_l.append(dcnn_score)
		print( "Training Accuracy:", tr_score)

		dcnn_score = model.score( X_test, y_test)
		dcnn_score_l.append(dcnn_score)
		print( "Testing Accuracy:", dcnn_score)
		
		c_w, c_b = model.get_c_wb()
		print( "c_w.shape=", c_w.shape)
		c_w = c_w.reshape(-1, n_flt)
		c_wb_l.append( (c_w, c_b))

		y_cv[ te] = model.predict( X_test)

	print( dcnn_score_l)
	print( "Mean:{0}, Std:{1}".format( np.mean( dcnn_score_l), np.std( dcnn_score_l)))

	if graph:
		# One of weight vectors are drawn.
		c_w = c_wb_l[0][0] # 0 = c_w, 1 = c_b
		for ll in range(n_flt):
			#plt.figure()
			plt.plot( c_w[:,ll], label="Filter #{}".format(ll))
		plt.legend()  
	
	return dcnn_score_l, c_wb_l, y_cv	

def cv_deep_learning_pred_loss( X, y, nb_classes = 3, mode = "CNNC_Name", 
							 l = None, param_d = None, mp = 1,
							 graph = True, n_splits = 5):
	"""
	l =[X.shape[1], 3, 30, nb_classes]
	param_d = {"n_cv_flt": 3, "n_cv_ln": 50, "cv_activation": "relu"}
	"""
	
	KF = model_selection.KFold(n_splits, shuffle=True)
	kf = KF.split(X)

	dcnn_score_l, dcnn_loss_l = [], []
	c_wb_l = []
	y_cv = np.copy( y)
	for tr, te in kf:
		X_train, y_train = X[tr,:], y[tr]
		X_test, y_test = X[te,:], y[te]
		
		if mode == "CNNC_Name":
			model = kkeras.CNNC_Name( param_d["n_cv_flt"], param_d["n_cv_ln"],
				param_d["cv_activation"], l=l, mp=mp)
			n_flt = param_d["n_cv_flt"]
		elif mode == "CNNC_Name_Border":
			model = kkeras.CNNC_Name_Border( param_d["n_cv_flt"], param_d["n_cv_ln"], 
				param_d["cv_activation"], l=l, border_mode = 'valid')
			n_flt = param_d["n_cv_flt"]           
		elif mode == "MLPC_Name":
			model = kkeras.MLPC_Name(l=l)
			n_flt = l[1]
		else:
			raise ValueError("The given mode is not supported: mode={}".format(mode))
			
		model.fit( X_train, y_train, X_test, y_test, nb_classes, batch_size=6, nb_epoch = 5)
		tr_score_lossacc = model.evaluate(  X_train, y_train)
		tr_score = tr_score_lossacc[1]
		tr_loss = tr_score_lossacc[0]
		print( "Training Accuracy:", tr_score)
		print( "Training Loss:", tr_loss)

		dcnn_score_lossacc = model.evaluate( X_test, y_test)
		dcnn_score_l.append(dcnn_score_lossacc[1])
		dcnn_loss_l.append(dcnn_score_lossacc[0])	
		print( "Testing Accuracy:", dcnn_score_lossacc[1])
		print( "Testing Loss:", dcnn_score_lossacc[0])
		
		c_w, c_b = model.get_c_wb()
		print( "c_w.shape=", c_w.shape)
		c_w = c_w.reshape(-1, n_flt)
		c_wb_l.append( (c_w, c_b))

		y_cv[ te] = model.predict( X_test)

	print( dcnn_score_l)
	print( "[Accuracy] Mean:{0}, Std:{1}".format( np.mean( dcnn_score_l), np.std( dcnn_score_l)))
	print( dcnn_loss_l)
	print( "[Loss] Mean:{0}, Std:{1}".format( np.mean( dcnn_loss_l), np.std( dcnn_loss_l)))

	if graph:
		# One of weight vectors are drawn.
		c_w = c_wb_l[0][0] # 0 = c_w, 1 = c_b
		for ll in range(n_flt):
			#plt.figure()
			plt.plot( c_w[:,ll], label="Filter #{}".format(ll))
		plt.legend()  
	
	return dcnn_score_l, c_wb_l, y_cv, dcnn_loss_l		


def cv_deep_learning_tsplot( X, y, nb_classes = 3, mode = "CNNC_Name", 
							 l = None, param_d = None, n_ensemble = 1, graph = True):
	"""
	l =[X.shape[1], 3, 30, nb_classes]
	param_d = {"n_cv_flt": 3, "n_cv_ln": 50, "cv_activation": "relu"}
	"""
	ci = 95
	dcnn_score_l = []
	c_wb_l = []
	c_w_l = []
	
	for n_e in range( n_ensemble):
		print("Index of Ensemble Iteration:", n_e)
		KF = model_selection.KFold(5, shuffle=True)
		kf = KF.split(X)

		for tr, te in kf:
			X_train, y_train = X[tr,:], y[tr]
			X_test, y_test = X[te,:], y[te]
			
			if mode == "CNNC_Name":
				model = kkeras.CNNC_Name( param_d["n_cv_flt"], param_d["n_cv_ln"], param_d["cv_activation"],
						l=l)
				n_flt = param_d["n_cv_flt"]
			elif mode == "CNNC_Name_Border":
				model = kkeras.CNNC_Name_Border( param_d["n_cv_flt"], param_d["n_cv_ln"], param_d["cv_activation"],
						l=l, border_mode = 'valid')
				n_flt = param_d["n_cv_flt"]           
			elif mode == "MLPC_Name":
				model = kkeras.MLPC_Name(l=l)
				n_flt = l[1]
			else:
				raise ValueError("The given mode is not supported: mode={}".format(mode))
				
			model.fit( X_train, y_train, X_test, y_test, nb_classes, batch_size=6, nb_epoch = 5)
			dcnn_score = model.score( X_test, y_test)
			dcnn_score_l.append(dcnn_score)
			print( "Accuracy:", dcnn_score)
			
			c_w, c_b = model.get_c_wb()
			print( "c_w.shape=", c_w.shape)
			c_w = c_w.reshape(-1, n_flt)
			c_wb_l.append( (c_w, c_b))
			c_w_l.append( c_w)

	print( dcnn_score_l)
	print( "Mean:{0}, Std:{1}".format( np.mean( dcnn_score_l), np.std( dcnn_score_l)))

	if graph:
		c_w_a = np.array( c_w_l)
		print( c_w_a.shape)
		for i in range( c_w_a.shape[2]):
			plt.subplot( 1, c_w_a.shape[2], i+1)
			c_w_i = c_w_a[:,:,i]
			sns.tsplot( c_w_i, ci = ci)
			if i == 0:
				plt.ylabel("Maganitue")
			plt.xlabel('Time')
			plt.title('Filter#{}'.format(i))
			#plt.title( '{0} with ci={1}%'.format(mode, ci))
			#if i < c_w_a.shape[2] - 1:
			#	plt.xticks([])
			#else:
	
	return dcnn_score_l