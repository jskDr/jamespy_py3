# kcell.py
# python3

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import model_selection, svm, metrics, cluster, tree
import kkeras
import numpy as np
import seaborn as sns

def GET_clsf2_by_clst( nb_classes):
	def clsf2_by_clst( Xpart_cf, Xpart_ct):
		"""
		Clustering is performed and then, classification performed by clustered indices. 
		"""
		cl_model = cluster.KMeans(n_clusters=nb_classes)
		cl_model.fit(Xpart_ct)
		yint = cl_model.predict( Xpart_ct)

		X_train, X_test, y_train, y_test = 
			model_selection.train_test_split( Xpart_cf, yint, test_size = 0.2)

		model = tree.DecisionTreeClassifier()
		model.fit( X_train, y_train)
		dt_score = model.score( X_test, y_test)
		print( "DT-C:", dt_score)

		model = svm.SVC( kernel = 'linear')
		model.fit( X_train, y_train)
		sv_score = model.score( X_test, y_test)
		print( "SVC:", sv_score)

		model = kkeras.MLPC( [Xpart_cf.shape[1], 30, 10, nb_classes])
		model.fit( X_train, y_train, X_test, y_test, nb_classes)
		mlp_score = model.score( X_test, y_test)
		print( "MLP:", mlp_score)

		return dt_score, sv_score, mlp_score

	return clsf2_by_clst

def GET_clsf2_by_yint( nb_classes):
	def clsf2_by_yint( X1part, yint):
		"""
		classification is performed by yint
		"""
		X_train, X_test, y_train, y_test = 
			model_selection.train_test_split( X1part, yint, test_size = 0.2)

		model = tree.DecisionTreeClassifier()
		model.fit( X_train, y_train)
		dt_score = model.score( X_test, y_test)
		print( "DT:", dt_score)

		model = svm.SVC( kernel = 'linear')
		model.fit( X_train, y_train)
		sv_score = model.score( X_test, y_test)
		print( "SVC:", sv_score)

		model = kkeras.MLPC( [X1part.shape[1], 30, 10, nb_classes])
		model.fit( X_train, y_train, X_test, y_test, nb_classes)
		mlp_score = model.score( X_test, y_test)
		print( "MLP:", mlp_score)

		return dt_score, sv_score, mlp_score

	return clsf2_by_yint

def pd_clsf2_by_clst( ix, Xpart_ct, Xpart_cf, nb_classes):
	VI = {1:"Velocity", 2:"Intensity", 12:"Combined"}

	print( "Type", ix, "- Clustering:", ix[1], "Classification:", ix[0])
	s_l = GET_clsf2_by_clst(nb_classes)(Xpart_cf, Xpart_ct)
	df_i = pd.DataFrame()
	df_i["Type"] = ["KMenas: " + str( ix)] * 3
	df_i["Clustering"] = [ VI[ix[0]]] * 3
	df_i["Classification"] = [ VI[ix[1]]] * 3
	df_i["Clustering method"] = [ "KMeans"] * 3
	df_i["Classification method"] = [ "DT", "SVC", "DNN"]
	df_i["Pc"] = s_l

	return df_i

def pd_clsf2_by_yint( ix, yint, Xpart_cf, nb_classes):
	VI = {1:"Velocity", 2:"Intensity", 12:"Combined"}

	print( "Type", ix, "- Clustering:", ix[1], "Classification:", ix[0])
	s_l = GET_clsf2_by_yint(nb_classes)(Xpart_cf, yint)
	df_i = pd.DataFrame()
	df_i["Type"] = ["Science: "+str( ix)] * 3
	df_i["Clustering"] = [ VI[ix[0]]] * 3
	df_i["Classification"] = [ VI[ix[1]]] * 3
	df_i["Clustering method"] = [ "Sceince method"] * 3
	df_i["Classification method"] = [ "DT", "SVC", "DNN"]
	df_i["Pc"] = s_l

	return df_i

class _Subclustering_r0():
	def __init__(self, X1part, X2part, y, cell, 
			X1_ylim = [-1.5, 1.5], X2_ylim = [-2, 4], 
			cmethod = "KMenas", 
			cparam_d = {"n_clusters": 2}):

		self.X1part = X1part
		self.X2part = X2part
		self.y = y
		self.cell = cell
		self.X1_ylim = X1_ylim
		self.X2_ylim = X2_ylim
		self.cmethod = cmethod
		self.cparam_d = cparam_d

	def show_both( self, c):
		X1part = self.X1part
		X2part = self.X2part
		y = self.y
		cell = self.cell
		X1_ylim = self.X1_ylim
		X2_ylim = self.X2_ylim
		cmethod = self.cmethod
		cparam_d = self.cparam_d

		#print("Cluster:", c)
		X3_int = X2part[ np.where(y==c)[0],:]
		X3_vel = X1part[ np.where(y==c)[0],:]
		
		#km = cluster.KMeans(2)
		#km = getattr(cluster, cmethod)(2)
		km = getattr(cluster, cmethod)(**cparam_d)
		y3 = km.fit_predict( X3_int)

		plt.figure(figsize=(9,4))
		plt.subplot(1,2,1)
		#print("Intensity")
		n_0 = X3_int[ np.where( y3==0)[0]].shape[0]
		n_1 = X3_int[ np.where( y3==1)[0]].shape[0]

		sns.tsplot( X3_int[ np.where( y3==0)[0],:], color="blue")
		sns.tsplot( X3_int[ np.where( y3==1)[0],:], color="green")
		plt.ylim(X2_ylim)
		plt.title("Cluster{0}:X2 {1}:{2}".format(c, n_0, n_1))
		#plt.show()

		plt.subplot(1,2,2)
		#print("Velocity")
		sns.tsplot( X3_vel[ np.where( y3==0)[0],:], color="blue")
		sns.tsplot( X3_vel[ np.where( y3==1)[0],:], color="green")
		plt.ylim(X1_ylim)
		plt.title("Cluster{0}:X1 {1}:{2}".format(c, n_0, n_1))
		plt.show()
		
		cell3 = cell[ np.where(y==c)[0]]
		plt.subplot(1,2,1)
		plt.stem( cell3[np.where( y3==0)[0]], linefmt='b-', markerfmt='bo')
		plt.title("Cell Index - Subcluster 1")
		plt.subplot(1,2,2)
		plt.stem( cell3[np.where( y3==1)[0]], linefmt='g-', markerfmt='go')   
		plt.title("Cell Index - Subcluster 2")
		plt.show()
		
		return y3
	
	def show_both_cell( self, c, cell_id):
		X1part = self.X1part
		X2part = self.X2part
		y = self.y
		cell = self.cell
		X1_ylim = self.X1_ylim
		X2_ylim = self.X2_ylim
		cmethod = self.cmethod

		X3_int = X2part[ np.where(y==c)[0],:]
		X3_vel = X1part[ np.where(y==c)[0],:]
		cell3 = cell[ np.where(y==c)[0]]
		
		#km = cluster.KMeans(2)
		#km = getattr(cluster, cmethod)(2)
		km = getattr(cluster, cmethod)(**cparam_d)		
		y3 = km.fit_predict( X3_int)

		# redefine based on cell_id
		X3_int = X3_int[ np.where(cell3==cell_id)[0],:]
		X3_vel = X3_vel[ np.where(cell3==cell_id)[0],:]   
		y3 = y3[np.where(cell3==cell_id)[0]]
		
		n_0 = X3_int[ np.where( y3==0)[0]].shape[0]
		n_1 = X3_int[ np.where( y3==1)[0]].shape[0]

		plt.figure(figsize=(9,4))
		plt.subplot(1,2,1)
		if n_0 > 0: sns.tsplot( X3_int[ np.where( y3==0)[0],:], color="blue")
		if n_1 > 0: sns.tsplot( X3_int[ np.where( y3==1)[0],:], color="green")
		plt.ylim(X2_ylim)
		plt.title("Cluster{0}:Intensity {1}:{2}".format(c, n_0, n_1))
		#plt.show()

		plt.subplot(1,2,2)
		#print("Velocity")
		if n_0 > 0: sns.tsplot( X3_vel[ np.where( y3==0)[0],:], color="blue")
		if n_1 > 0: sns.tsplot( X3_vel[ np.where( y3==1)[0],:], color="green")
		plt.ylim(X1_ylim)
		plt.title("Cluster{0}:Velocity {1}:{2}".format(c, n_0, n_1))
		plt.show()

class Subclustering():
	def __init__(self, X1part, X2part, y, cell, 
			X1_ylim = [-1.5, 1.5], X2_ylim = [-2, 4], 
			cmethod = "KMenas", 
			cparam_d = {"n_clusters": 2}):

		self.X1part = X1part
		self.X2part = X2part
		self.y = y
		self.cell = cell
		self.X1_ylim = X1_ylim
		self.X2_ylim = X2_ylim
		self.cmethod = cmethod
		self.cparam_d = cparam_d

	def show_both( self, c):
		X1part = self.X1part
		X2part = self.X2part
		y = self.y
		cell = self.cell
		X1_ylim = self.X1_ylim
		X2_ylim = self.X2_ylim
		cmethod = self.cmethod
		cparam_d = self.cparam_d

		#print("Cluster:", c)
		X3_int = X2part[ np.where(y==c)[0],:]
		X3_vel = X1part[ np.where(y==c)[0],:]
		
		#km = cluster.KMeans(2)
		#km = getattr(cluster, cmethod)(2)
		km = getattr(cluster, cmethod)(**cparam_d)
		y3 = km.fit_predict( X3_int)

		plt.figure(figsize=(9,4))
		plt.subplot(1,2,1)
		#print("Intensity")
		n_0 = X3_int[ np.where( y3==0)[0]].shape[0]
		n_1 = X3_int[ np.where( y3==1)[0]].shape[0]

		sns.tsplot( X3_int[ np.where( y3==0)[0],:], color="blue")
		sns.tsplot( X3_int[ np.where( y3==1)[0],:], color="green")
		plt.ylim(X2_ylim)
		plt.title("Cluster{0}:X2 {1}:{2}".format(c, n_0, n_1))
		#plt.show()

		plt.subplot(1,2,2)
		#print("Velocity")
		sns.tsplot( X3_vel[ np.where( y3==0)[0],:], color="blue")
		sns.tsplot( X3_vel[ np.where( y3==1)[0],:], color="green")
		plt.ylim(X1_ylim)
		plt.title("Cluster{0}:X1 {1}:{2}".format(c, n_0, n_1))
		plt.show()
		
		cell3 = cell[ np.where(y==c)[0]]
		plt.subplot(1,2,1)
		plt.stem( cell3[np.where( y3==0)[0]], linefmt='b-', markerfmt='bo')
		plt.title("Cell Index - Subcluster 1")
		plt.subplot(1,2,2)
		plt.stem( cell3[np.where( y3==1)[0]], linefmt='g-', markerfmt='go')   
		plt.title("Cell Index - Subcluster 2")
		plt.show()
		
		return y3
	
	def show_both_cell( self, c, cell_id):
		X1part = self.X1part
		X2part = self.X2part
		y = self.y
		cell = self.cell
		X1_ylim = self.X1_ylim
		X2_ylim = self.X2_ylim
		cmethod = self.cmethod

		X3_int = X2part[ np.where(y==c)[0],:]
		X3_vel = X1part[ np.where(y==c)[0],:]
		cell3 = cell[ np.where(y==c)[0]]
		
		km = getattr(cluster, cmethod)(**cparam_d)		
		y3 = km.fit_predict( X3_int)

		# redefine based on cell_id
		X3_int = X3_int[ np.where(cell3==cell_id)[0],:]
		X3_vel = X3_vel[ np.where(cell3==cell_id)[0],:]   
		y3 = y3[np.where(cell3==cell_id)[0]]
		
		n_0 = X3_int[ np.where( y3==0)[0]].shape[0]
		n_1 = X3_int[ np.where( y3==1)[0]].shape[0]

		plt.figure(figsize=(9,4))
		plt.subplot(1,2,1)
		if n_0 > 0: sns.tsplot( X3_int[ np.where( y3==0)[0],:], color="blue")
		if n_1 > 0: sns.tsplot( X3_int[ np.where( y3==1)[0],:], color="green")
		plt.ylim(X2_ylim)
		plt.title("Cluster{0}:Intensity {1}:{2}".format(c, n_0, n_1))
		#plt.show()

		plt.subplot(1,2,2)
		#print("Velocity")
		if n_0 > 0: sns.tsplot( X3_vel[ np.where( y3==0)[0],:], color="blue")
		if n_1 > 0: sns.tsplot( X3_vel[ np.where( y3==1)[0],:], color="green")
		plt.ylim(X1_ylim)
		plt.title("Cluster{0}:Velocity {1}:{2}".format(c, n_0, n_1))
		plt.show()

	def show_both_kmeans( self, c):
		X1part = self.X1part
		X2part = self.X2part
		y = self.y
		cell = self.cell
		X1_ylim = self.X1_ylim
		X2_ylim = self.X2_ylim
		cmethod = self.cmethod
		cparam_d = self.cparam_d

		nc = cparam_d["n_clusters"]

		#print("Cluster:", c)
		X3_int = X2part[ y==c,:]
		X3_vel = X1part[ y==c,:]
		
		#km = cluster.KMeans(2)
		#km = getattr(cluster, cmethod)(2)
		assert cmethod == "KMeans"
		km = cluster.KMeans( nc)
		y3 = km.fit_predict( X3_int)

		plt.figure(figsize=(9,4))
		plt.subplot(1,2,1)
		#print("Intensity")
		n_l = [ X3_int[ y3==i].shape[0] for i in range(nc)]

		for i in range(nc):
			sns.tsplot( X3_int[ y3==i,:], color=plt.cm.rainbow(i/nc))
		plt.ylim(X2_ylim)
		plt.title("Cluster{0}:X2 {1}".format(c, n_l))
		#plt.show()

		plt.subplot(1,2,2)
		#print("Velocity")
		for i in range(nc):
			sns.tsplot( X3_vel[ y3==i,:], color=plt.cm.rainbow(i/nc))
		plt.ylim(X1_ylim)
		plt.title("Cluster{0}:X1 {1}".format(c, n_l))
		plt.show()

		return y3