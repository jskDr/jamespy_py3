# kcell.py
# python3

import pandas as pd
from sklearn import cross_validation, svm, metrics, cluster, tree, ensemble
import kkeras


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

	model = ensemble.RandomForestClassifier( n_estimators=10)
	model.fit( X_train, y_train)
	rf_score = model.score( X_test, y_test)
	print( "RF:", rf_score)

	return dt_score, sv_score, mlp_score, rf_score


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
	df_i["Classification method"] = [ "DT", "SVC", "DNN", "RF"]
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
