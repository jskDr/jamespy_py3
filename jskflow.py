# Py3
# james files for skflow
# import tensorflow.contrib.learn as skflow

from sklearn import metrics
import kutil

def eval_score( model, X_test, y_test, string = "Test", graph = False):
	print()
	print(  "Evaluation of", string)
	print('--------')
	yP = model.predict(X_test)
	score_r2 = metrics.r2_score(y_test, yP)
	score_MedAE = metrics.median_absolute_error(y_test, yP)
	print('Accuracy')
	print('R2: {0:f}, MedAE: {1:f}'.format(score_r2, score_MedAE))
	print()
	
	if graph:
		kutil.regress_show4( y_test, yP)