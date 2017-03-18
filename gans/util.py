"""
Utility codes used for GANs 
"""
from sklearn import cluster
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import linalg
import numpy as np

def show_km(y, 
			n=4, 
			c=['b', 'g', 'r', 'k'],
			title='KMeans Clustering'):
	km = cluster.KMeans(n)
	yi = km.fit_predict(y)
	#c = ['b', 'g', 'r', 'k']
	for i in range(4):
	    sns.tsplot(y[yi==i], color=c[i])
	plt.title(title)


def evd_r0(Vorg, 
		no_order=None, 
		no_samples=1000,
		output_l=None):
	"""
	if output_l is not None, outputs will be saved to output_l
	output_l = [generated_samples]
	"""
	n =  Vorg - np.mean(Vorg, axis=0)
	y0 =  np.mean(Vorg, axis=0)
	Rnn = np.dot(n.T, n) / n.shape[0]
	if no_order is not None:
		w, v = np.linalg.eig(Rnn)
		w, v = w[:no_order], v[:,:no_order]
		Rnn = np.dot(np.dot(v, np.diag(w)), v.T)
	new_n = np.random.randn(no_samples, Vorg.shape[1]) 
	y = y0.reshape(1,-1) + np.dot(new_n, linalg.sqrtm(Rnn))

	if output_l is not None:
		output_l.append(y)

def evd(Vorg, 
		no_order=None, 
		no_samples=1000):

	n =  Vorg - np.mean(Vorg, axis=0)
	y0 =  np.mean(Vorg, axis=0)
	Rnn = np.dot(n.T, n) / n.shape[0]
	if no_order is not None:
		w, v = np.linalg.eig(Rnn)
		w, v = w[:no_order], v[:,:no_order]
		Rnn = np.dot(np.dot(v, np.diag(w)), v.T)
		Rnn = np.abs(Rnn)
	new_n = np.random.randn(no_samples, Vorg.shape[1]) 
	y = y0.reshape(1,-1) + np.dot(new_n, linalg.sqrtm(Rnn))
	y = np.real(y)

	return y