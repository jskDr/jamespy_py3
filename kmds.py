# Manifold methods
# sklern.manifold.X where X = MDS, etc.

from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
					 discriminant_analysis, random_projection)

#----------------------------------------------------------------------
# Scale and visualize the embedding vectors
def plot_embedding(X, y, title=None, digit=True):
	x_min, x_max = np.min(X, 0), np.max(X, 0)
	X = (X - x_min) / (x_max - x_min)

	plt.figure()
	ax = plt.subplot(111)
	if digit:
		for i in range(X.shape[0]):
			plt.text(X[i, 0], X[i, 1], str(y[i]),
					 color=plt.cm.Set1(y[i] / 10.),
					 fontdict={'weight': 'bold', 'size': 9})
	else:
		for i in range(X.shape[0]):
			plt.plot(X[i, 0], X[i, 1], 'o',
					 color=plt.cm.Set1(y[i] / 10.))

	plt.xticks([]), plt.yticks([])
	if title is not None:
		plt.title(title)

def plot_sparse_random_projection(X, y, random_state=42):
	"""
	Random 2D projection using a random unitary matrix
	"""
	n_components=2 # Because of 2D project, it is fixed to 2.
	print("Computing random projection")
	rp = random_projection.SparseRandomProjection(n_components=n_components, random_state=random_state)
	X_projected = rp.fit_transform(X)
	plot_embedding(X_projected, y, "Random Projection of the digits")

def plot_pca_projection(X, y):
	"""
	Projection on to the first 2 principal components
	"""
	print("Computing PCA projection")
	t0 = time()
	X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
	plot_embedding(X_pca, y,
				   "Principal Components projection of the digits (time %.2fs)" %
				   (time() - t0))

def plot_isomap(X, y, n_neighbors):
	"""
	Isomap projection of the digits dataset
	"""
	print("Computing Isomap embedding")
	t0 = time()
	X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
	print("Done.")
	plot_embedding(X_iso, y,
				   "Isomap projection of the digits (time %.2fs)" %
				   (time() - t0))

def plot_locally_linear_embedding(X, y, n_neighbors):
	"""
	Locally linear embedding of the digits dataset
	"""
	print("Computing LLE embedding")
	clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
										  method='standard')
	t0 = time()
	X_lle = clf.fit_transform(X)
	print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
	plot_embedding(X_lle, y,
				   "Locally Linear Embedding of the digits (time %.2fs)" %
				   (time() - t0))

def plot_mds(X, y, n_init=1, max_iter=100):
	"""
	MDS  embedding of the digits dataset
	"""
	print("Computing MDS embedding")
	clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
	t0 = time()
	X_mds = clf.fit_transform(X)
	print("Done. Stress: %f" % clf.stress_)
	plot_embedding(X_mds, y,
				   "MDS embedding of the digits (time %.2fs)" %
				   (time() - t0))

def plot_other_manifold(X, y, n_neighbors, n_estimators = 200, max_depth=5, random_state=0):
	#----------------------------------------------------------------------
	# Modified Locally linear embedding of the digits dataset
	print("Computing modified LLE embedding")
	clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
										  method='modified')
	t0 = time()
	X_mlle = clf.fit_transform(X)
	print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
	plot_embedding(X_mlle, y,
				   "Modified Locally Linear Embedding of the digits (time %.2fs)" %
				   (time() - t0))


	#----------------------------------------------------------------------
	# HLLE embedding of the digits dataset
	print("Computing Hessian LLE embedding")
	clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
										  method='hessian')
	t0 = time()
	X_hlle = clf.fit_transform(X)
	print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
	plot_embedding(X_hlle, y, 
				   "Hessian Locally Linear Embedding of the digits (time %.2fs)" %
				   (time() - t0))


	#----------------------------------------------------------------------
	# LTSA embedding of the digits dataset
	print("Computing LTSA embedding")
	clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
										  method='ltsa')
	t0 = time()
	X_ltsa = clf.fit_transform(X)
	print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
	plot_embedding(X_ltsa, y, 
				   "Local Tangent Space Alignment of the digits (time %.2fs)" %
				   (time() - t0))

	#----------------------------------------------------------------------
	# Random Trees embedding of the digits dataset
	print("Computing Totally Random Trees embedding")
	hasher = ensemble.RandomTreesEmbedding(n_estimators=n_estimators, random_state=random_state,
										   max_depth=max_depth)
	t0 = time()
	X_transformed = hasher.fit_transform(X)
	pca = decomposition.TruncatedSVD(n_components=2)
	X_reduced = pca.fit_transform(X_transformed)

	plot_embedding(X_reduced, y,
				   "Random forest embedding of the digits (time %.2fs)" %
				   (time() - t0))

	#----------------------------------------------------------------------
	# Spectral embedding of the digits dataset
	print("Computing Spectral embedding")
	embedder = manifold.SpectralEmbedding(n_components=2, random_state=random_state,
										  eigen_solver="arpack")
	t0 = time()
	X_se = embedder.fit_transform(X)

	plot_embedding(X_se, y,
				   "Spectral embedding of the digits (time %.2fs)" %
				   (time() - t0))

def plot_tSNE(X, y, random_state=0, digit = True):
	#----------------------------------------------------------------------
	# t-SNE embedding of the digits dataset
	print("Computing t-SNE embedding")
	tsne = manifold.TSNE(n_components=2, init='pca', random_state=random_state)
	t0 = time()
	X_tsne = tsne.fit_transform(X)

	plot_embedding(X_tsne, y,
	               "t-SNE embedding of the digits (time %.2fs)" %
	               (time() - t0), digit = digit)