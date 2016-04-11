#james sklearn utility

import matplotlib.pyplot as plt
import numpy as np

def plot_ic_criterion(model, name, color):
	"""
	Take from plot_lasso_model_selection.py in
	http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html
	"""
	alpha_ = model.alpha_
	alphas_ = model.alphas_
	criterion_ = model.criterion_
	plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
		linewidth=3, label='%s criterion' % name)
	plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
		label='alpha: %s estimate' % name)
	plt.xlabel('-log(alpha)')
	plt.ylabel('criterion')