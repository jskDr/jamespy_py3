from sklearn.decomposition import TruncatedSVD
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
import numpy as np

class CVGP:
    def __init__(self, kernel_type=1):
        self.kernel_type = kernel_type
    
    def cv_gp_kernel(self, kernel, n, cv=5):
        X = self.X
        y = self.y
        Xn = TruncatedSVD(n).fit_transform(X)
        cv = cross_val_score(GaussianProcessClassifier(kernel=kernel), Xn, y, cv=cv)
        return cv

    def cv_gp(self, n, cv=5):
        kernel_type = self.kernel_type
        if kernel_type == 1:
            kernel = 1.0 * RBF([1.0])
        elif kernel_type == 2:
            kernel = 1.0 * RBF([1.0]*n)
        else:
            raise ValueError('The kernel_type is not support {}'.format(kernel_type))
        return self.cv_gp_kernel(kernel, n, cv=cv)

    def run(self, X, y, rg=range(2,5)):
        self.X = X
        self.y = y
        acc_ll = [] 
        for n in rg:
            acc = self.cv_gp(n)
            print(n, acc, np.mean(acc), np.std(acc))
            acc_ll.append(acc)
        return acc_ll