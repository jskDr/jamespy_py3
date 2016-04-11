"""
> using PyCall
> pyinitialize("python3")

For reload - jjulia can be changed for appropriate type
> jjulia = pyimport(:jjulia)

# This can be used for import and reload both.
# If @pyimport is used, reload is not workinng since lib names are assigned as constant variales. 
> jjulia = pywrap(PyObject(ccall(pysym(:PyImport_ReloadModule), PyPtr, (PyPtr,), pyimport("jjulia"))))
"""
import numpy as np
from sklearn import linear_model

import julia

def hello():
	j = julia.Julia()
	j.println("Hello")

def py_sumN( N):
	zz = 0
	for ii in range( N):
		for jj in range( N):
			for kk in range( N):
				zz += ii * jj * kk
	return zz


def hello( name):
	print("hi {}".format( name))
	print("This is James")
	print("Julia calls a python function.")
	print("Now reload module is working.")

def _regression_r0( X, y):
	print X
	print X.shape
	print y
	print y.shape

	xM = np.mat( X)
	yV = np.mat( y).T

	w = np.linalg.pinv( xM) * yV

	return np.array(w)

def regression(X, y):
	clf = linear_model.LinearRegression()
	clf.fit(X, y)
	return clf.coef_

"""
The following functions call Julia while them can be used in Python.
"""
jl = julia.Julia()

jl.eval('include("/home/jamessungjinkim/Dropbox/Aspuru-Guzik/julia_lab/jslib.jl")')
mg_to_log_mol = jl.eval( 'mg_to_log_mol')

julia_sum = jl.eval("""
function mysum(a,b)
	return a+b
	end
""")

julia_sumN = jl.eval("""
function mysum(N)
	zz = 0
	for ii = 1:N
		for jj = 1:N
			for kk = 1:N
				zz += ii*jj*kk
			end
		end
	end
	return zz
end
""")
