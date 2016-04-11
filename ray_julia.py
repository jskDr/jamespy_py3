"""
Python codes to be loaded in Julia
"""

from __future__ import print_function


def slice3( A, axis, layer):
	
	assert axis >= 0 and axis < 3

	if axis == 0:
		a = A[ layer, : , :]
	elif axis == 1:
		a = A[ :, layer, :]
	else:
		a = A[ :, :, layer]

	return a