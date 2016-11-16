
"""
knumpy.py
Collection of my numpy array related fuctions.
"""

import numpy as np

def list_swap(y_id, a_l, b_l):
    """
    Swapping array values
    """
    y_a = np.array(y_id)
    x_a = np.copy(y_a)
    for a, b in zip(a_l, b_l):
        x_a[y_a == a] = b
    return x_a

def print_scalar(xorg_a, xnew_a, i=0):
    """
    Comparing statistics of Xorg and X: mean, std
    For each column (i)
    """
    s_l = [np.std(xorg_a[:, i]), np.std(xnew_a[:, i]), np.mean(xorg_a[:, i]), np.mean(xnew_a[:, i])]
    print("Std: {0:.2f} -> {1:.2f}, Mean: {2:.2f} -> {3:.2f}".format(*s_l))
