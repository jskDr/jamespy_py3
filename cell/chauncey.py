# chauncey.py
import pandas as pd
import numpy as np


def load_clustered_data(fname_base, n_clusters=4):
    """
    Load n_clusters fragment csv files.
    Merge them and generate an index file.

    Return
    ======
    Merged data, np_array, 2D
    Generated index, np_array, 1D

    Example
    =======
    fname_base = '../data/For_feature_classification_2/Matlab_f2_each_clustering/vasp4/group_pro_cl_vel_{}.csv'
    v_a, c_a = chauncey_load_clustered_data( fname_base)
    """
    v_l = []
    c_l = []
    for i in range(n_clusters):
        v = pd.read_csv(fname_base.format(i + 1), header=None)
        v_l.append(v.values)
        c_l.append([i] * v.shape[0])
    v_a = np.concatenate(v_l, axis=0)
    c_a = np.concatenate(c_l, axis=0)
    print(v_a.shape, c_a.shape)

    return v_a, c_a
