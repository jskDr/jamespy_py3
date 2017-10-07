import h5py
import numpy as np
from scipy import io


loadmat = io.loadmat


def loadmat73(filepath):
    """
    INPUTS
    ------
    filepath: '/path/to/data.mat'

    OUTPUT
    ------
    dictionary
    """

    arrays = {}
    f = h5py.File(filepath)
    for k, v in f.items():
        arrays[k] = np.array(v)

    return arrays
