#print(__doc__)
# My utility functions related to matplotlib.pyplot
import matplotlib.pyplot as plt

def plot_cdata( X, labels_true):
    """
    Ploting data with clustering information
    
    X, nd.bytearray
    2D Numpy arrays 

    labels_true, int array
    1D int array indicating clustering indices 
    """
    for l in range(max(labels_true)+1):
        Xl = X[labels_true==l,:]
        plt.plot( Xl[:,0], Xl[:,1], 'o', markerfacecolor=plt.cm.rainbow(l/max(labels_true)))

