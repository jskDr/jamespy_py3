import matplotlib.pyplot as plt
import numpy as np


def cimshow(Xc, mode='absangle', colorbar_flag=False):
    def cb():
        if colorbar_flag:
            plt.colorbar()

    if mode == 'absangle':
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.imshow(np.abs(Xc))
        plt.title('Magnitude')
        cb()
        plt.subplot(1,2,2)
        plt.imshow(np.angle(Xc))
        plt.title('Phase')
        cb()
    elif mode == 'realimag':
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.imshow(np.real(Xc))
        plt.title('Re')
        cb()
        plt.subplot(1,2,2)
        plt.imshow(np.imag(Xc))
        plt.title('Im')
        cb()
    else:
        raise ValueError('Mode of {} is not supported!'.format(mode))     
