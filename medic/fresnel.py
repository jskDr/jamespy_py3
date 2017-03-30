"""
Fresenel Diffraction with a new approach
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

def f(x_um, y_um, z_mm=0.5, l_nm=405):
        return np.exp(1j * np.pi * (np.power(x_um, 2) + np.power(y_um,2)) / (l_nm * z_mm))

def cimshow(f_impulse):
    plt.figure(figsize=(7,5))
    plt.subplot(2,2,1)
    plt.imshow(np.real(f_impulse))
    plt.colorbar()
    plt.title('Re{}')
    
    plt.subplot(2,2,2)
    plt.imshow(np.imag(f_impulse))
    plt.colorbar()
    plt.title('Img{}')
    
    plt.subplot(2,2,2+1)
    plt.imshow(np.abs(f_impulse))
    plt.colorbar()
    plt.title('Magnitude')

    plt.subplot(2,2,2+2)
    plt.imshow(np.angle(f_impulse))
    plt.colorbar()
    plt.title('Phase')
    
def xy(MAX_x_um = 55, pixel_um=2.2, oversample_rate=4):
    N = int(MAX_x_um / (pixel_um / oversample_rate))
    x = np.dot(np.ones((N,1)), np.linspace(-MAX_x_um,MAX_x_um,N).reshape(1,-1))
    y = np.dot(np.linspace(-MAX_x_um,MAX_x_um,N).reshape(-1,1), np.ones((1,N)))
    return x, y

def u(x, y, alpha):
    out = np.zeros_like(x)
    out[(y>=-alpha/2)&(y<=alpha/2)&(x>=-alpha/2)&(x<=alpha/2)] = 1.0
    return out

def u_circle(x,y,radius):
    xy2 = np.power(x,2) + np.power(y,2)
    # Since x is already matrix for griding, out shape is copied just from x. 
    # If x is a vector, the shape of out should be redefined to have 2-D form. 
    out = np.zeros_like(x)
    out[xy2<=np.power(radius,2)] = 1.0
    return out