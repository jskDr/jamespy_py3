import numpy as np
import numba as nb

# Imitate static variable for a python function using decorate and setattr
def static_vars(**kwargs):
    '''
    @static_vars(counter=0)
    def foo():
        foo.counter += 1
        print("Counter is %d" % foo.counter)
    '''
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@nb.jit
def encode(u1, u2):
    return (u1 + u2) % 2, u2

@nb.jit
def f_neg(a, b):
    return np.log((np.exp(a + b) + 1)/(np.exp(a)+np.exp(b)))

@nb.jit
def f_pos(a, b, u):
    return (-1)**u*a + b

@nb.jit
def decode(y1, y2):
    l1 = f_neg(y1, y2)
    u1_hard = 0 if l1 > 0 else 1
    l2 = f_pos(y1, y2, u1_hard)
    u2_hard = 0 if l2 > 0 else 1
    return u1_hard, u2_hard, l1, l2

@nb.jit
def channel(x1, x2):
    y1 = 1.0 - x1*2
    y2 = 1.0 - x2*2
    return y1, y2

@nb.jit
def coding(u1, u2):
    x1, x2 = encode(u1, u2)
    y1, y2 = channel(x1, x2)
    u1_, u2_, l1, l2 = decode(y1, y2)
    e1, e2 = u1 - u1_, u2 - u2_
    return e1, e2

@nb.jit
def coding_array(u_array, e_array):
    for i in range(len(u_array)):
        e_array[i] = u_array[i] - u_array[i]