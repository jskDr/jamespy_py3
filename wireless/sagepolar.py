from sage.all import *

def polar_transform(u):
    if len(u) == 1:
        x = u
    else:
        u1 = u[0::2]
        u2 = u[1::2]
        u1u2 = []
        for u1_i, u2_i in zip(u1, u2):
            u1u2.append(u1_i + u2_i)
        x = polar_transform(u1u2) + polar_transform(u2)
    return x

def test_polar_transform():
    u = list(var('U', n=8))
    x = polar_transform(u)
    show(x)