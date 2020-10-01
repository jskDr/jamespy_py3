from sage.all import *

def polar_transform(u):
    u = list(u)
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

def test_polar_transform_n(n_all=[2,4,8]):
    for n in n_all:
        u = var('U',n=n)
        x = polar_transform(u)
        print('u:', u)
        print('x:', x)
        pretty_print(x)

def polar_transform_bitindex(u, bit_index):
    """
    bit_index = 0, 1, ..., len(u)-1 for original
    """
    u = list(u)
    if len(u) == 1:
        x = u
        b = bit_index
    else:
        u1 = u[0::2]
        u2 = u[1::2]
        b1 = bit_index[0::2]
        b2 = bit_index[1::2]        
        u1u2 = []
        for u1_i, u2_i in zip(u1, u2):
            u1u2.append(u1_i + u2_i)
        x1, b1 = polar_transform_bitindex(u1u2, b1)
        x2, b2 = polar_transform_bitindex(u2, b2)
        x = x1 + x2
        b = b1 + b2
    return x, b        

def test_polar_transform_bitindex():
    u = list(var('U', n=8))
    x, b = polar_transform_bitindex(u, list(range(len(u))))
    show(x)
    show(b)


##########################
# Decoding part
##########################
def decode_n(y_array):
    y1_array = y_array[0::2]
    y2_array = y_array[1::2]    
    
    if len(y_array) == 1:
        ud_array = y_array
        xd_array = ud_array
    else:
        l1_array = []
        for y1, y2 in zip(y1_array, y2_array):
            l1 = y1 + y2
            l1_array.append(l1)
        ud1_array, xd1_array = decode_n(l1_array)

        l2_array = []
        for y1, y2, xd1 in zip(y1_array, y2_array, xd1_array):
            if xd1 + y1 == y2:
                l2 = y2
            else:
                l2 = y1 # make error
            l2_array.append(l2)
        ud2_array, xd2_array = decode_n(l2_array)

        ud_array = ud1_array + ud2_array         
        
        for i in range(len(xd1_array)):
            xd1_array[i]= xd1_array[i]+ xd2_array[i]
        L2 = len(xd1_array)
        xd_array = [0] * L2 * 2
        xd_array[0::2] = xd1_array
        xd_array[1::2] = xd2_array
    return ud_array, xd_array

def gen_u(N=2):
    u_str = ""
    for i in range(N):
        if i == N - 1:
            u_str = u_str + f'U{i}'
        else:
            u_str = u_str + f'U{i}' + ','
        
    u = polygen(IntegerModRing(2), u_str)
    return u    

def test_decode_n():
    u = gen_u(2)
    x = polar_transform(u)
    show('N=2')
    show(x)
    show(decode_n(x))

    u = gen_u(4)
    x = polar_transform(u)
    show('N=4')
    show(x)
    show(decode_n(x))

    #R.<U0, U1, U2, U3, U4, U5, U6, U7> = IntegerModRing(2)[]
    #u = [U0, U1, U2, U3, U4, U5, U6, U7]
    u = gen_u(8)
    x = polar_transform(u)
    show('N=8')
    show(x)
    show(decode_n(x))