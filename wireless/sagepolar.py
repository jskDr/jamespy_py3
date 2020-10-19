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
    for N in [2,4,8]:
        show('N=', N)
        u = gen_u(N)
        show('U:', u)
        x = polar_transform(u)
        show('X:', x)
        ud, _ = decode_n(x)
        show('UD:', ud)


#============================================================
# NPolar 부호화 및 복호화
# 여기서는 AE는 구현하지 않고 Polar encoding으로 emulation함
#============================================================
def npolar_transform(u,N_P=4/1):
    u = list(u)
    # print(len(u), u)
    if len(u) == 1:
        x = u
    elif len(u) > N_P:
        # 처음이 1이고 갈수록 2, 4, 8이 된다.
        # N_P = N/P 값에 따라 AE 담당할 앞부분은 polar 변환하지 않고 뒤집기만 한다.
        u1 = u[0::2]
        u2 = u[1::2]
        x = npolar_transform(u1,N_P) + npolar_transform(u2,N_P)                
    else:
        u1 = u[0::2]
        u2 = u[1::2]
        u1u2 = []
        for u1_i, u2_i in zip(u1, u2):
            u1u2.append(u1_i + u2_i)
        x = npolar_transform(u1u2,N_P) + npolar_transform(u2,N_P)
    return x

def npolar_coding(N=8, P=1):
    """
    Input:
    P=1: AE 입력의 크기임. NPolar의 경우, P=N을 제외하고는 AE를 one-hot vector로 처리하지 않음.
    """    
    N_P = N // P
    #u = var('U',n=N)
    u = gen_u(N)
    y_polar = npolar_transform(u, N_P=N_P)
    #print('y_polar:', y_polar)
    x_polar = []    
    x_polar_idx = []
    ae_polar = y_polar.copy()
    idx = list(range(N))
    for i in range(N_P):
        y_polar_l = y_polar[i::N_P]
        idx_l = idx[i::N_P]
        x_polar.append(y_polar_l)
        x_polar_idx.append(idx_l)
        ae_polar_l = ae_coding_emul(x_polar[-1])
        for i, j in enumerate(x_polar_idx[-1]):
            ae_polar[j] = ae_polar_l[i]
        #print('x_polar:', x_polar)
    print("x_polar, x_polar_idx:", x_polar, x_polar_idx)
    return ae_polar

def bit_forward(x):
    """
    polar encoding에서 사용한 방법을 역으로 수행함.
    bit_reverse를 역으로 복원함. 이 방법은 디코딩에서도 내부적으로 사용되고 있음.
    Polar encoding: x = encoding(x[0::2]) + encoding(x[1::2]) if len(x) > 1
    """
    LN = len(x)
    if LN == 1:
        return x
    else:
        Half_LN = LN // 2
        y = x.copy()
        #print(x, y)
        y[0::2] = bit_forward(x[:Half_LN])
        y[1::2] = bit_forward(x[Half_LN:])
    return y

def ae_coding_emul(x):
    """
    AE가 해야할 일을 Polar로 emulation시킴
    Polar로 emulation시키기 위해서는 bit reverse 되어 있는걸 복원해야 함.
    """
    print('AE:', x)
    u = bit_forward(x)
    print('bit_forward:', u)
    x = polar_transform(u)
    return x

def test_npolar(N=8,P=4):
    N, P = 8, 4
    print(f'N={N}, P={P}')
    x = npolar_coding(N, P)
    show('X:', x)
    ud, _ = decode_n(x)
    show('UD:', ud)
