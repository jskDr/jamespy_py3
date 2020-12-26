import numpy as np
import numba as nb
import matplotlib.pyplot as plt

def calc_ber(e_array):
    return np.mean(np.abs(e_array))

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
    u1_, u2_, _, _ = decode(y1, y2)
    e1, e2 = u1 - u1_, u2 - u2_
    return e1, e2


@nb.jit
def coding_array(u_array, e_array):
    for i in range(len(u_array)):
        e_array[i,0], e_array[i,1] = coding(u_array[i,0], u_array[i,1])        
        
def run_coding():
    u_array = np.array([(1,1), (1,0), (0,1), (0,0)])
    e_array = np.zeros_like(u_array)
    coding_array(u_array, e_array)
    print(e_array)        
        
@nb.jit
def channel_awgn(x1, x2, SNRdB):
    SNR = np.power(10, SNRdB/10)
    Nsig = 1/np.sqrt(SNR)
    n1 = np.random.normal(0) * Nsig
    n2 = np.random.normal(0) * Nsig
    y1 = 1.0 - x1*2 + n1
    y2 = 1.0 - x2*2 + n2
    return y1, y2

@nb.jit
def coding_awgn(u1, u2, SNRdB):
    x1, x2 = encode(u1, u2)
    y1, y2 = channel_awgn(x1, x2, SNRdB)
    u1_, u2_, _, _ = decode(y1, y2)
    e1, e2 = u1 - u1_, u2 - u2_
    return e1, e2

@nb.jit
def coding_array_awgn(u_array, e_array, SNRdB):
    for i in range(len(u_array)):
        e_array[i,0], e_array[i,1] = coding_awgn(u_array[i,0], u_array[i,1], SNRdB)        
        
def run_coding_awgn(SNRdB=10):
    u_array = np.array([(1,1), (1,0), (0,1), (0,0)])
    e_array = np.zeros_like(u_array)
    coding_array_awgn(u_array, e_array, SNRdB)
    # print(e_array)
    BER = np.sum(np.abs(e_array)) / np.prod(e_array.shape)
    return BER

def main_run_coding_awgn(SNRdB_list=list(range(10))):
    BER_list = []
    for SNRdB in SNRdB_list:
        BER = run_coding_awgn(SNRdB)
        BER_list.append(BER)

    plt.semilogy(SNRdB_list, BER_list)
    plt.grid()
    plt.xlabel('SNR(dB)')
    plt.ylabel('BER')
    plt.title('Performance of Polar Code')
    plt.show()

def run_coding_awgn_tile(SNRdB=10, Ntile=1):
    u_array_unit = np.array([(1,1), (1,0), (0,1), (0,0)])
    u_array = np.tile(u_array_unit, (Ntile, 1))
    e_array = np.zeros_like(u_array)
    coding_array_awgn(u_array, e_array, SNRdB)
    BER = np.sum(np.abs(e_array)) / np.prod(e_array.shape)
    return BER

def main_run_coding_awgn_tile(SNRdB_list=list(range(10)), Ntile=1, flag_fig=False):
    BER_list = []
    for SNRdB in SNRdB_list:
        BER = run_coding_awgn_tile(SNRdB, Ntile)
        BER_list.append(BER)

    if flag_fig:
        plt.semilogy(SNRdB_list, BER_list)
        plt.grid()
        plt.xlabel('SNR(dB)')
        plt.ylabel('BER')
        plt.title('Performance of Polar Code')
        plt.show()

@nb.jit
def encode_array(u_array):
    x_array = np.zeros_like(u_array)
    for i in range(len(u_array)):
        x_array[i,0], x_array[i,1] = encode(u_array[i,0], u_array[i,1])        
    return x_array

@nb.jit
def channel_array(x_array):
    y_array = np.zeros(x_array.shape, dtype=nb.float_)
    for i in range(len(x_array)):
        y_array[i,0], y_array[i,1] = channel(x_array[i,0], x_array[i,1])        
    return y_array

@nb.jit
def decode_array(y_array):
    ud_array = np.zeros(y_array.shape, dtype=nb.int_)
    for i in range(len(y_array)):
        ud_array[i,0], ud_array[i,1], _, _ = decode(y_array[i,0], y_array[i,1])        
    return ud_array

@nb.jit
def coding_array_all(u_array):
    e_array = np.zeros_like(u_array)
    x_array = encode_array(u_array)
    y_array = channel_array(x_array)
    ud_array = decode_array(y_array)
    e_array = u_array - ud_array
    return e_array

def run_coding_array_all():
    u_array = np.array([(1,1), (1,0), (0,1), (0,0)])
    e_array = coding_array_all(u_array)
    print(e_array)

@nb.jit
def channel_array_awgn(x_array, SNRdB):
    y_array = np.zeros(x_array.shape, dtype=nb.float_)
    for i in range(len(x_array)):
        y_array[i,0], y_array[i,1] = channel_awgn(x_array[i,0], x_array[i,1], SNRdB)        
    return y_array

@nb.jit
def _coding_array_all_awgn(u_array, SNRdB=10):
    e_array = np.zeros_like(u_array)
    x_array = encode_array(u_array)
    y_array = channel_array_awgn(x_array, SNRdB)
    ud_array = decode_array(y_array)
    e_array = u_array - ud_array
    return e_array

def run_coding_array_all_awgn(SNRdB=10):
    u_array = np.array([(1,1), (1,0), (0,1), (0,0)])
    e_array = coding_array_all_awgn(u_array, SNRdB=SNRdB)
    print(e_array)

def run_coding_array_all_awgn_tile(SNRdB=10, Ntile=1):
    u_array_unit = np.array([(1,1), (1,0), (0,1), (0,0)])
    u_array = np.tile(u_array_unit, (Ntile, 1))
    e_array = coding_array_all_awgn(u_array, SNRdB=SNRdB)
    BER = np.sum(np.abs(e_array)) / np.prod(e_array.shape)
    # print(BER)
    return BER

def main_run_coding_array_all_awgn_tile(SNRdB_list=list(range(10)), Ntile=1, flag_fig=False):
    BER_list = []
    for SNRdB in SNRdB_list:
        BER = run_coding_array_all_awgn_tile(SNRdB, Ntile)
        BER_list.append(BER)

    if flag_fig:
        plt.semilogy(SNRdB_list, BER_list)
        plt.grid()
        plt.xlabel('SNR(dB)')
        plt.ylabel('BER')
        plt.title('Performance of Polar Code')
        plt.show()    

@nb.jit
def coding_array_all_awgn(u_array, SNRdB=10):
    e_array = np.zeros_like(u_array)
    x_array = encode_array(u_array)
    y_array = channel_numpy_awgn(x_array, SNRdB)
    ud_array = decode_array(y_array)
    e_array = u_array - ud_array
    return e_array

@nb.jit
def channel_numpy_awgn(x_array, SNRdB):
    """
    출력을 (0,1) --> (1,-1)로 바꾸고 가우시안 노이즈를 더함.
    """
    #y_array = np.zeros(x_array.shape, dtype=nb.float_)
    SNR = np.power(10, SNRdB/10)
    noise_sig = 1/np.sqrt(SNR)
    n_array = np.random.normal(0.0, noise_sig, size=x_array.shape)
    y_array = 1.0 - x_array*2 + n_array
    return y_array   


# Usage list
# main_run_coding_awgn()
# run_coding_awgn()
# run_coding()

# N >= 2 Polar coding (Generalized)

@nb.jit
def encode_n(u):
    """
    x = uBF(xn) where n = log(N), N=len(u), B is bit-reverse
    """
    x = np.copy(u)
    L = len(u)
    if L != 1:
        u1 = u[0::2]
        u2 = u[1::2]
        u1u2 = np.mod(u1 + u2, 2) 
        x[:L/2] = encode_n(u1u2)
        x[L/2:] = encode_n(u2)
    return x

@nb.jit
def encode_array_n(u_array):
    x_array = np.zeros_like(u_array)
    for i in range(len(u_array)):
        x_array[i] = encode_n(u_array[i])        
    return x_array

@nb.jit
def f_neg_n(a, b):
    return np.log((np.exp(a + b) + 1)/(np.exp(a)+np.exp(b)))

@nb.jit
def f_pos_n(a, b, u):
    return (-1)**u*a + b

@nb.jit
def decode_n_r0(y_array):
    """
    u_hard: input hard decision
    x_hard: output hard decision
    """
    u_hard = np.zeros(y_array.shape, dtype=nb.int_) 
    x_hard = np.zeros(y_array.shape, dtype=nb.int_) 
    L = len(y_array)
    if L == 1:
        u_hard[0] = 0 if y_array[0] > 0 else 1
        x_hard[0] = u_hard[0]
    else:
        y1 = y_array[0::2]
        y2 = y_array[1::2]    
        # print(L, y1, y2)
        
        l1 = f_neg_n(y1, y2)
        u_hard[:L/2], x_hard[:L/2] = decode_n(l1)
        # print('[:L/2] ', l1, u_hard[:L/2], x_hard[:L/2])
    
        l2 = f_pos_n(y1, y2, x_hard[:L/2])
        u_hard[L/2:], x_hard[L/2:] = decode_n(l2)
         
        x_hard[:L/2] = np.mod(x_hard[:L/2] + x_hard[L/2:], 2)

    return u_hard, x_hard


@nb.jit
def decode_n(y_array):
    """
    u_hard: input hard decision
    x_hard: output hard decision
    """
    u_hard = np.zeros(y_array.shape, dtype=nb.int_) 
    x_hard = np.zeros(y_array.shape, dtype=nb.int_) 
    x_temp = np.zeros(y_array.shape, dtype=nb.int_) 
    L = len(y_array)
    if L == 1:
        u_hard[0] = 0 if y_array[0] > 0 else 1
        x_hard[0] = u_hard[0]
    else:
        y1 = y_array[0::2]
        y2 = y_array[1::2]    
        # print(L, y1, y2)
        
        l1 = f_neg_n(y1, y2)
        u_hard[:L/2], x_temp[:L/2] = decode_n(l1)
        # print('[:L/2] ', l1, u_hard[:L/2], x_hard[:L/2])
    
        l2 = f_pos_n(y1, y2, x_temp[:L/2])
        u_hard[L/2:], x_temp[L/2:] = decode_n(l2)
         
        x_temp[:L/2] = np.mod(x_temp[:L/2] + x_temp[L/2:], 2)
        x_hard[0::2] = x_temp[:L/2]
        x_hard[1::2] = x_temp[L/2:]

    return u_hard, x_hard    

@nb.jit
def decode_array_n(y_array):
    ud_array = np.zeros(y_array.shape, dtype=nb.int_) #nb.int_)
    for i in range(len(y_array)):
        ud_array[i], _ = decode_n(y_array[i])      
    return ud_array

@nb.jit
def coding_array_all_awgn_n(u_array, SNRdB=10):
    e_array = np.zeros_like(u_array)
    x_array = encode_array_n(u_array)
    y_array = channel_numpy_awgn(x_array, SNRdB)  
    ud_array = decode_array_n(y_array)
    e_array = u_array - ud_array
    return e_array

class PolarCode:
    def __init__(self, N_code=2, K_code=2):
        """
        N_code: Code block size
        K_code: Information bit size
        """
        self.N_code = N_code
        self.K_code = K_code 

    def plot(self, SNRdB_list, BER_list):
        plt.semilogy(SNRdB_list, BER_list)
        plt.grid()
        plt.xlabel('SNR(dB)')
        plt.ylabel('BER')
        plt.title('Performance of Polar Code')
        plt.show()    

    def run(self, 
        SNRdB_list=list(range(10)), N_iter=1, flag_fig=False):
        u_array = np.random.randint(2, size=(N_iter, self.N_code))
        
        BER_list = []
        for SNRdB in SNRdB_list:
            e_array = coding_array_all_awgn_n(u_array, SNRdB=SNRdB)
            BER = np.sum(np.abs(e_array)) / np.prod(e_array.shape)
            BER_list.append(BER)

        if flag_fig:
            self.plot(SNRdB_list, BER_list)  

        self.BER_list = BER_list         

# ====================================================================
# Frozen을 고려하는 Polar Coding 시스템 
# ====================================================================
@nb.jit
def _decode_frozen_n(y_array, frozen_flag_n):
    """
    u_hard: input hard decision
    x_hard: output hard decision
    """
    u_hard = np.zeros(y_array.shape, dtype=nb.int_) 
    x_hard = np.zeros(y_array.shape, dtype=nb.int_) 
    x_temp = np.zeros(y_array.shape, dtype=nb.int_) 
    L = len(y_array)
    if L == 1:
        if frozen_flag_n[0]:
            u_hard[0] = 0
        else:
            u_hard[0] = 0 if y_array[0] > 0 else 1
        x_hard[0] = u_hard[0]
    else:
        y1 = y_array[0::2]
        y2 = y_array[1::2]    
        # print(L, y1, y2)
        
        l1 = f_neg_n(y1, y2)
        u_hard[:L/2], x_temp[:L/2] = decode_frozen_n(l1, frozen_flag_n[:L/2])
        # print('[:L/2] ', l1, u_hard[:L/2], x_hard[:L/2])
    
        l2 = f_pos_n(y1, y2, x_temp[:L/2])
        u_hard[L/2:], x_temp[L/2:] = decode_frozen_n(l2, frozen_flag_n[L/2:])
         
        x_temp[:L/2] = np.mod(x_temp[:L/2] + x_temp[L/2:], 2)
        x_hard[0::2] = x_temp[:L/2]
        x_hard[1::2] = x_temp[L/2:]

    return u_hard, x_hard    

@nb.jit
def decode_frozen_n(y_array, frozen_flag_n):
    """
    u_hard: input hard decision
    x_hard: output hard decision
    """
    u_hard = np.zeros(y_array.shape, dtype=nb.int_) 
    x_hard = np.zeros(y_array.shape, dtype=nb.int_) 
    x_temp = np.zeros(y_array.shape, dtype=nb.int_) 
    L = len(y_array)
    if L == 1:
        u_hard[0] = 0 if y_array[0] > 0 else 1
        if frozen_flag_n[0]:
            x_hard[0] = 0
        else:
            x_hard[0] = u_hard[0]
    else:
        y1 = y_array[0::2]
        y2 = y_array[1::2]    
        # print(L, y1, y2)
        
        l1 = f_neg_n(y1, y2)
        u_hard[:L/2], x_temp[:L/2] = decode_frozen_n(l1, frozen_flag_n[:L/2])
        # print('[:L/2] ', l1, u_hard[:L/2], x_hard[:L/2])
    
        l2 = f_pos_n(y1, y2, x_temp[:L/2])
        u_hard[L/2:], x_temp[L/2:] = decode_frozen_n(l2, frozen_flag_n[L/2:])
         
        x_temp[:L/2] = np.mod(x_temp[:L/2] + x_temp[L/2:], 2)
        x_hard[0::2] = x_temp[:L/2]
        x_hard[1::2] = x_temp[L/2:]

    return u_hard, x_hard  


@nb.jit
def decode_frozen_array_n(y_array, frozen_flag_n):
    ud_array = np.zeros(y_array.shape, dtype=nb.int_) 
    for i in range(len(y_array)):
        ud_array[i], _ = decode_frozen_n(y_array[i], frozen_flag_n)      
    return ud_array

@nb.jit
def frozen_encode_n(uf, u, f):    
    """
    Input:
    uf: 길이 N_code인 코딩 블럭
    u: 길이 K_code인 정보 블럭
    f: 길이 N_code인 비트가 frozen인지 아닌지를 나타내는 벡터
    """
    k = 0 
    for n in range(len(uf)):
        if f[n]:
            uf[n] = 0
        else: 
            uf[n] = u[k]        
            k += 1

@nb.jit
def frozen_encode_array_n(u_array, frozen_flag_n):    
    N_iter = len(u_array)
    N_code = len(frozen_flag_n)
    uf_array = np.zeros(shape=(N_iter,N_code), dtype=nb.int_)    
    for i in range(N_iter):
            frozen_encode_n(uf_array[i], u_array[i], frozen_flag_n)
    return uf_array  

@nb.jit
def frozen_decode_n(ud, ufd, f):    
    """
    Input:
    ufd: 길이 N_code인 디코딩한 블럭
    ud: 길이 K_code인 검출한 정보 블럭
    f: 길이 N_code인 비트가 frozen인지 아닌지를 나타내는 벡터
    """
    k = 0 
    for n in range(len(f)):
        if f[n] == 0:
            ud[k] = ufd[n]        
            k += 1

@nb.jit
def frozen_decode_array_n(ufd_array, frozen_flag_n):    
    N_iter = len(ufd_array)
    N_code = len(frozen_flag_n)
    K_code = N_code - np.sum(frozen_flag_n)
    ud_array = np.zeros(shape=(N_iter,K_code), dtype=nb.int_)    
    for i in range(N_iter):
            frozen_decode_n(ud_array[i], ufd_array[i], frozen_flag_n)
    return ud_array  

@nb.jit
def coding_array_all_awgn_frozen_n(u_array, frozen_flag_n, SNRdB=10):
    e_array = np.zeros_like(u_array)

    # encode를 하기 전과 decode 끝난 후에 frozen처리를 포함하면 됨.
    # u_array는 길이가 K_code인 벡터들의 모임이고, uf_array는 길이가 N_code인 벡터들의 모임이다.
    uf_array = frozen_encode_array_n(u_array, frozen_flag_n) 

    # encode_array_n()은 frozen 여부를 알 필요가 없음.
    x_array = encode_array_n(uf_array)
    y_array = channel_numpy_awgn(x_array, SNRdB)  
    ufd_array = decode_frozen_array_n(y_array, frozen_flag_n) # frozen을 고려한 함수로 변경되어야 함!
    # ufd_array = decode_array_n(y_array)
    
    ud_array = frozen_decode_array_n(ufd_array, frozen_flag_n)
    
    e_array = u_array - ud_array
    return e_array

class _PolarCodeFrozen:
    def __init__(self, N_code=2, K_code=2, frozen_flag='manual', frozen_flag_n=np.zeros(2,dtype=int)):
        """
        N_code=4: Code block size
        K_code=2: Information bit size
        frozen_flag_n=[1,1,0,0]: 코드 블럭 안의 매 비트가 frozen인지 아닌지를 표시함. Frozen이면 1, 아니면 0임.
            Frozen 아닌 비트들의 갯 수는 Code_K와 동일해야 함.
        """
        if frozen_flag == 'auto':
            frozen_flag_n = polar_design_bec(N_code=N_code, K_code=K_code)
            print('Auto: frozen_flag_n =', frozen_flag_n)
        assert N_code == len(frozen_flag_n)
        assert N_code - K_code == np.sum(frozen_flag_n)
        self.N_code = N_code
        self.K_code = K_code 
        self.frozen_flag_n = frozen_flag_n

    def plot(self, SNRdB_list, BER_list):
        plt.semilogy(SNRdB_list, BER_list)
        plt.grid()
        plt.xlabel('SNR(dB)')
        plt.ylabel('BER')
        plt.title('Performance of Polar Code')
        plt.show()    

    def run(self, 
        SNRdB_list=list(range(10)), N_iter=1, flag_fig=False):
        # 정보 비트수느느 K_code가 되어야 함. 나머지는 frozen_flag_n에 따라 0로 채워야 함.
        u_array = np.random.randint(2, size=(N_iter, self.K_code))
        
        BER_list = []
        for SNRdB in SNRdB_list:
            e_array = coding_array_all_awgn_frozen_n(u_array, frozen_flag_n=self.frozen_flag_n, SNRdB=SNRdB)
            BER = np.sum(np.abs(e_array)) / np.prod(e_array.shape)
            BER_list.append(BER)

        if flag_fig:
            self.plot(SNRdB_list, BER_list)  
        print("SNRdB_list, BER_list")
        print(SNRdB_list, BER_list)

        self.BER_list = BER_list         

def _polar_bsc(N_code=4, p=0.11, N_iter=1000):
    """
    (0,1)에 대한 원래 코드를 (1,-1)로 바꾸어서 p대신 2*p를 사용해 디자인했음.
    Input:
    p=0.11: 오류 확률을 넣는다. 그리고 p*100%의 심볼은 1로 오튜가 났다고 가정하고
        0, 1에 상관없이 오류는 p만큼 모든 심볼에 들어가는 걸로 되어 있음. 
    Comments:
    udhat는 frozen bit에 대한 실제 데이터를 결정한 값이다. 이 값은 통상 BER 계산시에는 사용되지 않는다.
    frozen bit로 설정해 오류 역전파가 없다는 가정으로 각 채널의 성능 평가를 위해 사용한다.
    """
    # 모든 비트를 frozen 시킴
    f = np.ones(N_code, dtype=int)
    biterrd = np.zeros(N_code)    
    for _ in range(N_iter):
        # 정상 입력은 모두 0으로 가정함.
        y = np.ones(N_code) - 2*p
        y[np.random.rand(N_code)<p] = -1 + 2*p
        ud_hat, _ = decode_frozen_n(y, f)
        biterrd += ud_hat        
    biterrd /= N_iter    
    return biterrd

def polar_bsc(N_code=4, p=0.11, N_iter=1000):
    """
    (0,1)에 대한 원래 코드를 (1,-1)로 바꾸어서 p대신 2*p를 사용해 디자인했음.
    Input:
    p=0.11: 오류 확률을 넣는다. 그리고 p*100%의 심볼은 1로 오튜가 났다고 가정하고
        0, 1에 상관없이 오류는 p만큼 모든 심볼에 들어가는 걸로 되어 있음. 
    Comments:
    udhat는 frozen bit에 대한 실제 데이터를 결정한 값이다. 이 값은 통상 BER 계산시에는 사용되지 않는다.
    frozen bit로 설정해 오류 역전파가 없다는 가정으로 각 채널의 성능 평가를 위해 사용한다.
    """
    # 모든 비트를 frozen 시킴
    f = np.ones(N_code, dtype=int)
    biterrd = np.zeros(N_code)    
    for _ in range(N_iter):
        # 정상 입력은 모두 0으로 가정함.
        y_bin = np.zeros(N_code) + p
        y_bin[np.random.rand(N_code)<p] = 1 - p
        ud_hat, _ = decode_frozen_n(1-2*y_bin, f)
        biterrd += ud_hat        
    biterrd /= N_iter    
    return biterrd

@nb.jit
def polar_bec(N_code=4, erase_prob=0.5):
    """
    BEC에 대해 Polar code의 예측 성능을 구한다. 단, 비트당 제거 오류율은 erase_prob로 가정한다.
    """
    n = int(np.log2(N_code))
    E = np.zeros(N_code)
    # E_out = np.zeros(N_code)    
    E[0] = erase_prob
    for i in range(n):
        LN = 2**i
        # print(i, LN)
        # print('E in:', E)
        # i stage에서 끝은 LN*2이다. 안그러면 broadcast되어 버린다.
        E[LN:LN*2] = E[:LN] * E[:LN]
        E[:LN] = 1-(1-E[:LN])*(1-E[:LN])   
        # print('E out:', E)        
    return E

def polar_design_bec(N_code=4, K_code=2, erase_prob=0.5):
    """
    BEC일 경우에 각 비트의 오류율을 계산해 frozen_flag를 만든다.
    """
    biterrd = polar_bec(N_code=N_code, erase_prob=erase_prob)
    idx = np.argsort(biterrd)
    frozen_flag_n = np.ones(N_code, dtype=int)
    frozen_flag_n[idx[:K_code]] = 0
    print('BER for each bit', biterrd)
    return frozen_flag_n

class _PolarCodeFrozen:
    def __init__(self, N_code=2, K_code=2, frozen_flag='manual', frozen_flag_n=np.zeros(2,dtype=int)):
        """
        N_code=4: Code block size
        K_code=2: Information bit size
        frozen_flag_n=[1,1,0,0]: 코드 블럭 안의 매 비트가 frozen인지 아닌지를 표시함. Frozen이면 1, 아니면 0임.
            Frozen 아닌 비트들의 갯 수는 Code_K와 동일해야 함.
        """
        if frozen_flag == 'auto':
            frozen_flag_n = polar_design_bec(N_code=N_code, K_code=K_code)
            print('Auto: frozen_flag_n =', frozen_flag_n)
        assert N_code == len(frozen_flag_n)
        assert N_code - K_code == np.sum(frozen_flag_n)
        self.N_code = N_code
        self.K_code = K_code 
        self.frozen_flag_n = frozen_flag_n

    def plot(self, SNRdB_list, BER_list):
        plt.semilogy(SNRdB_list, BER_list)
        plt.grid()
        plt.xlabel('SNR(dB)')
        plt.ylabel('BER')
        plt.title('Performance of Polar Code')
        plt.show()    

    def run(self, 
        SNRdB_list=list(range(10)), N_iter=1, flag_fig=False):
        # 정보 비트수느느 K_code가 되어야 함. 나머지는 frozen_flag_n에 따라 0로 채워야 함.
        u_array = np.random.randint(2, size=(N_iter, self.K_code))
        
        BER_list = []
        for SNRdB in SNRdB_list:
            e_array = coding_array_all_awgn_frozen_n(u_array, frozen_flag_n=self.frozen_flag_n, SNRdB=SNRdB)
            BER = np.sum(np.abs(e_array)) / np.prod(e_array.shape)
            BER_list.append(BER)

        if flag_fig:
            self.plot(SNRdB_list, BER_list)  
        print("SNRdB_list, BER_list")
        print(SNRdB_list, BER_list)

        self.BER_list = BER_list  


class PolarCodeFrozen:
    def __init__(self, N_code=2, K_code=2, frozen_flag='manual', frozen_flag_n=np.zeros(2,dtype=int)):
        """
        N_code=4: Code block size
        K_code=2: Information bit size
        frozen_flag_n=[1,1,0,0]: 코드 블럭 안의 매 비트가 frozen인지 아닌지를 표시함. Frozen이면 1, 아니면 0임.
            Frozen 아닌 비트들의 갯 수는 Code_K와 동일해야 함.
        """
        if frozen_flag == 'auto':
            frozen_flag_n = polar_design_bec(N_code=N_code, K_code=K_code)
            print('Auto: frozen_flag_n =', frozen_flag_n)
        assert N_code == len(frozen_flag_n)
        assert N_code - K_code == np.sum(frozen_flag_n)
        self.N_code = N_code
        self.K_code = K_code 
        self.frozen_flag_n = frozen_flag_n

    def plot(self, SNRdB_list, BER_list):
        plt.semilogy(SNRdB_list, BER_list)
        plt.grid()
        plt.xlabel('SNR(dB)')
        plt.ylabel('BER')
        plt.title('Performance of Polar Code')
        plt.show()    

    def display_flag(self, SNRdB_list, BER_list, flag_fig):
        if flag_fig:
            self.plot(SNRdB_list, BER_list) 
        print("SNRdB_list, BER_list")
        print(SNRdB_list, BER_list)
        self.BER_list = BER_list  

    def run(self, SNRdB_list=list(range(10)), N_iter=1, flag_fig=False):
        # 정보 비트수는 K_code가 되어야 함. 나머지는 frozen_flag_n에 따라 0로 채워야 함.
        u_array = np.random.randint(2, size=(N_iter, self.K_code))
        BER_list = []
        for SNRdB in SNRdB_list:
            e_array = coding_array_all_awgn_frozen_n(u_array, frozen_flag_n=self.frozen_flag_n, SNRdB=SNRdB)
            BER = np.sum(np.abs(e_array)) / np.prod(e_array.shape)
            BER_list.append(BER)
        self.display_flag(SNRdB_list, BER_list, flag_fig)  

#========================================================================================
# NPolar
#========================================================================================
@nb.jit
def np_encode_n_NP(u, N_P):
    """
    x = uBF(xn) where n = log(N), N=len(u), B is bit-reverse
    """
    x = np.copy(u)
    L = len(u)
    if L != 1:
        if L > N_P:
            u1 = u[0::2]
            u2 = u[1::2]
            x[:L/2] = np_encode_n_NP(u1, N_P)
            x[L/2:] = np_encode_n_NP(u2, N_P)           
        else:
            u1 = u[0::2]
            u2 = u[1::2]
            u1u2 = np.mod(u1 + u2, 2) 
            x[:L/2] = np_encode_n_NP(u1u2, N_P)
            x[L/2:] = np_encode_n_NP(u2, N_P)
    return x

@nb.jit
def _np_encode_n(u, P):
    """
    Inputs:
    P=2: AE 입력의 길이, P=1이면 AE 미사용, P=N이면 Polar 미사용
    """
    return np_encode_n_NP(u, len(u)//P)

@nb.jit
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
        y = np.zeros_like(x)
        y[0::2] = bit_forward(x[:LN/2])
        y[1::2] = bit_forward(x[LN/2:])
    return y

@nb.jit
def ae_emul(u):
    """
    AE가 해야할 일을 Polar로 emulation시킴
    Polar로 emulation시키기 위해서는 bit reverse 되어 있는걸 복원해야 함.
    """
    u = bit_forward(u)
    x = encode_n(u)
    return x

@nb.jit
def np_encode_n(u, P):
    """
    Inputs:
    P=2: AE 입력의 길이, P=1이면 AE 미사용, P=N이면 Polar 미사용
    """
    N_P = len(u)//P
    y = np_encode_n_NP(u, N_P)    
    x = np.zeros_like(y) # ae_polar
    ix = np.arange(len(u))
    for i in range(N_P):
        ae_in = y[i::N_P]
        ae_ix = ix[i::N_P]
        ae_out = ae_emul(ae_in)
        x[ae_ix] = ae_out
    return x

@nb.jit
def np_encode_array_n(u_array, P_code):
    x_array = np.zeros_like(u_array)
    for i in range(len(u_array)):
        x_array[i] = np_encode_n(u_array[i], P_code)        
    return x_array

@nb.jit
def np_coding_array_all_awgn_frozen_n(u_array, P_code, frozen_flag_n, SNRdB=10):
    e_array = np.zeros_like(u_array)

    # encode를 하기 전과 decode 끝난 후에 frozen처리를 포함하면 됨.
    # u_array는 길이가 K_code인 벡터들의 모임이고, uf_array는 길이가 N_code인 벡터들의 모임이다.
    uf_array = frozen_encode_array_n(u_array, frozen_flag_n) 

    # encode_array_n()은 frozen 여부를 알 필요가 없음.
    # print('P_code:', P_code)
    x_array = np_encode_array_n(uf_array, P_code)
    y_array = channel_numpy_awgn(x_array, SNRdB)  
    ufd_array = decode_frozen_array_n(y_array, frozen_flag_n) # frozen을 고려한 함수로 변경되어야 함!
    # ufd_array = decode_array_n(y_array)
    
    ud_array = frozen_decode_array_n(ufd_array, frozen_flag_n)
    
    e_array = u_array - ud_array
    return e_array

class NPolarCodeFrozen(PolarCodeFrozen):
    """
    - Emulation version: AE is emulated by Polar encoding
    - AE 해야할 기능을 Polar 파트가 대신하게 함. 
    - P_code 부분은 AE가 돌리는 걸 가정했고 그 가능성은 Polar가 대신하고 있음.    
    """
    def __init__(self, N_code=4, K_code=4, P_code=1, frozen_flag='manual', frozen_flag_n=np.zeros(2,dtype=int)):    
        """
        Inputs:
        P_code: the number of AE input bits
        """
        super().__init__(N_code=N_code, K_code=K_code, 
            frozen_flag=frozen_flag, frozen_flag_n=frozen_flag_n)
        self.P_code = P_code

    def run(self, SNRdB_list=list(range(10)), N_iter=1, flag_fig=False):
        # 정보 비트수는 K_code가 되어야 함. 나머지는 frozen_flag_n에 따라 0로 채워야 함.
        u_array = np.random.randint(2, size=(N_iter, self.K_code))
        BER_list = []
        for SNRdB in SNRdB_list:
            e_array = np_coding_array_all_awgn_frozen_n(u_array, self.P_code,
                        frozen_flag_n=self.frozen_flag_n, SNRdB=SNRdB)
            BER = np.sum(np.abs(e_array)) / np.prod(e_array.shape)
            BER_list.append(BER)
        self.display_flag(SNRdB_list, BER_list, flag_fig)  


if __name__ == '__main__':
    # main_run_coding_awgn()
    # main_run_coding_array_all_awgn_tile(Ntile=100000, flag_fig=True)
    #f = polar_design_bec(2,1)
    #print(f)
    #polar = NPolarCodeFrozen(2, 2, 'auto')
    #polar.run([5], 10)
    polar = NPolarCodeFrozen(N_code=4, K_code=4, P_code=4, frozen_flag='auto')
    polar.run(SNRdB_list=list(range(10)), N_iter=10000, flag_fig=True)    