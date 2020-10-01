import numpy as np
import numba as nb
import matplotlib.pyplot as plt

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

def calc_ber(e_array):
    return np.mean(np.abs(e_array))

if __name__ == '__main__':
    # main_run_coding_awgn()
    main_run_coding_array_all_awgn_tile(Ntile=100000, flag_fig=True)