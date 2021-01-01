# This code is developed based on RL code in the following repository in github
# https://github.com/rlcode/reinforcement-learning-kr-v2
import numpy as np

np.random.seed(1)

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

def int_1d_array_to_binary_2d_array(messages, K_code): 
    messages = np.vectorize(np.binary_repr)(messages, width=K_code)
    messages = np.array([list(m) for m in messages])
    return messages.astype(int)

def encode_array_n_hamming(G, u_array):
    return np.mod(np.dot(u_array, G),2)

def decode_array_n_hamming(G, y_array, K_code:int=4):    
    m = np.arange(2**K_code, dtype=int)
    m = int_1d_array_to_binary_2d_array(m, K_code)
    x = 1.0 - 2.0*np.mod(np.dot(m, G),2) # 16 x 7
    r_array = np.dot(y_array, x.T)
    argmax_r_arry = np.argmax(r_array,axis=1)
    ud_array = m[argmax_r_arry,:]
    return ud_array

def coding_array_all_awgn_hamming(G, u_array, SNRdB=10, K_code=4):
    # u_array = np.random.randint(2, size=(5, 4))
    x_array = encode_array_n_hamming(G, u_array)
    y_array = channel_numpy_awgn(x_array, SNRdB)
    ud_array = decode_array_n_hamming(G, y_array, K_code)
    e_array = u_array - ud_array
    return e_array  

class Env():
    def __init__(self, N_code:int=7, K_code:int=4, Target_BLER:float=0.01):
        self.N_code = N_code
        self.K_code = K_code
        self.target_BLER = Target_BLER

        self.rewards = []
        self.state = np.concatenate([np.eye(K_code,dtype=int), np.zeros((K_code, N_code-K_code),dtype=int)],axis=1).reshape(-1)
        self.state_size = len(self.state)
 
        # Gaussian policy, 최고치 한개만 선택되는 softmax와는 다르게 모든 element가 영향을 준다.
        # action_space: Gaussian policy --> (a - mu(s)) * G
        self.action_size = K_code * (N_code-K_code) 
        self.title = 'REINFORCE-GaussianPolicy'

    def reset(self):
        self.reset_reward()
        return self.get_state()

    def reset_reward(self):
        self.rewards.clear()
        # self.goal.clear()

    def get_state(self):    
        return self.state

    def step(self, action, target_BLER=0.01):
        # self.y = self.messages * G
        self.state, reward, done = self.coding_run(action)
        return self.state, reward, done

    def coding_run(self, action, MAX_SNRdB=10, N_iter=10000):
        G = self.state.reshape(self.K_code, self.N_code)
        G[:,self.K_code:] = action.reshape(self.K_code, self.N_code - self.K_code)
        u_array = np.random.randint(2, size=(N_iter, self.K_code))
        # done = False
        for SNRdB in range(MAX_SNRdB, -1, -1):
            e_array = coding_array_all_awgn_hamming(G, u_array, SNRdB=SNRdB, K_code=self.K_code)
            BER = np.sum(np.abs(e_array)) / np.prod(e_array.shape) #BLER로 변경 필요
            if BER > self.target_BLER:
                assert SNRdB != MAX_SNRdB, "MAX_SNRdB should be increased"
                SNRdB = SNRdB + 1
                # done = True
                break
        done = True
        return G.reshape(-1), -SNRdB, done
