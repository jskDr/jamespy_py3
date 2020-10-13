# Autoencoder transceiver
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Layer, BatchNormalization
from tensorflow.keras import Sequential

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from wireless import qam_lib

class QAM_Code:
    def __init__(self, Code_K=2, N_sample=1000):        
        """
        - 입력의 길이는 가능한 메시지 수이다. 이에 대해 onehot 변환이 되어 있어야 함.
        - 여기에 비해 Encoder로 출력되는 심볼의 수는 Code_N/N_mod*2가 되어 매우 줄어들게 된다.
        
        - 입력과 출력 모두 onehot encoding할 결과가 되어야 한다. 
        - X_train, X_test, Y_train, Y_test = train_test_split(S_onehot, S_onehot, test_size=0.2)
        - y는 onehot이 다시 복원된 index 값이 들어가야 한다. 반면, X는 onehot을 처리한 것이 들어가야 함.
        """
        N_messages = 2 ** Code_K
        S = np.random.randint(N_messages, size=N_sample) 
        S_onehot = tf.one_hot(S, N_messages, dtype='float32').numpy()
        print(S.shape, S_onehot.shape)

        X_train, X_test, Y_train, Y_test = train_test_split(S_onehot, S, test_size=0.2)
        dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        dataset = dataset.shuffle(buffer_size=1024).batch(64) # 여기서 1024, 64는 임의의 값이다.
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        test_dataset = test_dataset.shuffle(buffer_size=1024).batch(64)   
        
        #self.N_mod = N_mod
        self.N_messages = N_messages
        self.dataset = dataset
        self.test_dataset = test_dataset


class AE(Layer):
    def __init__(self, N_mod_bits=1, Code_N=2, Code_K=2, N_sample=1000, EbNo_dB=10):
        """
        - AE의 입력은 변조 차원에 상관없이 Code_K가 되어야 하고
          출력은 Code_N이 되어야 한다. 
        - 변조는 Code_N 이후에 다루어져야 하기에 
          BPSK는 Code_N 심볼이 생기고, QPSK는 Code_N/2 심볼이 생긴다.
        - Code_N/변조차수가 정수가 아니라면 현재 개념으로는 AE가 처리하기 힘들기 때문에, 
          AE에 맞게 Code_N/변주차수가 정수가 되도록 Code_N을 설정해야 한다. 
        - (7,4) 해밍코드의 경우 BPSK로 변조는 가능하지만 QPSK 변조와 그 이상은 안되고, 
          (8,4) 코드를 가정한다면 BPSK(1bit), QPSK(2bits), 16-QAM(4bits), 256-QAM(8bits)이 가능하다.
        - 1차 관문은 AWGN, BPSK의 경우 Polar와 성능 비교이다. (2020-10-9)

        - 변조는 채널과 곱해지면서 그 특성이 결정된다. 채널을 고려하지 않는 단계에서는 크게 중요하지 않다. 
        - 그리고 I, Q 축의 bit수를 동일하게 하라는 현재 방법보다 I+jQ의 갯수를 동시에 지정하는 것이 더 바람직하다.
        - 이를 위해서는 복소수 신호처리 과정이 들어가야 한다. Tensorflow가 복소수 전반에 대해 지원하는지 확인이 필요하다.

        - EbNo_dB는 추론시는 학습시와 다른 값을 넣어도 되지만 여기서는 같은 값을 사용하는 걸 가정해 구했다. 
          최고의 성능을 구한 뒤, 일반화하는 것은 다음에 하기로 했기 때문이다. 
        """
        super(AE, self).__init__()
        self.N_mod_bits = N_mod_bits
        self.Code_N = Code_N
        self.Code_K = Code_K
        # 출력은 one-hot encoding을 가정하고 있는 만큼 N_messages 만큼의 이진수 배열이 나와야 한다.
        self.N_messages = 2 ** Code_K

        self.set_data(N_sample, EbNo_dB)

        # the length of input is Code_K
        if N_mod_bits == 1: # BPSK, 즉 IQ를 이용하지 않는 경우임.
            # length of a code block / Modulation order
            self.Encoder = Sequential([Dense(Code_N)])     
        else: # IQ를 이용하는 경우임. 2개가 쌍이 되어서 나와야 함. 
            # (I,Q) * length of a code block / Modulation order
            assert 2*Code_N//N_mod_bits == 2*Code_N/N_mod_bits
            self.Encoder = Sequential([Dense(2*Code_N/N_mod_bits)]) 
        self.Decoder = Sequential([Dense(self.N_messages)]) 

    def set_data(self, N_sample, EbNo_dB):
        Code_K = self.Code_K
        N_mod_bits = self.N_mod_bits

        qam = QAM_Code(Code_K, N_sample)
        self.N_messages = qam.N_messages
        self.dataset = qam.dataset
        self.test_dataset = qam.test_dataset

        self.EbNo_dB = EbNo_dB
        SNRdB = EbNo_dB + 10*np.log10(N_mod_bits) 
        SNR = np.power(10, SNRdB / 10)
        self.noise_sig = 1 / np.sqrt(SNR)

    def norm_awgn(self, x, noise_sig):
        # transmit power normalization
        #x = x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=0, keepdims=True))
        # if more than a symbol are used in a code block, additional power is needed.
        norm_pwr = tf.reduce_mean(tf.reduce_sum(tf.square(x),axis=-1)) / self.Code_N
        noise = tf.random.normal(x.shape) * noise_sig
        x = x + noise
        return x, norm_pwr
        
    def call(self, x, noise_sig):
        # Merge messages to a code block
        # x = self.merge(x)        
        x = self.Encoder(x) # in: 2**Code_K, out: Code_N/N_mod*2
        x, norm_pwr = self.norm_awgn(x, noise_sig)        
        x = self.Decoder(x) # in: Code_N/N_mod*2, out: 2**Code_K
        # x = self.split(x)
        return x, norm_pwr

def ae_train_test(
    N_mod_bits=2,
    Code_K=2,
    Code_N=4,
    N_sample=1000,
    N_episodes=20,
    EbNo_dB=10):

    # This is code rate
    model = AE(N_mod_bits, Code_N, Code_K, N_sample, EbNo_dB)
    # not softmax is used at the end of the network
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    optimizer = tf.keras.optimizers.Adam()

    w_list = []

    loss_train_l = []
    acc_train_l = []
    loss_test_l = []
    acc_test_l = []

    noise_sig = model.noise_sig
    dataset = model.dataset
    test_dataset = model.test_dataset

    for e in range(N_episodes):    
        for x, y in dataset:
            with tf.GradientTape() as tape:
                code_logits, pwr = model(x, noise_sig)
                # print(code_logits.shape, y.shape)
                loss_value = loss(y, code_logits) + tf.square(pwr - 1)
            gradients = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(gradients, model.trainable_weights))
            accuracy.update_state(y, code_logits)
        loss_train_l.append(float(loss_value))
        acc_train_l.append(float(accuracy.result()))

        w_list.append([model.Encoder.weights[0].numpy(), model.Encoder.weights[1].numpy(), 
            model.Decoder.weights[0].numpy(), model.Decoder.weights[1].numpy()])
        
        # Testing is stated
        accuracy.reset_states()
        for x, y in test_dataset:
            test_code_logits, pwr = model(x, noise_sig)
            test_loss_value = loss(y, test_code_logits) + tf.square(pwr - 1)
            accuracy.update_state(y, test_code_logits)
        loss_test_l.append(float(test_loss_value)) 
        acc_test_l.append(float(accuracy.result())) 

        if e % 10 == 0:
            #print(model.Encoder.weights[0].numpy(), model.Encoder.weights[1].numpy(), 
            #      model.Decoder.weights[0].numpy(), model.Decoder.weights[1].numpy())
            print('Epoch:', e, end=', ')
            # train_loss_value = loss_train_l[-1]
            print('트레이닝의 손실과 마지막 스텝 정확도, 오류:', float(loss_value), acc_train_l[-1], 1 - acc_train_l[-1])
            print('테스트의 손실과 마지막 스텝 정확도, 오류:', float(test_loss_value), acc_test_l[-1], 1 - acc_test_l[-1])
            # print(model.trainable_weights)    

    qam_lib.plot_loss_acc_l(loss_train_l, loss_test_l, acc_train_l, acc_test_l)    

    return model

def large_test(model, N_sample=10000):
    """
    - 통신 시스템의 특성상 많은 비트들에 대해서 추가 테스트를 실시해 BER을 구한다.
    - 여기에 필요한 대부분의 파라메터들은 model을 만들때 정해졌으므로, 
      모델에 그 파라메터를 포함하도록 코드를 재구성했다.
    """
    noise_sig = model.noise_sig
    N_messages = model.N_messages
    S = np.random.randint(N_messages, size=N_sample) 
    S_onehot = tf.one_hot(S, N_messages, dtype='float32').numpy()
    test_dataset = tf.data.Dataset.from_tensor_slices((S_onehot, S))
    test_dataset = test_dataset.shuffle(buffer_size=5000).batch(1000)   

    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    for x, y in test_dataset:
        test_code_logits, _ = model(x, noise_sig)
        # test_loss_value = loss(y, test_code_logits) + tf.square(pwr - 1)
        accuracy.update_state(y, test_code_logits)

    return 1 - float(accuracy.result())

def test_snr_range(
        EbNo_dB_l = range(10),
        N_mod_bits=1,
        Code_K=2,Code_N=2,
        N_episodes=500,
        N_sample_train=2000,
        N_sample_test=100000):
    """
    Inputs:
    EbNo_dB_l = range(10): 0 ~ 9까지 EbNo(dB)에 대해 테스트를 수행함.
    """
    BER_l = []
    for EbNo_dB in EbNo_dB_l:
        trained_model = ae_train_test(N_mod_bits=N_mod_bits,Code_K=Code_K,Code_N=Code_N,
            N_sample=N_sample_train,N_episodes=N_episodes,EbNo_dB=EbNo_dB)   
        BER = large_test(trained_model, N_sample=N_sample_test)
        BER_l.append(BER)
        print(EbNo_dB, BER)    

    plt.semilogy(EbNo_dB_l, BER_l)
    plt.xlabel('SNR(dB)')
    plt.ylabel('BER')
    plt.title(f'AE transceiver: $(N,K)=({Code_N},{Code_K})$, Modulation(bits):{N_mod_bits}')
    plt.grid()
    plt.show()    


if __name__ == '__main__':
    trained_model = ae_train_test(
    N_mod_bits=2,
    Code_K=2,
    Code_N=4,
    N_sample=1000,
    N_episodes=100,
    EbNo_dB=10)  

    BER = large_test(trained_model, N_sample=100000)
    print(BER)