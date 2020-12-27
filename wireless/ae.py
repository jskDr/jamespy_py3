# Autoencoder transceiver
"""
[변조 방식]
- 변조 방식은 BPSK를 가정함
- QAM 계열이 될려면 I,Q 채널이 따로 존재해야 함. 그리고 채널을 고려해야 함.
- 채널을 고려하지 않으면 어짜피 BPSK와 동일하게 동작하여 아래 코드가 정당함. 

[페이딩 채널 고려]
- 채널이 있게 되면, 수신단에서는 채널을 알아야 성능이 제대로 나올 수 있음. 수신 채널을 어떻게 사용할지 고민해야 함. 
  수신 채널의 역이 입력 신호에 곱해지는 형태가 되어야 함. y * (1/h) 또는 MMSE 방식을 사용할 수 있음. 
- 수신 채널이 고정되면 AI가 학습을 통해 스스로 찾을 수 있음. 그러나 실제 수신시는 다른 채널이 적용되기 때문에 
  고정된 수신 채널로 학습한 경우는 사용할 수가 없음. 파일럿을 이용해 수신 채널을 알아내고 이 정보를 사용하는 형태로 구성되어야 함. 
- 채널 코딩 복호시도 classification에 사용하듯 ML이 아니라 이를 단순화하는 방식이 필요함. 그러나 인공지능은 아직도 ML로 사용하고 있음.
  CNN, RNN 계열을 사용하게 되면 ML 방식의 사용에 따른 부담을 줄 수 있는건지 확인이 필요함. 

[One-hot 입력/출력]
- 입력 뿐 아니라 출력도 one-hot으로 처리해야 함.
  - 특히 성능을 좌우하는 출력이 one-hot으로 만들어져야 함.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Layer, BatchNormalization
from tensorflow.keras import Sequential

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from wireless import qam_lib

class InputBlocks: # Pulse amplitude modulation (PAM) such as Pulse shift keying (PSK)
    def __init__(self, Code_K=2, Total_blocks=1000, buffer_size=1024, batch_size=64):        
        """
        - 여기에 정리된 심볼은 I,Q를 구분하지 않기 때문에 PAM이라고 보는게 맞다.
        - Amplitude-shift keying (ASK), v[i] = 2A*i/(L-1) - A for [-A, A],와 유사하지만 진폭 차이는 AI 결정함
        - 입력의 길이는 가능한 메시지 수이다. 이에 대해 onehot 변환이 되어 있어야 함.
        - 여기에 비해 Encoder로 출력되는 심볼의 수는 Code_N/N_mod*2가 되어 매우 줄어들게 된다.
        
        - 입력과 출력 모두 onehot encoding할 결과가 되어야 한다. 
        - X_train, X_test, Y_train, Y_test = train_test_split(S_onehot, S_onehot, test_size=0.2)
        - y는 onehot이 다시 복원된 index 값이 들어가야 한다. 반면, X는 onehot을 처리한 것이 들어가야 함.
        """
        Total_messages = 2 ** Code_K
        S = np.random.randint(Total_messages, size=Total_blocks) 
        S_onehot = tf.one_hot(S, Total_messages, dtype='float32').numpy()
        print(S.shape, S_onehot.shape)

        X_train, X_test, Y_train, Y_test = train_test_split(S_onehot, S, test_size=0.2)
        dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size) # 여기서 1024, 64는 임의의 value이다.
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        test_dataset = test_dataset.shuffle(buffer_size=buffer_size).batch(batch_size)   
        
        #self.N_mod = N_mod
        self.Total_messages = Total_messages
        self.dataset = dataset
        self.test_dataset = test_dataset


class AE(Layer):
    def __init__(self, Total_mod_bits=1, Code_N=2, Code_K=2, Total_blocks=1000, EbNo_dB=10, Model_type='linear', Modulation_type='QAM'):
        """
        Inputs:
        Modulation_Type = QAM or PAM 

        설명:
        - AE의 입력은 변조 차원에 상관없이 Code_K가 되어야 하고
          출력은 Code_N이 되어야 한다. 
        - 변조는 Code_N 이후에 다루어져야 하기에 
          BPSK는 Code_N 심볼이 생기고, QPSK는 Code_N/2 심볼이 생긴다.
        - Code_N/변조차수가 정수가 아니라면 현재 개념으로는 AE가 처리하기 힘들기 때문에, 
          AE에 맞게 Code_N/변주차수가 정수가 되도록 Code_N을 설정해야 한다. 
        - (7,4) 해밍코드의 경우 BPSK로 변조는 가능하지만 QPSK 변조와 그 이상은 안되고, 
          (8,4) 코드를 가정한다면 BPSK(1bit), QPSK(2bits), 16-QAM(4bits), 256-QAM(8bits)이 가능하다.
          사실 현재는 I, Q를 구분하지 않고 있어서 BPSK -> PAM, QPSK -> 2PAM
        - 1차 관문은 AWGN, BPSK의 경우 Polar와 성능 비교이다. (2020-10-9)

        - 변조는 채널과 곱해지면서 그 특성이 결정된다. 채널을 고려하지 않는 단계에서는 크게 중요하지 않다. 
        - 그리고 I, Q 축의 bit수를 동일하게 하라는 현재 방법보다 I+jQ의 갯수를 동시에 지정하는 것이 더 바람직하다.
        - 이를 위해서는 복소수 신호처리 과정이 들어가야 한다. Tensorflow가 복소수 전반에 대해 지원하는지 확인이 필요하다.

        - EbNo_dB는 추론시는 학습시와 다른 값을 넣어도 되지만 여기서는 같은 값을 사용하는 걸 가정해 구했다. 
          최고의 성능을 구한 뒤, 일반화하는 것은 다음에 하기로 했기 때문이다. 
        """
        super(AE, self).__init__()
        self.Total_mod_bits = Total_mod_bits
        self.Code_N = Code_N
        self.Code_K = Code_K
        # 출력은 one-hot encoding을 가정하고 있는 만큼 Total_messages 만큼의 이진수 배열이 나와야 한다.
        self.Total_messages = 2 ** Code_K
        self.Modulation_type = Modulation_type 
        self.set_data(Total_blocks, EbNo_dB)
        self.init_model(Model_type)

    def init_model(self, Model_type):
        """
        Model_type이 linear이면 linear 방식의 뉴럴넷을 사용하고 입력과 출력은 자동으로 결정된다.
        nolinear인 경우는 송신과 수신 뉴럴넷 모두 한 개의 추가 히든 계층을 사용하게 된다.
        히든 계층의 노드 수는 출력과 동일하게 일단은 지정하고 추후에 BO등을 활용해 최적화할 예정이다.
        """
        if self.Modulation_type == 'QAM':
            if Model_type == 'nonlinear': 
                print('Set nonlinear model')
                # the length of input is Code_K
                if self.Total_mod_bits == 1: # BPSK, 즉 IQ를 이용하지 않는 경우임.
                    # length of a code block / Modulation order
                    self.Encoder = Sequential([Dense(self.Code_N, activation='relu'), Dense(self.Code_N)])     
                else: 
                    """
                    - 이 부분이 수정되어야 함. PAM이므로 2개 쌍이 되어서 나올 필요가 없음. 
                    - 입력 파라메터에서 PAM, QAM인지를 선택하게 해야 함. 현재는 Total_mod_bits가 2 이상인 경우, QAM이라고 가정하고 있음.
                    그러나 이 경우도 ASK와 유사한 PAM으로 보는게 더 맞아 보임. QAM의 경우는 추후 I, Q로 나눠보내고 채널도 복소수가 되어야 의미가 있음.
                    - Total_mod_bits가 2 이상의 경우, QAM이라고 가정한 이유는 이 경우가 노이즈는 동일하지만 
                    두 축으로 보낼 수 있어 PAM보다 수신 성능에 있어 이득이 있어서임. 
                    2-QAM은 2-PAM보다 성능이 우수하고 대부분의 논문들이 2-QAM을 가정하고 있음.
                    - 시뮬레이션 상에서는 2-QAM은 2 time을 사용하는 것으로 묘사하고 있으나 실제는 I, Q 채널로 날아가는 것임. 
                    10개의 심볼이 날아가면 짝수(0, 2, 4, ...)는 I축이고 홀수(1, 3, 5, ...)는 Q축이라고 볼 수 있음. 
                    - QAM: AE Encoder의 출력을 송신기에서 I와 Q로 분리하고 수신기에 다시 I와 Q로 결합하는 블럭을 시뮬레이션 상에서 고려할 수 있겠음. 
                    - IQ를 이용하는 경우임. 2개가 쌍이 되어서 나와야 함. 
                    - (I,Q) * length of a code block / Modulation order
                    """
                    assert 2*self.Code_N//self.Total_mod_bits == 2*self.Code_N/self.Total_mod_bits
                    ln_out = 2*self.Code_N/self.Total_mod_bits
                    self.Encoder = Sequential([Dense(ln_out,activation='relu'), Dense(ln_out)]) 
                self.Decoder = Sequential([Dense(self.Total_messages, activation='relu'), Dense(self.Total_messages)]) 
            elif Model_type == 'nonlinear_bn': # BatchNormalization()을 사용하는 경우임
                print('Set nonlinear & batchnormalization model')
                # the length of input is Code_K
                if self.Total_mod_bits == 1: # BPSK, 즉 IQ를 이용하지 않는 경우임.
                    # length of a code block / Modulation order
                    self.Encoder = Sequential([Dense(self.Code_N, activation='relu'), 
                                    BatchNormalization(), Dense(self.Code_N)])     
                else: # IQ를 이용하는 경우임. 2개가 쌍이 되어서 나와야 함. 
                    # (I,Q) * length of a code block / Modulation order
                    assert 2*self.Code_N//self.Total_mod_bits == 2*self.Code_N/self.Total_mod_bits
                    ln_out = 2*self.Code_N/self.Total_mod_bits
                    self.Encoder = Sequential([Dense(ln_out,activation='relu'), 
                                    BatchNormalization(), Dense(ln_out)]) 
                self.Decoder = Sequential([Dense(self.Total_messages, activation='relu'), 
                                BatchNormalization(), Dense(self.Total_messages)]) 
            else: # Model_type == 'linear'
                print('Set linear model')
                # the length of input is Code_K
                if self.Total_mod_bits == 1: # BPSK, 즉 IQ를 이용하지 않는 경우임.
                    # length of a code block / Modulation order
                    self.Encoder = Sequential([Dense(self.Code_N)])     
                else: # IQ를 이용하는 경우임. 2개가 쌍이 되어서 나와야 함. 
                    # (I,Q) * length of a code block / Modulation order
                    assert 2*self.Code_N//self.Total_mod_bits == 2*self.Code_N/self.Total_mod_bits
                    self.Encoder = Sequential([Dense(2*self.Code_N/self.Total_mod_bits)]) 
                self.Decoder = Sequential([Dense(self.Total_messages)]) 
        else: # self.Modulatin_type == 'PAM'            
            print('Set linear model')
            # the length of input is Code_K
            assert self.Code_N//self.Total_mod_bits == self.Code_N/self.Total_mod_bits
            self.Encoder = Sequential([Dense(self.Code_N/self.Total_mod_bits)]) # No of input features = Total_messages
            self.Decoder = Sequential([Dense(self.Total_messages)])             

    def set_data(self, Total_blocks, EbNo_dB):
        Code_K = self.Code_K
        Total_mod_bits = self.Total_mod_bits

        inblock = InputBlocks(Code_K, Total_blocks)
        self.Total_messages = inblock.Total_messages
        self.dataset = inblock.dataset
        self.test_dataset = inblock.test_dataset

        self.EbNo_dB = EbNo_dB
        SNRdB = EbNo_dB + 10*np.log10(Total_mod_bits) 
        SNR = np.power(10, SNRdB / 10)
        self.noise_sig = 1 / np.sqrt(SNR)

    def norm_awgn(self, x, noise_sig):
        # transmit power normalization
        #x = x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=0, keepdims=True))
        # if more than a symbol are used in a code block, additional power is needed.
        # norm_pwr = tf.reduce_mean(tf.reduce_sum(tf.square(x),axis=-1)) / self.Code_N
        norm_pwr = tf.reduce_mean(tf.reduce_sum(tf.square(x),axis=-1)) * (self.Code_K / self.Code_N)
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
    Total_mod_bits=2,
    Code_K=2,
    Code_N=4,
    Total_blocks=1000,
    Total_episodes=20,
    EbNo_dB=10,
    Model_type='linear',
    Modulation_type='PAM'):

    # This is code rate
    model = AE(Total_mod_bits, Code_N, Code_K, Total_blocks, EbNo_dB, Model_type, Modulation_type)
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

    for e in range(Total_episodes):    
        for x, y in dataset:
            with tf.GradientTape() as tape:
                code_logits, pwr = model(x, noise_sig)
                print(code_logits.shape, y.shape)
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

def large_test(model, Total_blocks=10000):
    """
    - 통신 시스템의 특성상 많은 비트들에 대해서 추가 테스트를 실시해 BER을 구한다.
    - 여기에 필요한 대부분의 파라메터들은 model을 만들때 정해졌으므로, 
      모델에 그 파라메터를 포함하도록 코드를 재구성했다.
    """
    noise_sig = model.noise_sig
    Total_messages = model.Total_messages
    S = np.random.randint(Total_messages, size=Total_blocks) 
    S_onehot = tf.one_hot(S, Total_messages, dtype='float32').numpy()
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
        Total_mod_bits=1,
        Code_K=2,Code_N=2,
        Total_episodes=500,
        Total_blocks_train=2000,
        Total_blocks_test=100000,
        Model_type='linear'):
    """
    Inputs:
    EbNo_dB_l = range(10): 0 ~ 9까지 EbNo(dB)에 대해 테스트를 수행함.
    """
    BER_l = []
    for EbNo_dB in EbNo_dB_l:
        trained_model = ae_train_test(Total_mod_bits=Total_mod_bits,Code_K=Code_K,Code_N=Code_N,
            Total_blocks=Total_blocks_train,Total_episodes=Total_episodes,EbNo_dB=EbNo_dB, Model_type=Model_type)   
        BER = large_test(trained_model, Total_blocks=Total_blocks_test)
        BER_l.append(BER)
        print(EbNo_dB, BER)    

    plt.semilogy(EbNo_dB_l, BER_l)
    plt.xlabel('SNR(dB)')
    plt.ylabel('BER')
    plt.title(f'AE transceiver: $(N,K)=({Code_N},{Code_K})$, Modulation(bits):{Total_mod_bits}')
    plt.grid()
    plt.show()    

    print('EbNo_dB_l, BER_l:')
    print(EbNo_dB_l, BER_l)


if __name__ == '__main__':
    trained_model = ae_train_test(
        Total_mod_bits=3,
        Code_K=2,
        Code_N=6,
        Total_blocks=1000,
        Total_episodes=200,
        EbNo_dB=5,
        Model_type='linear',
        Modulation_type='PAM')  
    BER = large_test(trained_model, Total_blocks=10000)
    print(BER)