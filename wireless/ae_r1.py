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
    def __init__(self, N_bits=2, Code_K=2, N_sample=1000):
        N_mod = 2 ** N_bits
        S = np.random.randint(0, N_mod, (N_sample, Code_K)) 
        S_onehot = tf.one_hot(S, N_mod, dtype='float32').numpy()
        print(S.shape, S_onehot.shape)

        X_train, X_test, y_train, y_test = train_test_split(S_onehot, S, test_size=0.2)
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.shuffle(buffer_size=1024).batch(64) # 64 --> 4 * N_mod
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.shuffle(buffer_size=1024).batch(64)  # 64 --> 4 * N_mod 
        
        self.N_mod = N_mod
        self.dataset = dataset
        self.test_dataset = test_dataset


class AE(Layer):
    def __init__(self, N_mod=4, Code_N=3, Code_K=2):
        super(AE, self).__init__()
        self.N_mod = N_mod
        self.Code_N = Code_N
        self.Code_K = Code_K
        self.Encoder = Sequential([Dense(2*Code_N)]) # (I,Q) * length of a code block
        self.Decoder = Sequential([Dense(N_mod*Code_K)])
        
    def merge(self, x):
        Code_K = self.Code_K
        assert Code_K == x.shape[-2]
        x_list = [x[...,i,:] for i in range(x.shape[-2])]
        x = tf.concat(x_list, axis=-1)
        return x # N_sample, 2 * Code_K
    
    def split(self, x):
        Code_K = self.Code_K
        N_mod = self.N_mod
        assert N_mod == x.shape[-1] // Code_K
        x_list = []
        for i in range(Code_K):
            x_list.append(tf.reshape(x[...,N_mod*i:N_mod*(i+1)], (-1,1,N_mod)))
        x = tf.concat(x_list,axis=-2)
        return x
        
    def norm_awgn(self, x, noise_sig):
        # transmit power normalization
        #x = x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=0, keepdims=True))
        # if more than a symbol are used in a code block, additional power is needed.
        norm_pwr = tf.reduce_mean(tf.reduce_sum(tf.square(x),axis=-1)) / Code_N
        noise = tf.random.normal(x.shape) * noise_sig
        x = x + noise
        return x, norm_pwr
        
    def call(self, x, noise_sig):
        # Merge messages to a code block
        x = self.merge(x)        
        x = self.Encoder(x)
        x, norm_pwr = self.norm_awgn(x, noise_sig)        
        x = self.Decoder(x)
        x = self.split(x)
        return x, norm_pwr

def ae_train(
    N_bits=2,
    Code_K=3,
    Code_N=5,
    N_sample=1000,
    N_episodes=20,
    EbNo_dB=10):

    qam = QAM_Code(N_bits, Code_K, N_sample)
    #N_bits = qam.N_bits
    N_mod = qam.N_mod
    dataset = qam.dataset
    test_dataset = qam.test_dataset
    print(N_mod)
    
    # This is code rate
    model = AE(N_mod, Code_N, Code_K)
    # not softmax is used at the end of the network
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    optimizer = tf.keras.optimizers.Adam()

    w_list = []

    loss_train_l = []
    acc_train_l = []
    loss_test_l = []
    acc_test_l = []

    SNRdB = EbNo_dB + 10*np.log10(N_bits) 
    SNR = np.power(10, SNRdB / 10)
    noise_sig = 1 / np.sqrt(SNR)

    for e in range(N_episodes):    
        for step, (x, y) in enumerate(dataset):
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
        for step, (x, y) in enumerate(test_dataset):
            test_code_logits, pwr = model(x, noise_sig)
            test_loss_value = loss(y, test_code_logits) + tf.square(pwr - 1)
            accuracy.update_state(y, test_code_logits)
        loss_test_l.append(float(test_loss_value)) 
        acc_test_l.append(float(accuracy.result())) 

        if e % 10 == 0:
            #print(model.Encoder.weights[0].numpy(), model.Encoder.weights[1].numpy(), 
            #      model.Decoder.weights[0].numpy(), model.Decoder.weights[1].numpy())
            print('Epoch:', e, end=', ')
            train_loss_value = loss_train_l[-1]
            print('트레이닝의 손실과 마지막 스텝 정확도:', float(loss_value), acc_train_l[-1])
            print('테스트의 손실과 마지막 스텝 정확도:', float(test_loss_value), acc_test_l[-1])
            # print(model.trainable_weights)    

    qam_lib.plot_loss_acc_l(loss_train_l, loss_test_l, acc_train_l, acc_test_l)            

if __name__ == '__main__':
    print('Hello')    