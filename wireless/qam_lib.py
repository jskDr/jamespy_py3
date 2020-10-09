import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

class QAM:
    def __init__(self, N_bits=2, N_sample=10000):
        """ N_bits is #bits used in modulation so that N_mod = 2 ** N_bits is modulation order. 
        """
        # Preparing input data
        self.N_bits = N_bits
        self.N_mod = 2 ** N_bits
        
        # 16QAM with one-hot encoding. So, 16 different messages are created. 
        self.S = np.random.randint(0, self.N_mod, (N_sample)) 
        
        # Preparing trainning and test data
        # tf.one_hot is used since it support one-hot function. I am wondering how to use this function inside the neural net. 
        self.S_onehot = tf.one_hot(self.S, self.N_mod, dtype='float32').numpy()
        
        print(self.S.shape, self.S_onehot.shape)

        # Don't specify too much. I can edit this code directly for later. 
        X_train, X_test, y_train, y_test = train_test_split(self.S_onehot, self.S, test_size=0.2)
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        self.dataset = dataset.shuffle(buffer_size=1024).batch(64) # 64 --> 4 * N_mod
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        self.test_dataset = test_dataset.shuffle(buffer_size=1024).batch(64)  # 64 --> 4 * N_mod   


class QAM_Code:
    def __init__(self, N_bits=2, Code_K=1, N_sample=10000):
        """ N_bits is #bits used in modulation so that N_mod = 2 ** N_bits is modulation order. 
        """
        # Preparing input data
        self.N_bits = N_bits
        self.N_mod = 2 ** N_bits
        
        # 16QAM with one-hot encoding. So, 16 different messages are created. 
        self.S = np.random.randint(0, self.N_mod, (N_sample, Code_K)) 
        
        # Preparing trainning and test data
        # tf.one_hot is used since it support one-hot function. I am wondering how to use this function inside the neural net. 
        S_onehot = tf.one_hot(self.S, self.N_mod, dtype='float32').numpy()
        self.S_onehot = tf.concat(S_onehot, axis=-2)

        print(self.S.shape, self.S_onehot.shape)

        # Don't specify too much. I can edit this code directly for later. 
        X_train, X_test, y_train, y_test = train_test_split(self.S_onehot, self.S, test_size=0.2)
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        self.dataset = dataset.shuffle(buffer_size=1024).batch(64) # 64 --> 4 * N_mod
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        self.test_dataset = test_dataset.shuffle(buffer_size=1024).batch(64)  # 64 --> 4 * N_mod  


def plot_loss_acc_l(loss_train_l, loss_test_l, acc_train_l, acc_test_l):
    """
    - 마지막에 plt.show()를 포함했다. 이를 통해 plot 결과가 중간에 나오도록 했다.
    """
    plt.plot(loss_train_l, label='Loss@Train')
    plt.plot(loss_test_l, label='Loss@Test')
    plt.plot(acc_train_l, label='Acc@Train')
    plt.plot(acc_test_l, label='Acc@Test')
    plt.xlabel('Episode')
    plt.ylabel('Loss/Acc')
    plt.legend(loc=0)
    plt.show()

def plot_map(model, N_bits):
    N_mod = 2 ** N_bits
    S = np.array(range(N_mod))
    S_onehot = tf.one_hot(S, N_mod, dtype='float32').numpy()
    X = model.Encoder(S_onehot)
    X = X.numpy()
    plt.figure(figsize=(5,5))
    for i in range(N_mod):
        plt.plot(X[i,0], X[i,1], 'o')
        plt.text(X[i,0], X[i,1], f'{np.binary_repr(i,N_bits)}')
    plt.grid()
    plt.title('Modulation Map')
    plt.show()
    
    print('Check power:')
    print(np.sum(np.power(X, 2),axis=-1))

def show_SER(model, noise_sig, N_mod=2, N_samples=1000000):
    S = np.random.randint(0, N_mod, N_samples) 
    S_onehot = tf.one_hot(S, N_mod, dtype='float32').numpy()
    R_logits, pwr = model(S_onehot, noise_sig)
    print(S_onehot.shape, R_logits.shape)

    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    accuracy.update_state(S, R_logits)

    print('Symbol error rate = ', float(accuracy.result()))
    print('Power constraint = ', pwr.numpy())
    
    print('[Modulation MAP]')
    plot_map(model, N_mod)   