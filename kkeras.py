# kkeras.py
# Kkeras is the most recent version and kkeras_rx versions aer old versions.

import numpy as np
import matplotlib.pyplot as plt
# np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, \
    MaxPooling1D, Convolution1D, Flatten
from keras.optimizers import RMSprop  # ,SGD, Adam,
from keras.utils import np_utils
from keras import callbacks
from keras.regularizers import l2

import kutil


def plot_acc(history):
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()


def plot_loss(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()


def plot_history(history):
    plt.subplot(2,1,1)
    plot_acc(history)
    plt.subplot(2,1,2)
    plot_loss(history)

class MLPC():
    """
    Multi layer perceptron classification
    Define multi layer perceptron using Keras
    """

    def __init__(self, l=[49, 30, 10, 3]):
        """
        modeling is performed in self.modeling()
        instead of direct performing in this function. 
        """
        model = self.modeling(l=l)
        model.compile(loss='categorical_crossentropy',
                      optimizer=RMSprop(),
                      metrics=['accuracy'])
        self.model = model

    def modeling(self, l=[49, 30, 10, 3]):
        """
        generate model
        """
        model = Sequential()
        model.add(Dense(l[1], input_shape=(l[0],)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(l[2]))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(l[3]))
        model.add(Activation('softmax'))

        return model

    def X_reshape(self, X_train_2D, X_val_2D=None):
        """
        Used for child classes such as convolutional networks 
        When the number of arguments is only one, 
        only one values will be returned. 
        """
        if X_val_2D is None:
            return X_train_2D
        else:
            return X_train_2D, X_val_2D

    def fit(self, X_train, y_train, X_val, y_val, nb_classes=None, batch_size=10, nb_epoch=20, verbose=0):
        model = self.model

        if nb_classes is None:
            nb_classes = max(set(y_train)) + 1

        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_val = np_utils.to_categorical(y_val, nb_classes)

        model.reset_states()
        earlyStopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=3, verbose=verbose, mode='auto')

        X_train, X_val = self.X_reshape(X_train, X_val)
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size, nb_epoch=nb_epoch,
                            verbose=verbose, validation_data=(X_val, Y_val), callbacks=[earlyStopping])

        self.nb_classes = nb_classes
        self.history = history

    def score(self, X_test, y_test):
        model = self.model
        nb_classes = self.nb_classes

        Y_test = np_utils.to_categorical(y_test, nb_classes)

        X_test = self.X_reshape(X_test)
        score = model.evaluate(X_test, Y_test, verbose=0)

        return score[1]

    def evaluate(self, X_test, y_test):
        model = self.model
        nb_classes = self.nb_classes

        Y_test = np_utils.to_categorical(y_test, nb_classes)

        X_test = self.X_reshape(X_test)
        score_l = model.evaluate(X_test, Y_test, verbose=0)

        return score_l  # loss, accuracy

    def predict(self, X_test):
        model = self.model
        nb_classes = self.nb_classes

        #Y_test = np_utils.to_categorical(y_test, nb_classes)

        X_test = self.X_reshape(X_test)
        Y_pred = model.predict(X_test, verbose=0)
        y_pred = np_utils.categorical_probas_to_classes(Y_pred)

        return y_pred


class CNNC(MLPC):

    def __init__(self, n_cv_flt=2, n_cv_ln=3, cv_activation='relu', l=[49, 30, 10, 3], mp=1):
        """
        Convolutional neural networks 
        """
        self.n_cv_flt = n_cv_flt
        self.n_cv_ln = n_cv_ln
        self.cv_activation = cv_activation
        self.mp = mp
        super().__init__(l=l)

    def modeling(self, l=[49, 30, 10, 3]):
        """
        generate model
        """
        n_cv_flt, n_cv_ln = self.n_cv_flt, self.n_cv_ln
        cv_activation = self.cv_activation
        mp = self.mp

        model = Sequential()

        # Direct: input_shape should be (l,0) not (l)
        # if l, it assume a scalar for an input feature.
        #model.add(Dense( l[1], input_shape=(l[0],)))

        # Convolution
        # print( "n_cv_flt, n_cv_ln, cv_activation", n_cv_flt, n_cv_ln, cv_activation)
        model.add(Convolution1D(n_cv_flt, n_cv_ln, activation=cv_activation,
                                border_mode='same', input_shape=(l[0], 1)))
        model.add(MaxPooling1D(mp))
        model.add(Flatten())
        model.add(Dense(l[1]))

        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(l[2]))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(l[3]))
        model.add(Activation('softmax'))
        return model

    def X_reshape(self, X_train_2D, X_val_2D=None):
        """
        1D convolution and 2D convolution ordering different
        1D: -1,1 (e.g., 50,1 and 50,3 for BK, RGB), input_shape = (50,1) or (50,3) 
        2D: 1,n,m (e.g., 1,128,128 and 3,128,128 for BK, RGB), input_shape = (1,128,128) or (3,128,128)
        """
        X_train_3D = X_train_2D.reshape(X_train_2D.shape[0], -1, 1)

        if X_val_2D is None:
            return X_train_3D
        else:
            X_val_3D = X_val_2D.reshape(X_val_2D.shape[0], -1, 1)
            return X_train_3D, X_val_3D


class _CNNC_Name_r0(CNNC):

    def modeling(self, l=[49, 30, 10, 3]):
        """
        generate model
        """
        self.c_name = 'conv'

        n_cv_flt, n_cv_ln = self.n_cv_flt, self.n_cv_ln
        cv_activation = self.cv_activation

        model = Sequential()

        # Direct: input_shape should be (l,0) not (l)
        # if l, it assume a scalar for an input feature.
        #model.add(Dense( l[1], input_shape=(l[0],)))

        # Convolution
        print("n_cv_flt, n_cv_ln, cv_activation",
              n_cv_flt, n_cv_ln, cv_activation)
        # model.add(Convolution1D( n_cv_flt, n_cv_ln, activation=cv_activation,
        #	border_mode='same', input_shape=(1, l[0]), name = 'conv'))
        model.add(Convolution1D(n_cv_flt, n_cv_ln, activation=cv_activation,
                                border_mode='same', input_shape=(l[0], 1), name=self.c_name))
        model.add(Flatten())
        model.add(Dense(l[1]))

        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(l[2]))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(l[3]))
        model.add(Activation('softmax'))

        self.layer_dict = dict([(layer.name, layer) for layer in model.layers])

        return model

    def get_layer(self, name):
        return self.layer_dict[name]

    def self_c_wb(self):
        self.c_w, self.c_b = self.get_layer(self.c_name).get_weights()
        return self

    def get_c_wb(self):
        self.self_c_wb()
        return self.c_w, self.c_b


class CNNC_Name(CNNC):

    def modeling(self, l=[49, 30, 10, 3]):
        """
        generate model
        """
        self.c_name = 'conv'
        mp = self.mp

        n_cv_flt, n_cv_ln = self.n_cv_flt, self.n_cv_ln
        cv_activation = self.cv_activation

        model = Sequential()

        # Direct: input_shape should be (l,0) not (l)
        # if l, it assume a scalar for an input feature.
        #model.add(Dense( l[1], input_shape=(l[0],)))

        # Convolution
        print("n_cv_flt, n_cv_ln, cv_activation",
              n_cv_flt, n_cv_ln, cv_activation)
        # model.add(Convolution1D( n_cv_flt, n_cv_ln, activation=cv_activation,
        #	border_mode='same', input_shape=(1, l[0]), name = 'conv'))
        model.add(Convolution1D(n_cv_flt, n_cv_ln, activation=cv_activation,
                                border_mode='same', input_shape=(l[0], 1), name=self.c_name))
        model.add(MaxPooling1D(mp))
        model.add(Flatten())
        model.add(Dense(l[1]))

        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(l[2]))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(l[3]))
        model.add(Activation('softmax'))

        self.layer_dict = dict([(layer.name, layer) for layer in model.layers])

        return model

    def get_layer(self, name):
        return self.layer_dict[name]

    def self_c_wb(self):
        self.c_w, self.c_b = self.get_layer(self.c_name).get_weights()
        return self

    def get_c_wb(self):
        self.self_c_wb()
        return self.c_w, self.c_b


class CNNC_Name_ConvOut(CNNC):

    def modeling(self, l=[49, 30, 10, 3]):
        """
        generate model
        """
        self.c_name = 'conv'

        n_cv_flt, n_cv_ln = self.n_cv_flt, self.n_cv_ln
        cv_activation = self.cv_activation

        model = Sequential()

        # Direct: input_shape should be (l,0) not (l)
        # if l, it assume a scalar for an input feature.
        #model.add(Dense( l[1], input_shape=(l[0],)))

        # Convolution
        print("n_cv_flt, n_cv_ln, cv_activation",
              n_cv_flt, n_cv_ln, cv_activation)
        # model.add(Convolution1D( n_cv_flt, n_cv_ln, activation=cv_activation,
        #	border_mode='same', input_shape=(1, l[0]), name = 'conv'))
        model.add(Convolution1D(n_cv_flt, n_cv_ln, activation=cv_activation,
                                border_mode='same', input_shape=(l[0], 1), name=self.c_name))
        # model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(l[1]))

        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(l[2]))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(l[3]))
        model.add(Activation('softmax'))

        self.layer_dict = dict([(layer.name, layer) for layer in model.layers])

        convout_model = Sequential()
        convout_model.add(model.layers[0])

        self.convout_model = convout_model

        return model

    def get_layer(self, name):
        return self.layer_dict[name]

    def self_c_wb(self):
        self.c_w, self.c_b = self.get_layer(self.c_name).get_weights()
        return self

    def get_c_wb(self):
        self.self_c_wb()
        return self.c_w, self.c_b


class CNNC_Name_Border(CNNC):

    def __init__(self, n_cv_flt=2, n_cv_ln=3, cv_activation='relu', l=[49, 30, 10, 3],
                 border_mode='same'):
        """
        Convolutional neural networks 
        """
        self.border_mode = border_mode

        super().__init__(n_cv_flt=n_cv_flt, n_cv_ln=n_cv_ln,
                         cv_activation=cv_activation, l=l)

    def modeling(self, l=[49, 30, 10, 3]):
        """
        generate model
        """
        border_mode = self.border_mode
        self.c_name = 'conv'

        n_cv_flt, n_cv_ln = self.n_cv_flt, self.n_cv_ln
        cv_activation = self.cv_activation

        model = Sequential()

        # Direct: input_shape should be (l,0) not (l)
        # if l, it assume a scalar for an input feature.
        #model.add(Dense( l[1], input_shape=(l[0],)))

        # Convolution
        print("n_cv_flt, n_cv_ln, cv_activation",
              n_cv_flt, n_cv_ln, cv_activation)
        # model.add(Convolution1D( n_cv_flt, n_cv_ln, activation=cv_activation,
        #	border_mode='same', input_shape=(1, l[0]), name = 'conv'))
        model.add(Convolution1D(n_cv_flt, n_cv_ln, activation=cv_activation,
                                border_mode=border_mode, input_shape=(l[0], 1), name=self.c_name))
        # model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(l[1]))

        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(l[2]))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(l[3]))
        model.add(Activation('softmax'))

        self.layer_dict = dict([(layer.name, layer) for layer in model.layers])

        return model

    def get_layer(self, name):
        return self.layer_dict[name]

    def self_c_wb(self):
        self.c_w, self.c_b = self.get_layer(self.c_name).get_weights()
        return self

    def get_c_wb(self):
        self.self_c_wb()
        return self.c_w, self.c_b


class MLPC_Name(MLPC):

    def modeling(self, l=[49, 30, 10, 3]):
        """
        generate model
        """
        self.c_name = 'dense'

        model = Sequential()
        # Direct: input_shape should be (l,0) not (l)
        # if l, it assume a scalar for an input feature.
        #model.add(Dense( l[1], input_shape=(l[0],)))

        # DNN
        model.add(Dense(l[1], input_shape=(l[0],), name=self.c_name))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(l[2]))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(l[3]))
        model.add(Activation('softmax'))

        self.layer_dict = dict([(layer.name, layer) for layer in model.layers])

        return model

    def get_layer(self, name):
        return self.layer_dict[name]

    def self_c_wb(self):
        self.c_w, self.c_b = self.get_layer(self.c_name).get_weights()
        return self

    def get_c_wb(self):
        self.self_c_wb()
        return self.c_w, self.c_b


class MLPR():  # Regression
    """
    Multi layer perceptron regression
    Define multi layer perceptron using Keras
    """

    def __init__(self, l=[49, 30, 10, 1]):
        """
        modeling is performed in self.modeling()
        instead of direct performing in this function. 
        """
        model = self.modeling(l=l)
        # model.compile(loss='categorical_crossentropy',
        #			  optimizer=RMSprop(),
        #			  metrics=['accuracy'])
        # , metrics=['accuracy'])
        model.compile(loss='mean_squared_error', optimizer='adam')
        self.model = model

    def modeling(self, l=[2121, 100, 50, 10, 1]):
        """
        generate model
        """
        model = Sequential()
        model.add(Dense(l[1], input_shape=(l[0],)))
        model.add(Activation('relu'))
        # model.add(Dropout(0.4))
        model.add(Dense(l[2]))
        model.add(Activation('relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(l[3]))
        model.add(Activation('relu'))
        model.add(Dense(l[4]))

        return model

    def X_reshape(self, X_train_2D, X_val_2D=None):
        """
        Used for child classes such as convolutional networks 
        When the number of arguments is only one, 
        only one values will be returned. 
        """
        if X_val_2D is None:
            return X_train_2D
        else:
            return X_train_2D, X_val_2D

    def fit(self, X_train, Y_train, X_val, Y_val, batch_size=10, nb_epoch=20, verbose=0):
        model = self.model

        # if nb_classes is None:
        #	nb_classes = max( set( y_train)) + 1

        #Y_train = np_utils.to_categorical(y_train, nb_classes)
        #Y_val = np_utils.to_categorical(y_val, nb_classes)

        model.reset_states()
        earlyStopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=3, verbose=verbose, mode='auto')

        X_train, X_val = self.X_reshape(X_train, X_val)
        history = model.fit(X_train, Y_train,
                            batch_size=batch_size, nb_epoch=nb_epoch,
                            verbose=verbose, validation_data=(X_val, Y_val), callbacks=[earlyStopping])

        #self.nb_classes = nb_classes
        self.history = history

    def score(self, X_test, Y_test, batch_size=32, verbose=0):
        model = self.model
        X_test = self.X_reshape(X_test)
        Y_test_pred = model.predict(
            X_test, batch_size=batch_size, verbose=verbose)

        return kutil.regress_show4(Y_test, Y_test_pred)

    def predict(self, X_new, batch_size=32, verbose=0):
        model = self.model
        #nb_classes = self.nb_classes

        #Y_test = np_utils.to_categorical(y_test, nb_classes)

        X_new = self.X_reshape(X_new)
        y_new = model.predict(X_new, batch_size=batch_size, verbose=verbose)
        return y_new


class MLPR_N(MLPR):  # Regression with N layers
    """
    Multi layer perceptron regression
    Define multi layer perceptron using Keras
    """

    def __init__(self, l=[49, 30, 10, 1], l2_param=0):
        """
        modeling is performed in self.modeling()
        instead of direct performing in this function. 
        """
        self.l2_param = l2_param
        super().__init__(l=l)

    def modeling(self, l=[2121, 100, 50, 10, 1]):
        """
        Generate a model with the give number of layers.
        Previously, always a 5 layer model is generated. 
        Now it is changed to make adaptive number of layers. 
        If l = [2121, 1], it is linear regressoin method. 

        - l2_param is a self parameter rather than an input parameter.
        """
        l2_param = self.l2_param

        model = Sequential()
        model.add(Dense(l[1], input_shape=(l[0],)))
        model.regularizers = [l2(l2_param)]

        for n_w_l in l[2:]:
            model.add(Activation('relu'))
            # model.add(Dropout(0.4))
            # model.add(Dense( n_w_l, W_regularizer = l2(.01)))
            model.add(Dense(n_w_l))

        return model
