from sklearn import model_selection, metrics
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras import backend as K
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, \
    Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import kkeras


class CNN(Model):
    def __init__(model, nb_classes, in_shape=None):
        model.nb_classes = nb_classes
        model.in_shape = in_shape
        model.build_model()
        super().__init__(model.x, model.y)
        model.compile()

    def build_model(model):
        nb_classes = model.nb_classes
        in_shape = model.in_shape

        # number of convolutional filters to use
        nb_filters = 8
        # size of pooling area for max pooling
        pool_size = (50, 50)
        # convolution kernel size
        kernel_size = (20, 20)

        # super(CNN, model).__init__()

        x = Input(in_shape)

        h = Conv2D(nb_filters, kernel_size, input_shape=in_shape)(x)
        h = BatchNormalization()(h)
        h = Activation('tanh')(h)
        h = Dropout(0.05)(h)
        h = MaxPooling2D(pool_size=pool_size)(h)
        h = Flatten()(h)
        z_cl = h

        h = Dense(4)(h)
        h = BatchNormalization()(h)
        h = Activation('tanh')(h)
        h = Dropout(0.05)(h)
        z_fl = h

        y = Dense(nb_classes, activation='softmax')(h)

        model.cl_part = Model(x, z_cl)
        model.fl_part = Model(x, z_fl)

        model.x, model.y = x, y

    def compile(model):
        Model.compile(model, loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])


class CNN_LENET(CNN):
    def __init__(model, nb_classes, in_shape):
        super().__init__(nb_classes, in_shape=in_shape)

    def build_model(model):
        """
        Tip
        ---
        Make a callable object using class
        - https://stackoverflow.com/questions/15719172/overload-operator-in-python
        -- def __call__(self, ...)
        """
        nb_classes = model.nb_classes
        in_shape = model.in_shape

        # super().__init__()
        x = Input(in_shape)
        h = Conv2D(32, kernel_size=(3, 3), activation='relu',
                   input_shape=in_shape)(x)
        h = Conv2D(64, (3, 3), activation='relu')(h)
        h = MaxPooling2D(pool_size=(2, 2))(h)
        h = Dropout(0.25)(h)
        h = Flatten()(h)
        h = Dense(128, activation='relu')(h)
        h = Dropout(0.5)(h)
        y = Dense(nb_classes, activation='softmax', name='preds')(h)

        model.x, model.y = x, y

    def compile(model):
        Model.compile(model, loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])


def cnn_lenet_org(nb_classes, in_shape):
    x = Input(in_shape)
    h = Conv2D(32, kernel_size=(3, 3), activation='relu',
               input_shape=in_shape)(x)
    h = Conv2D(64, (3, 3), activation='relu')(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = Dropout(0.25)(h)
    h = Flatten()(h)
    h = Dense(128, activation='relu')(h)
    h = Dropout(0.5)(h)
    y = Dense(nb_classes, activation='softmax', name='preds')(h)
    model = Model(x, y)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model


cnn_lenet = cnn_lenet_org


def cnn_lenet_32(nb_classes, in_shape):
    x = Input(in_shape)
    h = Conv2D(32, kernel_size=(3, 3), activation='relu',
               input_shape=in_shape)(x)
    h = Conv2D(64, (3, 3), activation='relu')(h)
    h = MaxPooling2D(pool_size=(2, 2))(h)
    h = Dropout(0.25)(h)
    h = Flatten()(h)
    h = Dense(32, activation='relu')(h)
    h = Dropout(0.5)(h)
    y = Dense(nb_classes, activation='softmax', name='preds')(h)
    model = Model(x, y)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model


class Data():
    def __init__(self, X, y, img_rows, img_cols, nb_classes):
        """
        X is originally vector. Hence, it will be transformed to 2D images with a channel (i.e, 3D).
        """
        if K.image_dim_ordering() == 'th':
            X = X.reshape(X.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            X = X.reshape(X.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        # the data, shuffled and split between train and test sets
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=0)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        self.X_train, self.X_test = X_train, X_test
        self.Y_train, self.Y_test = Y_train, Y_test
        self.y_train, self.y_test = y_train, y_test
        self.input_shape = input_shape


class DataSet():
    def __init__(self, X, y, nb_classes, scaling=True, test_size=0.2, random_state=0):
        """
        X is originally vector. Hence, it will be transformed to 2D images with a channel (i.e, 3D).
        """
        self.X = X
        self.add_channels()

        X = self.X
        # the data, shuffled and split between train and test sets
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=random_state)

        print(X_train.shape, y_train.shape)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        if scaling:
            # scaling to have (0, 1) for each feature (each pixel)
            scaler = MinMaxScaler()
            n = X_train.shape[0]
            X_train = scaler.fit_transform(X_train.reshape(n, -1)).reshape(X_train.shape)
            n = X_test.shape[0]
            X_test = scaler.transform(X_test.reshape(n, -1)).reshape(X_test.shape)
            self.scaler = scaler

        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        self.X_train, self.X_test = X_train, X_test
        self.Y_train, self.Y_test = Y_train, Y_test
        self.y_train, self.y_test = y_train, y_test
        # self.input_shape = input_shape

    def add_channels(self):
        X = self.X

        N, img_rows, img_cols = X.shape

        if K.image_dim_ordering() == 'th':
            X = X.reshape(X.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            X = X.reshape(X.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        self.X = X
        self.input_shape = input_shape
        
        
class Machine():
    def __init__(self, X, y, Lx, Ly, nb_classes=2, fig=True):

        data = Data(X, y, Lx, Ly, nb_classes)
        print('data.input_shape', data.input_shape)
        model = CNN(nb_classes, data.input_shape)

        self.data = data
        self.model = model
        self.fig = fig 
        
    def fit(self, nb_epoch=10, batch_size=128, verbose=1):
        data = self.data
        model = self.model

        history = model.fit(data.X_train, data.Y_train, batch_size=batch_size, epochs=nb_epoch,
                            verbose=verbose, validation_data=(data.X_test, data.Y_test))
        return history

    def run(self, nb_epoch=10, batch_size=128, verbose=1):
        data = self.data
        model = self.model
        fig = self.fig

        history = self.fit(nb_epoch=nb_epoch, batch_size=batch_size, verbose=verbose)
        model.save_weights('dl_model.h5') 

        score = model.evaluate(data.X_test, data.Y_test, verbose=0)

        print('Confusion matrix')
        Y_test_pred = model.predict(data.X_test, verbose=0)
        y_test_pred = np.argmax(Y_test_pred, axis=1)
        print(metrics.confusion_matrix(data.y_test, y_test_pred))

        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        if fig:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            kkeras.plot_acc(history)
            plt.subplot(1, 2, 2)
            kkeras.plot_loss(history)
            plt.show()
        
        self.history = history


class Machine_cnn_lenet(Machine):
    def __init__(self, X, y, nb_classes=2, fig=True):

        data = DataSet(X, y, nb_classes)
        model = cnn_lenet(nb_classes=nb_classes, in_shape=data.input_shape)

        self.data = data
        self.model = model
        self.fig = fig


class Machine_Generator(Machine_cnn_lenet):
    def __init__(self, X, y, nb_classes=2, steps_per_epoch=10, fig=True):
        super().__init__(X, y, nb_classes=nb_classes, fig=fig)
        self.set_generator(steps_per_epoch)
        
    def set_generator(self, steps_per_epoch=10):
        self.generator = ImageDataGenerator()
        self.generator.fit(self.data.X_train, seed=0)
        self.steps_per_epoch = steps_per_epoch
        
    def fit(self, nb_epoch=10, batch_size=64, verbose=1):
        model = self.model
        data = self.data
        generator = self.generator
        steps_per_epoch = self.steps_per_epoch
        
        history = model.fit_generator(generator.flow(data.X_train, data.Y_train, batch_size=batch_size), 
                    epochs=nb_epoch, steps_per_epoch=steps_per_epoch,
                    validation_data=(data.X_test, data.Y_test))  
        
        return history


def fit_scaling(scaler, x_train):
    s1 = x_train.shape[0]
    return scaler.fit_transform(x_train.reshape(s1, -1)).reshape(x_train.shape)


def scaling(scaler, x_train):
    s1 = x_train.shape[0]
    return scaler.transform(x_train.reshape(s1, -1)).reshape(x_train.shape)


def rescaling(scaler, x_train):
    s1 = x_train.shape[0]
    return scaler.inverse_transform(x_train.reshape(s1, -1)).reshape(x_train.shape)
