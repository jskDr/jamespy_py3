from sklearn import model_selection, metrics
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, BatchNormalization, \
                         Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras import optimizers 

import kkeras


class CNN_Sequential(Sequential):
    def __init__(model, input_shape, nb_classes):
        # number of convolutional filters to use
        nb_filters = 8
        # size of pooling area for max pooling
        pool_size = (50, 50)
        # convolution kernel size
        kernel_size = (20, 20)

        super(model.__class__, model).__init__()

        model.add(Conv2D(nb_filters, kernel_size,
                         padding='valid',
                         input_shape=input_shape))
        model.add(BatchNormalization())
        # model.add(Activation('relu'))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=pool_size))
        # model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(4))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        # model.add(Activation('relu'))
        # model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])


class CNN_ActBn(Model):
    def __init__(model, input_shape, nb_classes):
        """
        Because I added Activation layer before Batchnormalizaiotn, 
        the performance is not good. 
        """
        # number of convolutional filters to use
        nb_filters = 8
        # size of pooling area for max pooling
        pool_size = (50, 50)
        # convolution kernel size
        kernel_size = (20, 20)

        # super(CNN, model).__init__()

        x = Input(input_shape)

        h = Conv2D(nb_filters, kernel_size, padding='valid', activation='tanh',
                   input_shape=input_shape)(x)
        h = BatchNormalization()(h)
        h = MaxPooling2D(pool_size=pool_size)(h)

        h = Flatten()(h)
        h = Dense(4, activation='tanh')(h)
        h = BatchNormalization()(h)

        y = Dense(nb_classes, activation='softmax')(h)

        super(model.__class__, model).__init__(x, y)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])


class CNN_relu(Model):
    def __init__(model, input_shape, nb_classes):
        # number of convolutional filters to use
        nb_filters = 8
        # size of pooling area for max pooling
        pool_size = (50, 50)
        # convolution kernel size
        kernel_size = (20, 20)

        # super(CNN, model).__init__()

        x = Input(input_shape)

        h = Conv2D(nb_filters, kernel_size, padding='valid',
                   input_shape=input_shape)(x)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Dropout(0.05)(h)
        h = MaxPooling2D(pool_size=pool_size)(h)

        h = Flatten()(h)
        h = Dense(4)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Dropout(0.05)(h)

        y = Dense(nb_classes, activation='softmax')(h)

        super(model.__class__, model).__init__(x, y)

        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])


class CNN_opt(Model):
    def __init__(model, input_shape, nb_classes):
        # number of convolutional filters to use
        nb_filters = 8
        # size of pooling area for max pooling
        pool_size = (50, 50)
        # convolution kernel size
        kernel_size = (20, 20)

        # super(CNN, model).__init__()

        x = Input(input_shape)

        h = Conv2D(nb_filters, kernel_size, padding='valid',
                   input_shape=input_shape)(x)
        h = BatchNormalization()(h)
        h = Activation('tanh')(h)
        h = Dropout(0.05)(h)
        h = MaxPooling2D(pool_size=pool_size)(h)

        h = Flatten()(h)
        h = Dense(4)(h)
        h = BatchNormalization()(h)
        h = Activation('tanh')(h)
        h = Dropout(0.05)(h)

        y = Dense(nb_classes, activation='softmax')(h)

        super(model.__class__, model).__init__(x, y)

        opt = optimizers.RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy',
                      #optimizer='adadelta',
                      optimizer=opt,
                      metrics=['accuracy'])


class CNN(Model):
    def __init__(model, input_shape, nb_classes):
        # number of convolutional filters to use
        nb_filters = 8
        # size of pooling area for max pooling
        pool_size = (50, 50)
        # convolution kernel size
        kernel_size = (20, 20)

        # super(CNN, model).__init__()

        x = Input(input_shape)

        h = Conv2D(nb_filters, kernel_size, padding='valid',
                   input_shape=input_shape)(x)
        h = BatchNormalization()(h)
        h = Activation('tanh')(h)
        h = Dropout(0.05)(h)
        h = MaxPooling2D(pool_size=pool_size)(h)

        h = Flatten()(h)
        h = Dense(4)(h)
        h = BatchNormalization()(h)
        h = Activation('tanh')(h)
        h = Dropout(0.05)(h)

        y = Dense(nb_classes, activation='softmax')(h)

        super(model.__class__, model).__init__(x, y)

        # opt = optimizers.RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      # optimizer=opt,
                      metrics=['accuracy'])
        

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


class System():
    def __init__(self, X, y, Lx, Ly, nb_classes=2, nb_epoch=5000, batch_size=128, verbose=0):

        data = Data(X, y, Lx, Ly, nb_classes)
        model = CNN(data.input_shape, nb_classes)
        # model = CNN_opt(data.input_shape, nb_classes)

        history = model.fit(data.X_train, data.Y_train, batch_size=batch_size, epochs=nb_epoch,
                            verbose=verbose, validation_data=(data.X_test, data.Y_test))
        score = model.evaluate(data.X_test, data.Y_test, verbose=0)

        print('Confusion metrix')
        Y_test_pred = model.predict(data.X_test, verbose=0)
        y_test_pred = np.argmax(Y_test_pred, axis=1)
        print(metrics.confusion_matrix(data.y_test, y_test_pred))

        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        kkeras.plot_acc(history)
        plt.show()
        kkeras.plot_loss(history)
