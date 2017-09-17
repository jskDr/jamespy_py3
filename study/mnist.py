# mnist.py
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Input, UpSampling2D
from keras.models import Model
from keras.utils import np_utils
from keras import backend as K

from . import recon_one as recon
import kkeras

np.random.seed(1337)  # for reproducibility


class CNN():
    def __init__(self):
        """
        By invoke run(), all code is executed.
        """
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        self.Org = (X_train, y_train), (X_test, y_test)
        self.Data = self.Org

    def run(self, nb_epoch=12):
        batch_size = 128
        nb_classes = 10
        nb_epoch = nb_epoch

        # input image dimensions
        img_rows, img_cols = 28, 28
        # number of convolutional filters to use
        nb_filters = 32
        # size of pooling area for max pooling
        pool_size = (2, 2)
        # convolution kernel size
        kernel_size = (3, 3)

        # the data, shuffled and split between train and test sets
        # (X_train, y_train), (X_test, y_test) = mnist.load_data()
        (X_train, y_train), (X_test, y_test) = self.Data

        if K.image_dim_ordering() == 'th':
            X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

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

        model = Sequential()

        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid',
                                input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=1, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, Y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])


def holo_transform(Org):
    # Transform X_train and X_test using hologram filtering.
    (X_train, dump_train), (X_test, dump_test) = Org

    print('Performing hologram transformation...')
    sim = recon.Simulator(X_train.shape[1:])
    X_train_holo = np.array([sim.diffract(x) for x in X_train])
    X_test_holo = np.array([sim.diffract(x) for x in X_test])

    Data = (X_train_holo, dump_train), (X_test_holo, dump_test)
    return Data


def recon_transform(Holo):
    """
    One-shot Recon with Hologram Image
    """
    (X_train_holo, dump_train), (X_test_holo, dump_test) = Holo

    print('Performing first-shot recon...')
    sim = recon.Simulator(X_train_holo.shape[1:])
    X_train_recon = np.array([np.abs(sim.reconstruct(x))
                              for x in X_train_holo])
    X_test_recon = np.array([np.abs(sim.reconstruct(x))
                             for x in X_test_holo])

    Data = (X_train_recon, dump_train), (X_test_recon, dump_test)
    return Data


def update2(x_train, x_test):
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
    x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))
    return x_train, x_test


class CNN_HOLO(CNN):
    def __init__(self):
        """
        This CNN includes hologram transformation.
        After transformation, CNN is working similarly.
        """
        super().__init__()

    def _holo_transform_r0(self):
        # Transform X_train and X_test using hologram filtering.
        (X_train, y_train), (X_test, y_test) = self.Org

        print('Performing hologram transformation...')
        sim = recon.Simulator(X_train.shape[1:])
        X_train_holo = np.array([sim.diffract(x) for x in X_train])
        X_test_holo = np.array([sim.diffract(x) for x in X_test])

        self.Data = (X_train_holo, y_train), (X_test_holo, y_test)
        self.Holo = self.Data

    def holo_transform(self):
        self.Data = holo_transform(self.Org)
        self.Holo = self.Data

    def holo_complex_transform(self):
        # Transform X_train and X_test using hologram filtering.
        (X_train, y_train), (X_test, y_test) = self.Org

        print('Performing complex hologram transformation...')
        sim = recon.Simulator(X_train.shape[1:])

        def holo(X_train):
            X_train_holo_abs_l = []
            X_train_holo_ang_l = []
            for x in X_train:
                X_train_h = sim.diffract_full(x)
                X_train_holo_abs_l.append(np.abs(X_train_h))
                X_train_holo_ang_l.append(np.angle(X_train_h))
            X_train_holo = np.zeros(
                (X_train.shape[0], 2, X_train.shape[1], X_train.shape[2]))
            X_train_holo[:, 0, :, :] = np.array(X_train_holo_abs_l)
            X_train_holo[:, 1, :, :] = np.array(X_train_holo_ang_l)
            return X_train_holo

        X_train_holo = holo(X_train)
        X_test_holo = holo(X_test)

        self.Data = (X_train_holo, y_train), (X_test_holo, y_test)
        self.Holo_complex = self.Data
        self.complex_flag = True

    def _recon_transform_r0(self):
        if not hasattr(self, 'Holo'):
            self.holo_transform()

        (X_train_holo, y_train), (X_test_holo, y_test) = self.Holo

        print('Performing first-shot recon...')
        sim = recon.Simulator(X_train_holo.shape[1:])
        X_train_recon = np.array([np.abs(sim.reconstruct(x))
                                  for x in X_train_holo])
        X_test_recon = np.array([np.abs(sim.reconstruct(x))
                                 for x in X_test_holo])

        self.Data = (X_train_recon, y_train), (X_test_recon, y_test)
        self.Recon = self.Data

    def recon_transform(self):
        """
        self.recon_transform is performed using recon_transform()
        """
        if not hasattr(self, 'Holo'):
            self.holo_transform()
        self.Data = recon_transform(self.Holo)
        self.Recon = self.Data

    def run(self, nb_epoch=12):
        if hasattr(self, 'complex_flag') and self.complex_flag:
            print('Classification for complex input data...')
            self.run_complex(nb_epoch=nb_epoch)
        else:
            print('Classificaiton for real input data...')
            super().run(nb_epoch=nb_epoch)

    def run_complex(self, nb_epoch=12, kernel_size_1=None):
        batch_size = 128
        nb_classes = 10
        nb_epoch = nb_epoch

        # input image dimensions
        img_rows, img_cols = 28, 28
        # number of convolutional filters to use
        nb_filters = 32
        # size of pooling area for max pooling
        pool_size = (2, 2)
        # convolution kernel size
        kernel_size = (3, 3)
        if kernel_size_1 is None:
            kernel_size_1 = kernel_size

        # the data, shuffled and split between train and test sets
        # (X_train, y_train), (X_test, y_test) = mnist.load_data()
        (X_train, y_train), (X_test, y_test) = self.Data
        # number of input data sets - abs and angle
        nb_rgb = X_train.shape[1]

        if K.image_dim_ordering() == 'th':
            input_shape = (nb_rgb, img_rows, img_cols)
        else:
            raise ValueError('Only th ordering is support yet for RGB data')

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

        model = Sequential()

        model.add(Convolution2D(nb_filters, kernel_size_1[0], kernel_size_1[1],
                                border_mode='valid',
                                input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])

        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=1, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, Y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])


class AE:
    def __init__(self):
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # Modify input and output data to be appropritate for AE
        self.Org = (X_train, X_train), (X_test, X_test)
        self.Data = self.Org

    def modeling(self):
        input_img = Input(shape=(1, 28, 28))
        # set-1
        x = Convolution2D(16, 3, 3, activation='relu',
                          border_mode='same')(input_img)  # 16,28,28
        x = MaxPooling2D((2, 2), border_mode='same')(x)  # 16,14,14
        x = Dropout(0.25)(x)  # Use dropout after maxpolling

        # set-2
        x = Convolution2D(8, 3, 3, activation='relu',
                          border_mode='same')(x)  # 8,14,14
        x = MaxPooling2D((2, 2), border_mode='same')(x)  # 8,7,7
        x = Dropout(0.25)(x)  # Use dropout after maxpolling

        # set-3
        x = Convolution2D(8, 3, 3, activation='relu',
                          border_mode='same')(x)  # 8,7,7
        encoded = x

        x = Convolution2D(8, 3, 3, activation='relu',
                          border_mode='same')(encoded)  # 8,7,7
        # x = Dropout(0.25)(x) # Use dropout after maxpolling

        x = UpSampling2D((2, 2))(x)  # 8,14,14
        x = Convolution2D(8, 3, 3, activation='relu',
                          border_mode='same')(x)  # 8,14,14
        # x = Dropout(0.25)(x) # Use dropout after maxpolling

        x = UpSampling2D((2, 2))(x)  # 8, 28, 28
        x = Convolution2D(16, 3, 3, activation='relu',
                          border_mode='same')(x)  # 16, 28, 28
        # x = Dropout(0.25)(x) # Use dropout after maxpolling
        decoded = Convolution2D(
            1, 3, 3, activation='sigmoid', border_mode='same')(x)  # 1, 28, 28

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        self.autoencoder = autoencoder

    def run(self, nb_epoch=100):
        (x_train_in, x_train), (x_test_in, x_test) = self.Data

        x_train_in, x_test_in = update2(x_train_in, x_test_in)
        x_train, x_test = update2(x_train, x_test)

        self.modeling()
        autoencoder = self.autoencoder
        history = autoencoder.fit(x_train_in, x_train,
                                  nb_epoch=nb_epoch,
                                  batch_size=128,
                                  shuffle=True,
                                  verbose=1,
                                  validation_data=(x_test, x_test))
        kkeras.plot_loss(history)

        self.imshow()

    #def imshow(self, x_test, x_test_in):
    def imshow(self):
        (_, _), (x_test_in, x_test) = self.Data
        x_test_in, x_test = update2(x_test_in, x_test)
        autoencoder = self.autoencoder
        decoded_imgs = autoencoder.predict(x_test_in)

        n = 10
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + n + 1)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


class _AE_HOLO_r0(AE):
    def __init__(self):
        """
        Hologram transformation is performed
        """
        super().__init__()

    def holo_transform(self):
        (x_train, _), (x_test, _) = self.Org
        (x_train_in, _), (x_test_in, _) = holo_transform(self.Org)
        self.Data = (x_train_in, x_train), (x_test_in, x_test)
        self.Holo = self.Data

    def imshow(self):
        (_, _), (x_test_in, x_test) = self.Data
        x_test_in, x_test = update2(x_test_in, x_test)
        autoencoder = self.autoencoder
        decoded_imgs = autoencoder.predict(x_test_in)

        n = 10
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(3, n, n + i + 1)
            plt.imshow(x_test_in[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(3, n, n * 2 + i + 1)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


class AE_HOLO(AE):
    def __init__(self):
        """
        Hologram transformation is performed
        """
        super().__init__()
        (x_train, _), (x_test, _) = self.Org
        x_train_in, x_test_in = x_train, x_test
        self.Org = (x_train_in, x_train), (x_test_in, x_test)

    def holo_transform(self):
        CNN_HOLO.holo_transform(self)

    def recon_transform(self):
        CNN_HOLO.recon_transform(self)

    def imshow(self):
        (_, _), (x_test_in, x_test) = self.Data
        x_test_in, x_test = update2(x_test_in, x_test)
        autoencoder = self.autoencoder
        decoded_imgs = autoencoder.predict(x_test_in)

        n = 10
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(x_test[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(3, n, n + i + 1)
            plt.imshow(x_test_in[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(3, n, n * 2 + i + 1)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()