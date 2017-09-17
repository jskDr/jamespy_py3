# mnist.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from keras.datasets import mnist, cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.layers import Conv2D  # Convolution2D
from keras.layers import Input, UpSampling2D
from keras.models import Model
from keras.utils import np_utils
from keras import backend as K

from mgh import recon
from . import kkeras

K.set_image_data_format('channels_first')
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

        # number of convolutional filters to use
        nb_filters = 32
        # size of pooling area for max pooling
        pool_size = (2, 2)
        # convolution kernel size
        kernel_size = (3, 3)

        # the data, shuffled and split between train and test sets
        # (X_train, y_train), (X_test, y_test) = mnist.load_data()
        (X_train, y_train), (X_test, y_test) = self.Data

        ny, nx = X_train.shape[1:]
        # input image dimensions
        img_rows, img_cols = ny, nx

        if K.image_data_format() == 'channels_first':
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

        model.add(Conv2D(nb_filters, kernel_size,
                         padding='valid',
                         input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(nb_filters, kernel_size))
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

        model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
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


def holo_complex_transform(Org):
    (X_train, y_train), (X_test, y_test) = Org

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

    Data = (X_train_holo, y_train), (X_test_holo, y_test)
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

    ny, nx = x_train.shape[1:]
    x_train = np.reshape(x_train, (len(x_train), 1, ny, nx))
    x_test = np.reshape(x_test, (len(x_test), 1, ny, nx))
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

    def holo_complex_transform_r0(self):
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

    def holo_complex_transform(self):
        # Transform X_train and X_test using hologram filtering
        self.Data = holo_complex_transform(self.Org)
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

        ny, nx = self.Data.shape[1:]
        # input image dimensions
        img_rows, img_cols = ny, nx

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

        model.add(Conv2D(nb_filters, kernel_size_1,
                         padding='valid',
                         input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(nb_filters, kernel_size))
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

        model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                  verbose=1, validation_data=(X_test, Y_test))
        score = model.evaluate(X_test, Y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])


def rgb2gray(X_train):
    R = X_train[:, 0]
    G = X_train[:, 1]
    B = X_train[:, 2]
    X_train_gray = 0.299 * R + 0.587 * G + 0.114 * B
    return X_train_gray


class AE:
    def __init__(self, data_name="MNIST", binary_mode=True):
        """
        data_name can be MNIST, etc. (default is MNIST)
        """
        self.data_name = data_name
        (X_train, y_train), (X_test, y_test) = self.load_data()
        # (X_train, y_train), (X_test, y_test) = mnist.load_data()
        # Modify input and output data to be appropritate for AE
        self.Org = (X_train, X_train), (X_test, X_test)
        self.Data = self.Org
        self.binary_mode = binary_mode

    def load_data(self):
        data_name = self.data_name
        if data_name == "MNIST":
            return mnist.load_data()
        elif data_name == "CIFAR10-GRAY":
            (X_train, y_train), (X_test, y_test) = cifar10.load_data()
            X_train_gray = rgb2gray(X_train)
            X_test_gray = rgb2gray(X_test)
            return (X_train_gray, y_train), (X_test_gray, y_test)
        else:
            raise ValueError("data_name of {} is not supported!".format(data_name))

    def modeling(self, input_img=Input(shape=(1, 28, 28))):
        # mode: binary or not (non-binary)
        binary_mode = self.binary_mode

        # set-1                                         1, ny, nx
        x = Conv2D(16, (3, 3), activation='relu',
                   padding='same')(input_img)        # 16, ny, nx
        x = MaxPooling2D((2, 2), padding='same')(x)  # 16, ny/2,nx/2
        x = Dropout(0.25)(x)  # Use dropout after maxpolling

        # set-2
        x = Conv2D(8, (3, 3), activation='relu',
                   padding='same')(x)  # 8,14,14
        x = MaxPooling2D((2, 2), padding='same')(x)  # 8, ny/4, nx/4
        x = Dropout(0.25)(x)  # Use dropout after maxpolling

        # set-3
        x = Conv2D(8, (3, 3), activation='relu',
                   padding='same')(x)                # 8, ny/4, nx/4
        encoded = x

        x = Conv2D(8, (3, 3), activation='relu',
                   padding='same')(encoded)          # 8, ny/4, nx/4
        # x = Dropout(0.25)(x) # Use dropout after maxpolling

        x = UpSampling2D((2, 2))(x)                  # 8, ny/2, nx/2
        x = Conv2D(8, (3, 3), activation='relu',     # 8, ny/2, nx/2
                   padding='same')(x)
        # x = Dropout(0.25)(x) # Use dropout after maxpolling

        x = UpSampling2D((2, 2))(x)                  # 8, ny, nx
        x = Conv2D(16, (3, 3), activation='relu',
                   padding='same')(x)                # 16, ny, nx
        # x = Dropout(0.25)(x) # Use dropout after maxpolling
        if binary_mode:
            decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # 1, ny, nx
        else:
            decoded = Conv2D(1, (3, 3), padding='same')(x)  # 1, ny, nx

        autoencoder = Model(input_img, decoded)

        self.autoencoder = autoencoder
        self.model_compile()

    def model_compile(self):
        binary_mode = self.binary_mode
        autoencoder = self.autoencoder
        if binary_mode:
            autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        else:
            autoencoder.compile(optimizer='adadelta', loss='mse')

    def run(self, epochs=10):
        (x_train_in, x_train), (x_test_in, x_test) = self.Data
        ny, nx = x_train_in.shape[1:]

        # Update 2 reshape from 3d to 4d array by including a channel element
        x_train_in, x_test_in = update2(x_train_in, x_test_in)
        x_train, x_test = update2(x_train, x_test)

        self.modeling(input_img=Input(shape=(1, ny, nx)))
        autoencoder = self.autoencoder
        history = autoencoder.fit(x_train_in, x_train,
                                  epochs=epochs,
                                  batch_size=128,
                                  shuffle=True,
                                  verbose=1,
                                  validation_data=(x_test_in, x_test))
        kkeras.plot_loss(history)

        self.imshow()

    def imshow(self):
        (_, _), (x_test_in, x_test) = self.Data
        x_test_in, x_test = update2(x_test_in, x_test)
        autoencoder = self.autoencoder
        decoded_imgs = autoencoder.predict(x_test_in)

        ny, nx = x_test.shape[2:]
        n = 10
        plt.figure(figsize=(20, 2 * 3 + 2))
        for i in range(n):
            # AE Input
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(x_test_in[i].reshape(ny, nx))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title('AE Input')

            # AE Traget
            ax = plt.subplot(3, n, i + n + 1)
            plt.imshow(x_test[i].reshape(ny, nx))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title('AE Target')

            # AE Output
            ax = plt.subplot(3, n, i + 2 * n + 1)
            plt.imshow(decoded_imgs[i].reshape(ny, nx))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.title('AE Output')
        plt.show()


class AE_HOLO(AE):
    def __init__(self, **kwargs):
        """
        Hologram transformation is performed
        **kwargs are used for AE
        """
        super().__init__(**kwargs)
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

        ny, nx = x_test_in.shape[2:]
        n = 10
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # display original
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(x_test[i].reshape(ny, nx))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(3, n, n + i + 1)
            plt.imshow(x_test_in[i].reshape(ny, nx))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(3, n, n * 2 + i + 1)
            plt.imshow(decoded_imgs[i].reshape(ny, nx))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


class AE_HOLO_EXT(AE_HOLO):
    def __init__(self, new_shape=None, **kwargs):
        """
        Hologram transformation is performed
        """
        self.new_shape = new_shape
        super().__init__(**kwargs)

    def load_data(self):
        new_shape = self.new_shape

        if self.new_shape is None:
            return super().load_data()
        else:
            (X_train, y_train), (X_test, y_test) = super().load_data()
            X_train_new = padding_zeros(X_train, new_shape=new_shape)
            X_test_new = padding_zeros(X_test, new_shape=new_shape)
            return (X_train_new, y_train), (X_test_new, y_test)


class AE_HOLO_MODEL(AE_HOLO_EXT):
    def __init__(self, modeling_type="large_1st", **kwargs):
        """
        modeling_type can be None or 'large_1st'
        """
        self.modeling_type = modeling_type
        super().__init__(**kwargs)

    def modeling(self, input_img=Input(shape=(1, 28, 28))):
        if self.modeling_type is None:
            return super().modeling(input_img=input_img)
        elif self.modeling_type == 'large_1st':
            return self.modeling_large_1st(input_img=input_img)
        else:
            raise ValueError("Specify new modeling_type!")

    def modeling_large_1st(self, input_img=Input(shape=(1, 28, 28))):
        # mode: binary or not (non-binary)
        binary_mode = self.binary_mode

        # Additional large filters at the 1st layer
        ny, nx = int(input_img.shape[2]), int(input_img.shape[3])
        x = Conv2D(1, (ny, nx), activation='linear',
                   padding='same')(input_img)        # 16, ny, nx
        frontstage_out = x

        # set-1                                         1, ny, nx
        x = Conv2D(16, (3, 3), activation='relu',
                   padding='same')(x)        # 16, ny, nx
        x = MaxPooling2D((2, 2), padding='same')(x)  # 16, ny/2,nx/2
        x = Dropout(0.25)(x)  # Use dropout after maxpolling

        # set-2
        x = Conv2D(8, (3, 3), activation='relu',
                   padding='same')(x)  # 8,14,14
        x = MaxPooling2D((2, 2), padding='same')(x)  # 8, ny/4, nx/4
        x = Dropout(0.25)(x)  # Use dropout after maxpolling

        # set-3
        x = Conv2D(8, (3, 3), activation='relu',
                   padding='same')(x)                # 8, ny/4, nx/4
        encoded = x

        x = Conv2D(8, (3, 3), activation='relu',
                   padding='same')(encoded)          # 8, ny/4, nx/4
        # x = Dropout(0.25)(x) # Use dropout after maxpolling

        x = UpSampling2D((2, 2))(x)                  # 8, ny/2, nx/2
        x = Conv2D(8, (3, 3), activation='relu',     # 8, ny/2, nx/2
                   padding='same')(x)
        # x = Dropout(0.25)(x) # Use dropout after maxpolling

        x = UpSampling2D((2, 2))(x)                  # 8, ny, nx
        x = Conv2D(16, (3, 3), activation='relu',
                   padding='same')(x)                # 16, ny, nx
        # x = Dropout(0.25)(x) # Use dropout after maxpolling
        if binary_mode:
            decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # 1, ny, nx
        else:
            decoded = Conv2D(1, (3, 3), padding='same')(x)  # 1, ny, nx

        autoencoder = Model(input_img, decoded)

        self.autoencoder = autoencoder
        self.model_compile()

        # Make additional model
        self.frontstage = Model(input_img, frontstage_out)

    def imshow(self):
        (_, _), (x_test_in, x_test) = self.Data
        x_test_in, x_test = update2(x_test_in, x_test)
        autoencoder = self.autoencoder
        frontstage = self.frontstage
        frontstage_out = frontstage.predict(x_test_in)
        decoded_imgs = autoencoder.predict(x_test_in)

        ny, nx = x_test_in.shape[2:]
        n = 10
        plt.figure(figsize=(20, 5))
        for i in range(n):
            # display original
            ax = plt.subplot(4, n, i + 1)
            plt.imshow(x_test[i].reshape(ny, nx))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(4, n, n + i + 1)
            plt.imshow(x_test_in[i].reshape(ny, nx))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(4, n, n * 2 + i + 1)
            plt.imshow(frontstage_out[i].reshape(ny, nx))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(4, n, n * 3 + i + 1)
            plt.imshow(decoded_imgs[i].reshape(ny, nx))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


def reduce_resol(X_train, rate=2):
    X_train = X_train[:, ::rate, ::rate]
    X_train_zoom = np.zeros((X_train.shape[0], X_train.shape[1] * rate, X_train.shape[2] * rate), dtype=X_train.dtype)
    for i in range(X_train.shape[0]):
        X_train_zoom[i] = ndimage.interpolation.zoom(X_train[i], rate)
    return X_train_zoom


class CNN2(CNN):
        def __init__(self):
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            X_train, X_test = reduce_resol(X_train), reduce_resol(X_test)
            self.Org = (X_train, y_train), (X_test, y_test)
            self.Data = self.Org

        def __init___r0(self):
            """
            Load data but reduce the resolution (1/2, 1/2) for x and y direction
            After that, zoom is applied to expand larger size images.
            Then, the further processes are no needed to be updated.
            """
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            X_train, X_test = X_train[:, ::2, ::2], X_test[:, ::2, ::2]
            X_train_zoom = np.zeros((X_train.shape[0], X_train.shape[1] * 2, X_train.shape[2] * 2), dtype=X_train.dtype)
            X_test_zoom = np.zeros((X_test.shape[0], X_test.shape[1] * 2, X_test.shape[2] * 2), dtype=X_test.dtype)
            for i in range(X_train.shape[0]):
                X_train_zoom[i] = ndimage.interpolation.zoom(X_train[i], 2)
            for i in range(X_test.shape[0]):
                X_test_zoom[i] = ndimage.interpolation.zoom(X_test[i], 2)
            self.Org = (X_train_zoom, y_train), (X_test_zoom, y_test)
            self.Data = self.Org


class CNN2_HOLO(CNN_HOLO):
    def __init__(self):
        """
        After hologram, image size whould be reduced and 
        expanded using linear interpolation.
        - General hologram will be updated first. 
        - Same routine will be applied used at CNN2
        - complex hologram will be updated.
        """
        super().__init__()

    def holo_transform(self):
        (X_train, y_train), (X_test, y_test) = holo_transform(self.Org)
        X_train, X_test = reduce_resol(X_train), reduce_resol(X_test)
        self.Data = (X_train, y_train), (X_test, y_test)
        self.Holo = self.Data

    def holo_complex_transform(self):
        # Transform X_train and X_test using hologram filtering
        (X_train_holo, y_train), (X_test_holo, y_test) = holo_complex_transform(self.Org)
        X_train_holo[:,0,:,:] = reduce_resol(X_train_holo[:,0,:,:])
        X_train_holo[:,1,:,:] = reduce_resol(X_train_holo[:,1,:,:])
        X_test_holo[:,0,:,:] = reduce_resol(X_test_holo[:,0,:,:])
        X_test_holo[:,1,:,:] = reduce_resol(X_test_holo[:,1,:,:])

        self.Data = (X_train_holo, y_train), (X_test_holo, y_test)
        self.Holo_complex = self.Data
        self.complex_flag = True


def padding_zeros(X_train, new_shape=(28 * 2, 28 * 2)):
    mx, my = X_train.shape[1:]
    # mx and my should be even numbers
    assert (mx // 2 * 2 == mx and my // 2 * 2 == my)
    mx2, my2 = mx // 2, my // 2

    white_board = np.zeros(new_shape, dtype=X_train.dtype)
    XL, YL = white_board.shape
    # XL and YL should be larger than mx and my, respectively
    assert (XL >= mx and YL >= my)
    # XL and YL should be even numbers
    assert (XL // 2 * 2 == XL and YL // 2 * 2 == YL)
    XL2, YL2 = XL // 2, YL // 2

    board_l = []
    for X_tr in X_train:
        board = white_board.copy()
        board[XL2 - mx2:XL2 + mx2, YL2 - my2:YL2 + my2] += X_tr
        board_l.append(board)

    return np.array(board_l)
