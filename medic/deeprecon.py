"""
DeepRecon - Reconstruction using Deep Learning
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection

# import numpy
# from keras.datasets import mnist
from keras.models import Sequential
# from keras.layers import Dense
from keras.layers import Dropout
# from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
# from keras.layers.convolutional import MaxPooling2D
# from keras.utils import np_utils
from keras import backend as K
import k.keras

K.set_image_dim_ordering('th')


def baseline_model(input_shape=(1, 28, 28)):
    # create model
    model = Sequential()
    model.add(Convolution2D(16, 40, 40, border_mode='same',
                            input_shape=input_shape, activation='relu'))
    # model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.2))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.add(Convolution2D(1, 5, 5, border_mode='same',
                            input_shape=input_shape, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


def baseline_model_168080(input_shape=(1, 28, 28)):
    # create model
    model = Sequential()
    model.add(Convolution2D(16, 80, 80, border_mode='same',
                            input_shape=input_shape, activation='relu'))
    #model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.2))
    # model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    #model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.add(Convolution2D(1, 5, 5, border_mode='same',
                            input_shape=input_shape, activation='sigmoid'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


class DeepRecon():

    def __init__(self, disp=False):
        """
        Parameters
        ==========
        cell_df_ext, pd.DataFrame
        E.g.,
        fanme = '../0000_Base/sheet.gz/cell_db100_no_extra_beads_fd.cvs.gz'
        cell_df_ext = pd.read_csv(fname)
        """
        self.disp = disp

    def load_data(self, cell_df_ext):
        disp = self.disp

        Lx = cell_df_ext['x'].max() + 1
        Ly = cell_df_ext['y'].max() + 1
        Limg = cell_df_ext['ID'].max() + 1

        Img = cell_df_ext["image"].values.reshape(Limg, Lx, Lx)
        if cell_df_ext['freznel image'].values.dtype == 'complex128':
            FImg = cell_df_ext['freznel image'].values.reshape(Limg, Lx, Lx)
        else:
            FImg = cell_df_ext['freznel image'].apply(
                lambda val: complex(val.strip('()'))).values.reshape(Limg, Lx, Lx)

        if disp:
            print(Lx, Ly, Limg)
            print(Img.dtype, FImg.dtype)

        # Set seed for randomization as a constant
        seed = 7
        np.random.seed(seed)

        # Store to self variables
        self.cell_df_ext = cell_df_ext
        self.Lx, self.Ly, self.Limg = Lx, Ly, Limg
        self.Img, self.FImg = Img, FImg

    def imshow(self):
        """
        Plot Img and FImg for its real, image, and magnitue
        """
        Img, FImg = self.Img, self.FImg

        plt.figure(figsize=(10, 8))
        for i in range(5):
            plt.subplot(4, 5, i + 1)
            plt.imshow(Img[i, :, :], cmap='gray')
            plt.xticks([])
            plt.yticks([])

            plt.subplot(4, 5, i + 1 + 5)
            # Intensity is power (not real or image)
            plt.imshow(np.real(FImg[i, :, :]))
            plt.axis('off')

            plt.subplot(4, 5, i + 1 + 5 * 2)
            # Intensity is power (not real or image)
            plt.imshow(np.imag(FImg[i, :, :]))
            plt.axis('off')

            plt.subplot(4, 5, i + 1 + 5 * 3)
            # Intensity is power (not real or image)
            plt.imshow(np.abs(FImg[i, :, :]))
            plt.axis('off')
        plt.show()

    def setXY(self, disp=False):
        """
        Define X, Y for deep learning
        """
        cell_df_ext = self.cell_df_ext
        Img, FImg = self.Img, self.FImg

        # print(FImg.shape)
        cell_y = cell_df_ext[(cell_df_ext["x"] == 0) & (
            cell_df_ext["y"] == 0)]["celltype"].values
        # cell_y.shape

        # Using real and image as different plan like different color plain

        FImg_4D = FImg.reshape(FImg.shape[0], 1, FImg.shape[1], FImg.shape[2])
        # FImg_4D.shape

        X = np.concatenate([np.real(FImg_4D), np.imag(FImg_4D)], axis=1)
        Y = Img.reshape(Img.shape[0], 1, Img.shape[1], Img.shape[2])
        if(disp):
            print(X.shape, Y.shape)

        self.X, self.Y = X, Y

    def split(self):
        X, Y = self.X, self.Y

        X_scale = X / np.std(X.reshape(-1))
        Y_scale = Y / np.std(Y.reshape(-1))
        # load data
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
            X_scale, Y_scale, test_size=0.33, random_state=42)

        self.X_train, self.X_test, self.Y_train, self.Y_test = X_train, X_test, Y_train, Y_test

    def modeling(self, gen_model=baseline_model):
        Lx, Ly = self.Lx, self.Ly
        # build the model
        #model = gen_model((2, Lx, Ly))
        model = baseline_model((2, Lx, Ly))

        self.model = model

    def newmodeling(self, gen_model=baseline_model):
        Lx, Ly = self.Lx, self.Ly
        # build the model
        model = gen_model((2, Lx, Ly))
        # model = baseline_model((2, Lx, Ly))

        self.model = model

    def fit(self, nb_epoch=20, batch_size=200, verbose=1):
        model = self.model
        X_train, X_test, Y_train, Y_test = self.X_train, self.X_test, self.Y_train, self.Y_test
        # Fit the model
        history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                            nb_epoch=nb_epoch, batch_size=batch_size, verbose=verbose)
        # Final evaluation of the model
        scores = model.evaluate(X_test, Y_test, verbose=0)
        print("Baseline Error: %.2f%%" % (100 - scores[1] * 100))

        k.keras.plot_history(history)

    def plot_test(self, n_samples=10):
        model = self.model
        X_test = self.X_test
        Y_test = self.Y_test

        Y_test_pred = model.predict(X_test)
        for i in range(n_samples):
            plt.figure(figsize=(12, 2))
            plt.subplot(1, 6, 1)
            plt.imshow(Y_test_pred[i, 0, :, :])
            if i == 0:
                plt.title('Recon')

            plt.axis('off')
            plt.subplot(1, 6, 2)
            plt.imshow(Y_test[i, 0, :, :])
            if i == 0:
                plt.title('Original')
            plt.axis('off')

            plt.subplot(1, 6, 3)
            plt.imshow(X_test[i, 0, :, :])
            if i == 0:
                plt.title('Diff-Real')
            plt.axis('off')
            plt.subplot(1, 6, 4)
            plt.imshow(X_test[i, 1, :, :])
            if i == 0:
                plt.title('Diff-Image')
            plt.axis('off')

            plt.subplot(1, 6, 5)
            plt.imshow(np.abs(X_test[i, 0, :, :]))
            if i == 0:
                plt.title('Diff-Mag')
            plt.axis('off')
            plt.subplot(1, 6, 6)
            plt.imshow(np.angle(X_test[i, 1, :, :]))
            if i == 0:
                plt.title('Diff-Phase')
            plt.axis('off')

            plt.show()

    def run(self, cell_df_ext, nb_epoch=20):
        """
        This self function includes a general example how to use this class for data recon.
        """
        Data = self

        # fname = '../0000_Base/sheet.gz/cell_db100_no_extra_beads_fd.cvs.gz'
        # cell_df_ext = pd.read_csv(fname)
        # Data = deeprecon.DeepRecon(cell_df_ext, True)
        # Data.imshow()
        print('load_data()')
        Data.load_data(cell_df_ext=cell_df_ext)
        print("selfXY()")
        Data.setXY()
        print("split()")
        Data.split()
        print("modeling()")
        Data.modeling()
        print("fit()")
        Data.fit(nb_epoch=nb_epoch)
        print("plot_test()")
        Data.plot_test()

        return self

    def run_fname(self, fname=None, nb_epoch=20):
        print('Load csv file using pd.read_csv()')
        fname = '../0000_Base/sheet.gz/cell_db100_no_extra_beads_fd.cvs.gz'
        cell_df_ext = pd.read_csv(fname)
        self.run(cell_df_ext=cell_df_ext, nb_epoch=nb_epoch)
        return self
