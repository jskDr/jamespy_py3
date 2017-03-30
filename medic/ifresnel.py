"""
Inverse Fresnel 
Sungjin (James) Kim
Dec 17, 2016
"""

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1337)  # for reproducibility
from sklearn import model_selection
import pandas as pd

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.utils import np_utils
from keras import backend as K
from keras import callbacks

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model

import kkeras

class CDNN():
    def __init__(self, Lx, Ly, X, y, modeling_id = None):
        batch_size = 128
        nb_classes = 2
        nb_epoch = 1000

        # input image dimensions
        img_rows, img_cols = Lx, Ly

        # the data, shuffled and split between train and test sets
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size=0.2, random_state=0)

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

        if modeling_id is None:
            modeling = self.modeling
        else:
            modeling = getattr(self, "modeling_{}".format(modeling_id))
        model = modeling(input_shape, nb_classes)

        model.compile(loss='categorical_crossentropy',
                    optimizer='adadelta',
                    metrics=['accuracy'])

        # Early stoping is not used in this training 
        # earlyStopping=callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

        self.model = model
        self.X_train, self.Y_train = X_train, Y_train
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.X_test, self.Y_test = X_test, Y_test

    def modeling(self, input_shape, nb_classes):
        nb_filters = 8
        # size of pooling area for max pooling
        pool_size_l = [(4, 4), (4,4)] # 160 --> 40, 40 --> 10
        # convolution kernel size
        kernel_size = (20, 20)

        model = Sequential()

        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid',
                                input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size_l[0])) # 160 --> 40
        model.add(Dropout(0.25))

        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size_l[1])) # 40 --> 10
        model.add(Dropout(0.25))

        model.add(UpSampling2D(pool_size_l[1])) # 10 --> 40
        #model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
        #                        border_mode='valid'))
        #model.add(Activation('relu'))
        #model.add(UpSampling2D(pool_size_2))
        #model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(4))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        return model

    def fit(self):
        # Only required self parameters are used by localization
        model = self.model
        X_train, Y_train = self.X_train, self.Y_train
        batch_size = self.batch_size
        nb_epoch = self.nb_epoch
        X_test, Y_test = self.X_test, self.Y_test

        history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                verbose=0, validation_data=(X_test, Y_test))  # callbacks=[earlyStopping])

        plt.subplot(1,2,1)
        kkeras.plot_acc( history)
        plt.subplot(1,2,2)
        kkeras.plot_loss( history) 

    def eval(self):
        model = self.model
        X_test, Y_test = self.X_test, self.Y_test

        score = model.evaluate(X_test, Y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

    def fit_eval(self):
        self.fit()
        self.eval()

    def modeling_1(self, input_shape, nb_classes):
        print("Modeling_1")
        nb_filters = 8
        # size of pooling area for max pooling
        pool_size_l = [(4, 4), (4,4)] # 160 --> 40, 40 --> 10
        # convolution kernel size
        kernel_size = (20, 20)

        model = Sequential()

        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid',
                                input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size_l[0])) # 160 --> 40
        model.add(Dropout(0.25))

        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size_l[1])) # 40 --> 10
        model.add(Dropout(0.25))

        model.add(Convolution2D(nb_filters, 2, 2,
                                border_mode='valid'))
        model.add(Activation('relu'))
        model.add(UpSampling2D(pool_size_l[1])) # 10 --> 40
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(4))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        return model

    def modeling_2(self, input_shape, nb_classes):
        print("Modeling_2")
        nb_filters = 8
        # size of pooling area for max pooling
        pool_size_l = [(4, 4), (4,4)] # 160 --> 40, 40 --> 10
        # convolution kernel size
        kernel_size = (20, 20)

        model = Sequential()

        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid',
                                input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size_l[0])) # 160 --> 40
        model.add(Dropout(0.25))

        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                                border_mode='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size_l[1])) # 40 --> 10
        model.add(Dropout(0.25))

        model.add(Convolution2D(nb_filters, 2, 2,
                                border_mode='valid'))
        model.add(Activation('relu'))
        model.add(UpSampling2D(pool_size_l[1])) # 10 --> 40
        model.add(Dropout(0.25))

        model.add(Convolution2D(nb_filters, 2, 2,
                                border_mode='valid'))
        model.add(Activation('relu'))
        model.add(UpSampling2D(pool_size_l[0])) # 40 --> 160
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        return model

class AE():
    def __init__(self, csvname='sheet.gz/cell_fd_db100.cvs.gz'):
        cell_df_ext = pd.read_csv(csvname)
        # cell_df_ext.head()

        Lx = cell_df_ext['x'].max() + 1
        Ly = cell_df_ext['y'].max() + 1
        Limg = cell_df_ext['ID'].max() + 1
        print( Lx, Ly, Limg)

        Img = cell_df_ext["image"].values.reshape(Limg,Lx,Lx)
        FImg = cell_df_ext['freznel image'].apply(lambda val: complex(val.strip('()'))).values.reshape(Limg,Lx,Lx)
        # MFImg = cell_df_ext["mag freznel image"].values.reshape(Limg,Lx,Lx)
        cell_y = cell_df_ext[(cell_df_ext["x"]==0) & (cell_df_ext["y"]==0)]["celltype"].values
        print( Img.dtype, FImg.dtype, cell_y.shape)

        plt.figure(figsize=(10, 4))
        for i in range(5):
            plt.subplot(2,5,i+1)
            plt.imshow( Img[i,:,:], cmap='gray_r')
            plt.xticks([])
            plt.yticks([])
            
            plt.subplot(2,5,i+1+5)
            plt.imshow( np.power(np.abs(FImg[i,:,:]),2)) # Intensity is power (not real or image)
            plt.axis('off')
        plt.show()

        cell_img_a = Img 
        cell_img_f = np.abs(FImg)

        X = cell_img_f.reshape( cell_img_f.shape[0], -1)
        Y = cell_img_a.reshape( cell_img_a.shape[0], -1)

        plt.subplot(1,2,1)
        plt.imshow(cell_img_f[0,:,:], cmap='gray_r')
        plt.subplot(1,2,2)
        plt.imshow(cell_img_f[1,:,:], cmap='gray_r')

        print( Y.dtype, X.dtype)

        self.X, self.Y = X, Y
        self.Lx, self.Ly = Lx, Ly

    def run(self, nb_epoch=100):
        self.set_data()
        self.modeling()
        self.fit(nb_epoch=nb_epoch)
        self.plot()
    
    def set_data(self):
        X, Y = self.X, self.Y
        Lx, Ly = self.Lx, self.Ly

        x_train, x_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.33, random_state=0)

        x_train = np.reshape(x_train, (len(x_train), 1, Lx, Ly))
        x_test = np.reshape(x_test, (len(x_test), 1, Lx, Ly))

        y_train = y_train.astype('float32') / 255
        y_test = y_test.astype('float32') / 255
        y_train = np.reshape(y_train, (len(y_train), 1, Lx, Ly))
        y_test = np.reshape(y_test, (len(y_test), 1, Lx, Ly))

        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test

    def modeling(self):
        Lx, Ly = self.Lx, self.Ly
        input_img = Input(shape=(1, Lx, Ly))
        ks = 8

        x = Convolution2D(16, ks*2, ks*2, activation='relu', border_mode='same')(input_img)
        x = MaxPooling2D((2, 2), border_mode='same')(x)  # 160 --> 80
        x = Convolution2D(8, ks*2, ks*2, activation='relu', border_mode='same')(x)
        x = MaxPooling2D((2, 2), border_mode='same')(x)  # 80 --> 40
        x = Convolution2D(8, ks*2, ks*2, activation='relu', border_mode='same')(x)
        encoded = MaxPooling2D((2, 2), border_mode='same')(x) # 40 --> 20

        # at this point the representation is (8, 20, 20) 

        x = Convolution2D(8, ks, ks, activation='relu', border_mode='same')(encoded)
        x = UpSampling2D((2, 2))(x) # 20 --> 40
        x = Convolution2D(8, ks, ks, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x) # 20 --> 80
        x = Convolution2D(16, ks, ks, activation='relu', border_mode='same')(x)
        x = UpSampling2D((2, 2))(x) # 80 --> 160
        decoded = Convolution2D(1, ks, ks, activation='sigmoid', border_mode='same')(x)

        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        self.autoencoder = autoencoder 

    def fit(self, nb_epoch=10):
        x_train, x_test, y_train, y_test = self.x_train, self.x_test, self.y_train, self.y_test
        autoencoder = self.autoencoder

        history = autoencoder.fit(x_train, y_train,
                nb_epoch=nb_epoch,
                batch_size=128,
                shuffle=True,
                verbose=0,
                validation_data=(x_test, y_test))

        kkeras.plot_loss(history)

    def plot(self):
        x_test, y_test = self.x_test, self.y_test
        autoencoder = self.autoencoder
        Lx, Ly = self.Lx, self.Ly

        decoded_imgs = autoencoder.predict(x_test)
        n = 5
        plt.figure(figsize=(10, 6))
        for i in range(n):
            # display original
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(x_test[i].reshape(Lx, Ly), cmap='rainbow')
            #plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(3, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(Lx, Ly), cmap='rainbow')
            #plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            # display original
            ax = plt.subplot(3, n, i + 1 + n*2)
            plt.imshow(y_test[i].reshape(Lx, Ly), cmap='rainbow')
            #plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()