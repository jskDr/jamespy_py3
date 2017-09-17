# from sklearn import model_selection, metrics
# from sklearn.preprocessing import MinMaxScaler
import numpy as np
# import matplotlib.pyplot as plt
from keras import backend as K
# from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from keras.applications.vgg16 import VGG16  # preprocess_input
from keras.applications.resnet50 import ResNet50

from medic import dl
# import kkeras


class CNN():
    def __init__(self, input_shape, nb_classes, weights='imagenet'):
        base_model = ResNet50(weights=weights, include_top=False,
                              input_shape=input_shape)

        x = base_model.input
        h = base_model.output
        z_cl = h  # Saving for cl output monitoring.

        h = GlobalAveragePooling2D()(h)
        h = Dense(128, activation='relu')(h)
        h = Dropout(0.5)(h)
        z_fl = h  # Saving for fl output monitoring.

        y = Dense(nb_classes, activation='softmax', name='preds')(h)
        # y = Dense(4, activation='softmax')(h)

        for layer in base_model.layers:
            layer.trainable = False

        model = Model(x, y)
        model.compile(loss='categorical_crossentropy', 
                      optimizer='adadelta', metrics=['accuracy'])

        self.model = model
        self.cl_part = Model(x, z_cl)
        self.fl_part = Model(x, z_fl)

    def __call__(self):
        return self.model


class DataSet(dl.DataSet):
    def __init__(self, X, y, nb_classes, n_channels=3):
        self.n_channels = n_channels
        super().__init__(X, y, nb_classes)

    def add_channels(self):
        n_channels = self.n_channels

        if n_channels == 1:
            super().add_channels()
        else:
            X = self.X
            if X.ndim < 4:  # if X.dim == 4, no need to add a channel rank.
                N, img_rows, img_cols = X.shape
                if K.image_dim_ordering() == 'th':
                    X = X.reshape(X.shape[0], 1, img_rows, img_cols)
                    X = np.concatenate([X, X, X], axis=1)
                    input_shape = (n_channels, img_rows, img_cols)
                else:
                    X = X.reshape(X.shape[0], img_rows, img_cols, 1)
                    X = np.concatenate([X, X, X], axis=3)
                    input_shape = (img_rows, img_cols, n_channels)
            else:
                if K.image_dim_ordering() == 'th':
                    N, Ch, img_rows, img_cols = X.shape
                    if Ch == 1:
                        X = np.concatenate([X, X, X], axis=1)
                    input_shape = (n_channels, img_rows, img_cols)
                else:
                    N, img_rows, img_cols, Ch = X.shape
                    if Ch == 1:
                        X = np.concatenate([X, X, X], axis=3)
                    input_shape = (img_rows, img_cols, n_channels)

            self.X = X
            self.input_shape = input_shape
            # self.img_info = {'channels': n_channels,
            #                 'rows': img_rows, 'cols': img_cols}


class Machine(dl.Machine):
    def __init__(self, X, y, nb_classes=2, weights='imagenet'):
        data = DataSet(X, y, nb_classes, n_channels=3)
        # model = CNN(data.input_shape, nb_classes)
        model = CNN(data.input_shape, nb_classes, weights=weights)()

        self.data = data
        self.model = model
