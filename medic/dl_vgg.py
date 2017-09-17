# from sklearn import model_selection, metrics
# from sklearn.preprocessing import MinMaxScaler
import numpy as np
# import matplotlib.pyplot as plt
from keras import backend as K
# from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from keras.applications.vgg16 import VGG16  # preprocess_input
from keras.applications.resnet50 import ResNet50

from medic import dl
# import kkeras


class CNN_nodropout(dl.CNN):
    def __init__(model, input_shape, nb_classes):
        model.in_shape = input_shape
        super().__init__(nb_classes)

    def build_model(model):
        nb_classes = model.nb_classes
        input_shape = model.in_shape
        # print(nb_classes)

        # base_model = VGG16(weights='imagenet', include_top=False)

        base_model = VGG16(weights='imagenet', include_top=False,
                           input_shape=input_shape)

        x = base_model.input
        h = base_model.output
        z_cl = h  # Saving for cl output monitoring.

        h = GlobalAveragePooling2D()(h)
        h = Dense(10, activation='relu')(h)
        z_fl = h  # Saving for fl output monitoring.

        y = Dense(nb_classes, activation='softmax', name='preds')(h)
        # y = Dense(4, activation='softmax')(h)

        for layer in base_model.layers:
            layer.trainable = False

        model.cl_part = Model(x, z_cl)
        model.fl_part = Model(x, z_fl)

        model.x = x
        model.y = y


class CNN(dl.CNN):
    def __init__(model, input_shape, nb_classes,
                 n_dense=128, p_dropout=0.5,
                 PretrainedModel=VGG16):
        model.in_shape = input_shape
        model.n_dense = n_dense
        model.p_dropout = p_dropout
        model.PretrainedModel = PretrainedModel
        super().__init__(nb_classes)

    def build_model(model):
        nb_classes = model.nb_classes
        input_shape = model.in_shape
        PretrainedModel = model.PretrainedModel
        # print(nb_classes)

        # base_model = VGG16(weights='imagenet', include_top=False)

        base_model = PretrainedModel(include_top=False, input_shape=input_shape)

        x = base_model.input
        h = base_model.output
        z_cl = h  # Saving for cl output monitoring.

        h = model.topmodel(h)

        z_fl = h  # Saving for fl output monitoring.

        y = Dense(nb_classes, activation='softmax', name='preds')(h)
        # y = Dense(4, activation='softmax')(h)

        for layer in base_model.layers:
            layer.trainable = False

        model.cl_part = Model(x, z_cl)
        model.fl_part = Model(x, z_fl)

        model.x = x
        model.y = y

    def topmodel(model, h):
        '''
        Define topmodel
        '''
        n_dense = model.n_dense
        p_dropout = model.p_dropout

        h = GlobalAveragePooling2D()(h)
        h = Dense(n_dense, activation='relu')(h)
        h = Dropout(p_dropout)(h)
        return h


class CNN_BN(CNN):
    def __init__(model, input_shape, nb_classes,
                 n_dense=128, p_dropout=0.5):
        super().__init__(input_shape, nb_classes,
                         n_dense=n_dense, p_dropout=p_dropout)

    def topmodel(model, h):
        """
        n_dense and p_dropout are used
        """
        n_dense = model.n_dense
        p_dropout = model.p_dropout

        h = GlobalAveragePooling2D()(h)
        h = Dense(n_dense)(h)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Dropout(p_dropout)(h)
        return h


class DataSet(dl.DataSet):
    def __init__(self, X, y, nb_classes, n_channels=3, scaling=True,
                 test_size=0.2, random_state=0):
        self.n_channels = n_channels
        super().__init__(X, y, nb_classes, scaling=scaling,
                         test_size=test_size, random_state=random_state)

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


class Machine_nodropout(dl.Machine):
    def __init__(self, X, y, nb_classes=2):
        data = DataSet(X, y, nb_classes, n_channels=3)
        model = CNN_nodropout(data.input_shape, nb_classes)
        # model = CNN_dropout(data.input_shape, nb_classes)

        self.data = data
        self.model = model


class Machine(dl.Machine):
    def __init__(self, X, y, nb_classes=2,
                 n_dense=128, p_dropout=0.5,
                 PretrainedModel=VGG16):
        """
        scaling becomes False for DataSet
        """

        data = DataSet(X, y, nb_classes, n_channels=3, scaling=False)
        # model = CNN(data.input_shape, nb_classes)
        model = CNN(data.input_shape, nb_classes,
                    n_dense=n_dense, p_dropout=p_dropout,
                    PretrainedModel=VGG16)

        self.data = data
        self.model = model
        
        
class Machine_Generator(dl.Machine_Generator):
    def __init__(self, X, y, nb_classes=2, steps_per_epoch=10, 
                 n_dense=128, p_dropout=0.5, scaling=False,
                 PretrainedModel=VGG16, fig=True):
        """
        scaling becomes False for DataSet
        """

        data = DataSet(X, y, nb_classes, n_channels=3, scaling=scaling)
        # model = CNN(data.input_shape, nb_classes)
        model = CNN(data.input_shape, nb_classes,
                    n_dense=n_dense, p_dropout=p_dropout,
                    PretrainedModel=VGG16)

        self.data = data
        self.model = model
        
        self.set_generator(steps_per_epoch)
        
        self.fig = fig
