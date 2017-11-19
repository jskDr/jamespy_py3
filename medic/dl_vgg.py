import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from keras.applications.imagenet_utils import preprocess_input
from keras.applications import VGG16
from keras.applications import VGG19

from medic import dl
import kgrid


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
                 n_dense=128, p_dropout=0.5, BN_flag=False,
                 PretrainedModel=VGG16):
        """
        If BN_flag is True, BN is used instaed of Dropout
        """
        model.in_shape = input_shape
        model.n_dense = n_dense
        model.p_dropout = p_dropout
        model.PretrainedModel = PretrainedModel
        model.BN_flag = BN_flag
        super().__init__(nb_classes)

    def build_model(model):
        nb_classes = model.nb_classes
        input_shape = model.in_shape
        PretrainedModel = model.PretrainedModel
        # print(nb_classes)

        # base_model = VGG16(weights='imagenet', include_top=False)

        base_model = PretrainedModel(
            weights='imagenet',
            include_top=False, input_shape=input_shape)

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
        if BN_Flag is True, BN is used instead of Dropout
        '''
        BN_flag = model.BN_flag

        n_dense = model.n_dense
        p_dropout = model.p_dropout

        h = GlobalAveragePooling2D()(h)
        h = Dense(n_dense, activation='relu')(h)
        if BN_flag:
            h = BatchNormalization()(h)
        else:
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
                 test_size=0.2, random_state=0,
                 preprocessing_flag=False):
        self.n_channels = n_channels
        self.preprocessing_flag = preprocessing_flag
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

            if self.preprocessing_flag:
                X = preprocess_input(X)
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
                 n_dense=128, p_dropout=0.5, BN_flag=False,
                 PretrainedModel=VGG16, 
                 preprocessing_flag=True,
                 fig=True):
        """
        scaling becomes False for DataSet
        """

        data = DataSet(X, y, nb_classes, n_channels=3, scaling=False, 
                       preprocessing_flag=preprocessing_flag)
        # model = CNN(data.input_shape, nb_classes)

        self.data = data
        self.nb_classes = nb_classes
        self.n_dense = n_dense
        self.p_dropout = p_dropout
        self.BN_flag = BN_flag
        self.PretrainedModel = PretrainedModel
        self.set_model()

        self.fig = fig

    def set_model(self):
        data = self.data
        nb_classes = self.nb_classes
        n_dense = self.n_dense
        p_dropout = self.p_dropout
        BN_flag = self.BN_flag
        PretrainedModel = self.PretrainedModel

        self.model = CNN(data.input_shape, nb_classes,
                         n_dense=n_dense, p_dropout=p_dropout, BN_flag=BN_flag,
                         PretrainedModel=PretrainedModel)

    def get_features(self):
        data = self.data
        PretrainedModel = self.PretrainedModel
        features = get_features_pretrained(data.X, PretrainedModel=PretrainedModel)
        return features

    def gs_SVC(self):
        y = self.data.y
        features = self.get_features()
        features1d = features.reshape(features.shape[0], -1)
        gs = kgrid.gs_SVC(features1d, y, params={'C':(1, 10, 100), 'gamma': (1e-1, 1e-2, 1e-3)})
        print("Best score (r2):", gs.best_score_)
        print("Best param:", gs.best_params_)
        return gs


class Machine_Generator(dl.Machine_Generator):
    def __init__(self, X, y, nb_classes=2, steps_per_epoch=10,
                 n_dense=128, p_dropout=0.5, BN_flag=False,
                 scaling=False,
                 PretrainedModel=VGG16, fig=True,
                 gen_param_dict=None):
        """
        scaling becomes False for DataSet
        """

        data = DataSet(X, y, nb_classes, n_channels=3, scaling=scaling)
        # model = CNN(data.input_shape, nb_classes)

        self.data = data
        self.nb_classes = nb_classes
        self.n_dense = n_dense
        self.p_dropout = p_dropout
        self.BN_flag = BN_flag
        self.PretrainedModel = PretrainedModel
        self.set_model()

        self.set_generator(steps_per_epoch, gen_param_dict=gen_param_dict)
        self.fig = fig

    def set_model(self):
        data = self.data
        nb_classes = self.nb_classes
        n_dense = self.n_dense
        p_dropout = self.p_dropout
        BN_flag = self.BN_flag
        PretrainedModel = self.PretrainedModel

        self.model = CNN(data.input_shape, nb_classes,
                         n_dense=n_dense, p_dropout=p_dropout, BN_flag=BN_flag,
                         PretrainedModel=PretrainedModel)


def get_features_pretrained(X, PretrainedModel=VGG19, preprocess_input=preprocess_input):
    """
    get features by pre-trained networks
    :param Pretrained: VGG19 is default
    :return: features
    """
    if preprocess_input is not None:
        X = preprocess_input(X)
    model = PretrainedModel(weights='imagenet', include_top=False, input_shape=X.shape[1:])
    features = model.predict(X)
    return features
