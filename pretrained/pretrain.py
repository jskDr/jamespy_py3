from importlib import reload
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input
from keras.utils import np_utils
from keras import datasets

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input

from actmap import img
from actmap.proc import CAM_process


def get_new_base_model_x(vgg_model, x):
    h = x
    for layer in vgg_model.layers[1:18]:
        h = layer(h)
    m = Model(x, h)
    return m


def get_new_base_model_x_h(vgg_model, x, h):
    for layer in vgg_model.layers[1:18]:
        h = layer(h)
    m = Model(x, h)
    return m


def run(epochs, PP=None, rgb_mean=None):
    # Load Data
    ig = image.ImageDataGenerator()  # rescale=1/255.0)
    it = ig.flow_from_directory(
        'tiny-imagenet-200/train/', target_size=(64, 64), batch_size=1000)
    ystr2y = it.class_indices
    Xt, Yt, yt = img.load_validation_data(
        it.class_indices, 'tiny-imagenet-200/val/images/', 'tiny-imagenet-200/val/val_annotations.txt')

    Num_classes = 200
    Input_shape = (64, 64, 3)

    # Build a Model
    vgg_model = VGG16(include_top=False)
    x = Input(shape=Input_shape)
    if PP:
        h = PP(8, 0.5, rgb_mean)(x)
        base_model = get_new_base_model_x_h(vgg_model, x, h)
    else:
        base_model = get_new_base_model_x(vgg_model, x)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(Num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    # Training
    model.fit_generator(it, 1000, epochs=epochs, validation_data=(Xt, Yt))

    # Class Activiation Map
    conv_model = base_model
    Ft = conv_model.predict(Xt[:20])

    Wb_all = model.get_weights()
    L_Wb = len(Wb_all)

    W_dense = Wb_all[L_Wb - 2]
    b_dense = Wb_all[L_Wb - 1]
    W_dense.shape, b_dense.shape

    CAM = CAM_process(Ft, yt, W_dense)

    from kakao import bbox
    for i in range(20):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(Xt[i])
        plt.subplot(1, 3, 2)
        CAM4 = img.cam_intp_reshape(CAM[i], Xt[i].shape[:2])
        plt.imshow(CAM4, interpolation='bicubic', cmap='jet_r')
        plt.subplot(1, 3, 3)
        plt.imshow(Xt[i])
        plt.imshow(CAM4, interpolation='bicubic', cmap='jet_r', alpha=0.5)
        plt.show()
