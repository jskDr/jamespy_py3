"""
GANs for the super-resolution application.
  - Example data is MNIST.
  - Input is subsampled MNIST data
  - Target is recovered high-resolution MNIST data
"""


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling1D, Convolution1D, MaxPooling1D
from keras.layers.core import Activation, Flatten, Dropout
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
from PIL import Image
import argparse
import math
from sklearn import model_selection

import time
import os

INPUT_LN = 56 * 2
N_GEN_l = [4, 4]
CODE_LN = int(INPUT_LN / np.prod(N_GEN_l))

# kera (Underdeveloping)
class Seq(Sequential):
    def __init__(self):
        pass

    def act(self):
        self.add(Activiation())
        return self


def generator_model(): #INPUT_LN=INPUT_LN, CODE_LN=CODE_LN, N_GEN_l=N_GEN_l):
    print(INPUT_LN, N_GEN_l, CODE_LN) 

    model = Sequential()
    model.add(Dense(int(INPUT_LN), input_dim=CODE_LN, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(INPUT_LN, init='uniform'))   
    return model


def discriminator_model():
    model = Sequential()
    model.add(Dense(int(INPUT_LN/2), input_dim=INPUT_LN, init='uniform'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    model.add(Dense(int(INPUT_LN/4), init='uniform'))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = generated_images.shape[2:]
    image = np.zeros((height * shape[0], width * shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
            img[0, :, :]
    return image


def train_VI(BATCH_SIZE, V, I=None, no_epoch = 100, disp=False):
    """
    Developing
    ==========
    [2017-3-14]
    - To include saving the progress: tsplot of input and output
    - Input can be saved separately. 
    """
    if I is None:
        X_train = V
    else:
        X_train = np.concatenate([V, I], axis=1)

    # X_train = X_train.astype(np.float32)
    # X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    discriminator = discriminator_model()
    generator = generator_model()
    discriminator_on_generator = \
        generator_containing_discriminator(generator, discriminator)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(
        loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((BATCH_SIZE, CODE_LN))

    for epoch in range(no_epoch):
        if disp:
            print("Epoch is", epoch)
            print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))
        elif epoch % 100 == 0:
            print("Epoch is", epoch)
            print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))

        no_index = int(X_train.shape[0] / BATCH_SIZE)
        for index in range(no_index):

            # =============================================
            # Training Discriminator after generation
            # =============================================
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, CODE_LN)
            #print('noise.shape -->', noise.shape)
            generated_images = generator.predict(noise, verbose=0)
            if index % 20 == 0:
                """
                Saving data
                """
                if disp:
                    print(index)
            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            if disp:
                print('image_batch.shape, generated_images.shape -->', 
                      image_batch.shape, generated_images.shape)
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)
            if disp:
                print("batch %d d_loss : %f" % (index, d_loss))

            # =============================================
            # Training Generator using discriminator information
            # =============================================
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, CODE_LN)
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * BATCH_SIZE)
            discriminator.trainable = True
            if disp:
                print("batch %d g_loss : %f" % (index, g_loss))

            if index == no_index - 1:
                if disp:
                    print('Save generator and discriminator!')
                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)      

def train_VI_randn(BATCH_SIZE, V, I=None, no_epoch = 100, disp=False):
    """
    Developing
    ==========
    [2017-3-14]
    - To include saving the progress: tsplot of input and output
    - Input can be saved separately. 
    """
    if I is None:
        X_train = V
    else:
        X_train = np.concatenate([V, I], axis=1)

    # X_train = X_train.astype(np.float32)
    # X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    discriminator = discriminator_model()
    generator = generator_model()
    discriminator_on_generator = \
        generator_containing_discriminator(generator, discriminator)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(
        loss='binary_crossentropy', optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((BATCH_SIZE, CODE_LN))

    for epoch in range(no_epoch):
        if disp:
            print("Epoch is", epoch)
            print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))
        elif epoch % 100 == 0:
            print("Epoch is", epoch)
            print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))

        no_index = int(X_train.shape[0] / BATCH_SIZE)
        for index in range(no_index):

            # =============================================
            # Training Discriminator after generation
            # =============================================
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.randn(CODE_LN)
            #print('noise.shape -->', noise.shape)
            generated_images = generator.predict(noise, verbose=0)
            if index % 20 == 0:
                """
                Saving data
                """
                if disp:
                    print(index)
            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            if disp:
                print('image_batch.shape, generated_images.shape -->', 
                      image_batch.shape, generated_images.shape)
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)
            if disp:
                print("batch %d d_loss : %f" % (index, d_loss))

            # =============================================
            # Training Generator using discriminator information
            # =============================================
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.randn(CODE_LN)
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * BATCH_SIZE)
            discriminator.trainable = True
            if disp:
                print("batch %d g_loss : %f" % (index, g_loss))

            if index == no_index - 1:
                if disp:
                    print('Save generator and discriminator!')
                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)  

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


def generate(no_images = 100):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator')

    noise = np.zeros((no_images, CODE_LN))
    for i in range(no_images):
        noise[i, :] = np.random.uniform(-1, 1, CODE_LN)

    generated_images = generator.predict(noise, verbose=1)

    return generated_images


def generate_randn(no_images = 100):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator')

    noise = np.zeros((no_images, CODE_LN))
    for i in range(no_images):
        noise[i, :] = np.random.randn(CODE_LN)

    generated_images = generator.predict(noise, verbose=1)

    return generated_images


class GAN1DV_DNN():
    def __init__(self, 
                 a_INPUT_LN=56,
                 a_generator_model=generator_model,     
                 rand='uniform',  
                 foldname='tmp',
                 disp=False):
        """
        Input
        =====
        foldname, string
        foldname of saving weights. 
        """

        global generator_model
        global INPUT_LN, CODE_LN

        self.rand = rand
        self.disp = disp
        self.foldname = foldname

        generator_model = a_generator_model
        INPUT_LN = a_INPUT_LN
        CODE_LN = int(a_INPUT_LN/2)

        if self.disp:
            print(INPUT_LN, CODE_LN) 

        os.makedirs(self.foldname, exist_ok=True)

    def train_r0(self, BATCH_SIZE, V, no_epoch=100):
        """
        Input
        =====
        rand can be 'uniform', 'randn'
        """

        if self.disp:
            print(INPUT_LN, CODE_LN) 

        s = time.time()
        if self.rand == 'randn':
            r = train_VI_randn(BATCH_SIZE, V, no_epoch=no_epoch, disp=self.disp)
        else:
            r = train_VI(BATCH_SIZE, V, no_epoch=no_epoch, disp=self.disp)
        e = time.time()
        print('Elasped time: {}s'.format(e - s))
        return r

    def train(self, BATCH_SIZE, V, no_epoch=100):
        """
        Input
        =====
        rand can be 'uniform', 'randn'
        """

        if self.disp:
            print(INPUT_LN, CODE_LN) 

        s = time.time()
        r = self.train_VI(BATCH_SIZE, V, no_epoch=no_epoch)
        e = time.time()
        print('Elasped time: {}s'.format(e - s))
        return r

    def generate(self, no_images):
        generator = generator_model()
        generator.compile(loss='binary_crossentropy', optimizer="SGD")
        generator.load_weights(self.foldname+'/generator')

        noise = np.zeros((no_images, CODE_LN))
        for i in range(no_images):
            noise[i, :] = self.random(CODE_LN)

        generated_images = generator.predict(noise, verbose=1)

        return generated_images            

    def random(self, n):
        if self.rand == 'randn':
            return np.random.randn(n)
        else:
            return np.random.uniform(-1, 1, n)

    def train_VI(self, BATCH_SIZE, V, I=None, no_epoch = 100):
        """
        Developing
        ==========
        [2017-3-14]
        - To include saving the progress: tsplot of input and output
        - Input can be saved separately. 
        """
        disp = self.disp

        if I is None:
            X_train = V
        else:
            X_train = np.concatenate([V, I], axis=1)

        # X_train = X_train.astype(np.float32)
        # X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        discriminator = discriminator_model()
        generator = generator_model()
        discriminator_on_generator = \
            generator_containing_discriminator(generator, discriminator)
        d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        generator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator_on_generator.compile(
            loss='binary_crossentropy', optimizer=g_optim)
        discriminator.trainable = True
        discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
        noise = np.zeros((BATCH_SIZE, CODE_LN))

        for epoch in range(no_epoch):
            if disp:
                print("Epoch is", epoch)
                print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))
            elif epoch % 100 == 0:
                print("Epoch is", epoch)
                print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))

            no_index = int(X_train.shape[0] / BATCH_SIZE)
            for index in range(no_index):

                # =============================================
                # Training Discriminator after generation
                # =============================================
                for i in range(BATCH_SIZE):
                    noise[i, :] = self.random(CODE_LN)
                #print('noise.shape -->', noise.shape)
                generated_images = generator.predict(noise, verbose=0)
                if index % 20 == 0:
                    """
                    Saving data
                    """
                    if disp:
                        print(index)
                image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
                if disp:
                    print('image_batch.shape, generated_images.shape -->', 
                          image_batch.shape, generated_images.shape)
                X = np.concatenate((image_batch, generated_images))
                y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
                d_loss = discriminator.train_on_batch(X, y)
                if disp:
                    print("batch %d d_loss : %f" % (index, d_loss))

                # =============================================
                # Training Generator using discriminator information
                # =============================================
                for i in range(BATCH_SIZE):
                    noise[i, :] = self.random(CODE_LN)
                discriminator.trainable = False
                g_loss = discriminator_on_generator.train_on_batch(
                    noise, [1] * BATCH_SIZE)
                discriminator.trainable = True
                if disp:
                    print("batch %d g_loss : %f" % (index, g_loss))

                if index == no_index - 1:
                    if disp:
                        print('Save generator and discriminator!')
                    generator.save_weights(self.foldname + '/generator', True)
                    discriminator.save_weights(self.foldname + '/discriminator', True)  