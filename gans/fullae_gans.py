"""
GANs with Autoencoder based on a combined cost function. 
"""

from keras.models import Sequential
from keras.layers import Dense, Merge
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
from keras import backend as K

import numpy as np
from PIL import Image
import argparse
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, subplot, imshow, axis, figure, title


# kera (Underdeveloping)
class Seq(Sequential):
    def __init__(self):
        pass

    def act(self):
        self.add(Activiation())
        return self


def generator_model_r0():  # CDNN Model
    model = Sequential()
    model.add(Convolution2D(
        1, 5, 5,
        border_mode='same',
        input_shape=(1, 14, 14)))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    #model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(1, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    return model


def generator_model():  # CDNN Model
    model = Sequential()
    model.add(Convolution2D(
        1, 5, 5,
        border_mode='same',
        input_shape=(1, 14, 14)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(1, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(Convolution2D(
        64, 5, 5,
        border_mode='same',
        input_shape=(1, 28, 28)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, 5, 5))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(generator, discriminator):
    model = Sequential()
    model.add(generator)
    discriminator.trainable = False
    model.add(discriminator)
    return model


def generator_containing_discriminator_ae(generator, discriminator):
    model_left = Sequential()
    model_left.add(generator)
    discriminator.trainable = False
    model_left.add(discriminator)

    model_right = Sequential()
    model_right.add(generator)
    model_right.add(Reshape((784,)))

    model = Sequential()
    model.add(Merge([model_left, model_right], mode='concat', concat_axis=1))
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


def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def merge_bc_mse(y_true, y_pred):
    c1 = binary_crossentropy(y_true[:,0], y_pred[:,0])
    c2 = mean_squared_error(y_true[:,1:], y_pred[:,1:])
    #return K.sum(c1, c2)
    return c1 + c2


def train_aegans(BATCH_SIZE, disp=True):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
    discriminator = discriminator_model()
    generator = generator_model()
    discriminator_on_generator = \
        generator_containing_discriminator_ae(generator, discriminator)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(loss=merge_bc_mse, optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((BATCH_SIZE, 100))
    for epoch in range(100):
        if disp:
            print("Epoch is", epoch)
            print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))

        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            # for i in range(BATCH_SIZE):
                #noise[i, :] = np.random.uniform(-1, 1, 100)
            #    noise[i, :] = X_train[i, :].reshape(-1)[np.round(np.linspace(0,783,100)).astype(int)]

            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            noise = image_batch[:, :, ::2, ::2].copy()
            generated_images = generator.predict(noise, verbose=0)

            if index % 30 == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch) + "_" + str(index) + ".png")

            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)
            if disp and index % 30 == 0:
                print("batch %d d_loss : %f" % (index, d_loss))

            noise = image_batch[:, :, ::2, ::2].copy()
            discriminator.trainable = False

            target_left = np.array([1] * BATCH_SIZE).reshape(BATCH_SIZE, 1)
            target_right = image_batch.reshape(BATCH_SIZE, -1)
            target = np.concatenate([target_left, target_right], axis=1)
            # Debuging code
            # print("epoch, index, target.shape -->", epoch, index, target.shape)
            g_loss = discriminator_on_generator.train_on_batch(
                noise, target)

            discriminator.trainable = True            
            if disp and index % 30 == 0:
                print("batch %d g_loss : %f" % (index, g_loss))

            if index % 10 == 9:
                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)

def train_org(BATCH_SIZE):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
    discriminator = discriminator_model()
    generator = generator_model()
    discriminator_on_generator = \
        generator_containing_discriminator(generator, discriminator)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(
        loss=binary_crossentropy, optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((BATCH_SIZE, 100))
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            # for i in range(BATCH_SIZE):
                #noise[i, :] = np.random.uniform(-1, 1, 100)
            #    noise[i, :] = X_train[i, :].reshape(-1)[np.round(np.linspace(0,783,100)).astype(int)]

            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            noise = image_batch[:, :, ::2, ::2].copy()
            generated_images = generator.predict(noise, verbose=0)

            if index % 30 == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch) + "_" + str(index) + ".png")

            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)
            if index % 30 == 0:
                print("batch %d d_loss : %f" % (index, d_loss))

            noise = image_batch[:, :, ::2, ::2].copy()
            discriminator.trainable = False

            target_left = np.array([1] * BATCH_SIZE).reshape(BATCH_SIZE, 1)
            target_right = image_batch.reshape(BATCH_SIZE, -1)
            target = np.concatenate([target_left, target_right], axis=1)

            # Debuging code
            print("epoch, index, target.shape -->", epoch, index, target.shape)
            
            # g_loss = discriminator_on_generator.train_on_batch(noise, target)
            # g_loss = discriminator_on_generator.train_on_batch(noise, target_left)
            g_loss = discriminator_on_generator.train_on_batch(noise, [1] * BATCH_SIZE)

            discriminator.trainable = True
            if index % 30 == 0:
                print("batch %d g_loss : %f" % (index, g_loss))

            if index % 10 == 9:
                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)


def train_gans(BATCH_SIZE):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
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
    noise = np.zeros((BATCH_SIZE, 100))
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            # for i in range(BATCH_SIZE):
                #noise[i, :] = np.random.uniform(-1, 1, 100)
            #    noise[i, :] = X_train[i, :].reshape(-1)[np.round(np.linspace(0,783,100)).astype(int)]

            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            noise = image_batch[:, :, ::2, ::2].copy()
            generated_images = generator.predict(noise, verbose=0)
            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch) + "_" + str(index) + ".png")
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))

            noise = image_batch[:, :, ::2, ::2].copy()
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * BATCH_SIZE)
            discriminator.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)


def train(BATCH_SIZE):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
    discriminator = discriminator_model()
    generator = generator_model()
    discriminator_on_generator = \
        generator_containing_discriminator_ae(generator, discriminator)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(
        loss=merge_bc_mse, optimizer=g_optim)
    discriminator.trainable = True
    discriminator.compile(loss='binary_crossentropy', optimizer=d_optim)
    noise = np.zeros((BATCH_SIZE, 100))
    for epoch in range(100):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0] / BATCH_SIZE))
        for index in range(int(X_train.shape[0] / BATCH_SIZE)):
            # for i in range(BATCH_SIZE):
                #noise[i, :] = np.random.uniform(-1, 1, 100)
            #    noise[i, :] = X_train[i, :].reshape(-1)[np.round(np.linspace(0,783,100)).astype(int)]

            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
            noise = image_batch[:, :, ::2, ::2].copy()
            generated_images = generator.predict(noise, verbose=0)

            if index % 30 == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    str(epoch) + "_" + str(index) + ".png")

            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)
            if index % 30 == 0:
                print("batch %d d_loss : %f" % (index, d_loss))

            noise = image_batch[:, :, ::2, ::2].copy()
            discriminator.trainable = False

            target_left = np.array([1] * BATCH_SIZE).reshape(BATCH_SIZE, 1)
            target_right = image_batch.reshape(BATCH_SIZE, -1)
            target = np.concatenate([target_left, target_right], axis=1)
            # Debuging code
            # print("epoch, index, target.shape -->", epoch, index, target.shape)
            g_loss = discriminator_on_generator.train_on_batch(
                noise, target)

            discriminator.trainable = True
            if index % 30 == 0:
                print("batch %d g_loss : %f" % (index, g_loss))

            if index % 10 == 9:
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

def to_normal(X_train):
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
    return X_train

def tsting_and_show(no_images = 100):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_test = to_normal(X_test)

    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator')

    noise = X_test[:,:,::2,::2]
    generated_images = generator.predict(noise, verbose=1)
    im = combine_images(generated_images[:no_images])
    im_org = combine_images(X_test[:no_images])
    im_dec = combine_images(noise[:no_images])

    figure(figsize=(4*3+2,4))
    subplot(1,3,1)
    imshow(im_org, cmap='gray')
    axis('off')
    title('Original')

    subplot(1,3,2)
    imshow(im_dec, cmap='gray')
    axis('off')
    title('Input 1/2x1/2 with Interpol')

    subplot(1,3,3)
    imshow(im, cmap='gray')
    axis('off')
    title('Expansion by AE-GANS')