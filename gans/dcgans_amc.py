from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.convolutional import Convolution1D, Conv1D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import Image
import argparse
import math


def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128 * 7 * 7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((128, 7, 7), input_shape=(128 * 7 * 7,)))
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


def train(BATCH_SIZE):
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
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            image_batch = X_train[index * BATCH_SIZE:(index + 1) * BATCH_SIZE]
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
            for i in range(BATCH_SIZE):
                noise[i, :] = np.random.uniform(-1, 1, 100)
            discriminator.trainable = False
            g_loss = discriminator_on_generator.train_on_batch(
                noise, [1] * BATCH_SIZE)
            discriminator.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                generator.save_weights('generator', True)
                discriminator.save_weights('discriminator', True)


def generate(BATCH_SIZE, nice=False):
    generator = generator_model()
    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    generator.load_weights('generator')
    if nice:
        discriminator = discriminator_model()
        discriminator.compile(loss='binary_crossentropy', optimizer="SGD")
        discriminator.load_weights('discriminator')
        noise = np.zeros((BATCH_SIZE * 20, 100))
        for i in range(BATCH_SIZE * 20):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        d_pret = discriminator.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE * 20)
        index.resize((BATCH_SIZE * 20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE, 1) +
                               (generated_images.shape[2:]), dtype=np.float32)
        for i in range(int(BATCH_SIZE)):
            idx = int(pre_with_index[i][1])
            nice_images[i, 0, :, :] = generated_images[idx, 0, :, :]
        image = combine_images(nice_images)
    else:
        noise = np.zeros((BATCH_SIZE, 100))
        for i in range(BATCH_SIZE):
            noise[i, :] = np.random.uniform(-1, 1, 100)
        generated_images = generator.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image * 127.5 + 127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args


class GEN():
    """
    This class generates sequence using mathematical model.
    """

    def __init__(self):
        self.bin = None
        self.bipolar = None
        self.seq = None

    def binary(self, no_bits):
        self.bin = np.random.randint(0, 2, no_bits)
        self.bipolar = np.power(-1, self.bin)
        return self

    def to_bpsk(self):
        self.seq = self.bipolar
        return self

    def to_qpsk(self):
        p = self.bipolar
        ln = math.ceil(len(p) / 2)
        self.seq = np.zeros((ln, 2))
        self.seq[:, 0] = p[0::2]
        self.seq[:, 1] = p[1::2]
        return self

    def to_16qam(self):
        p = self.bipolar
        ln = math.ceil(len(p) / 4)
        self.seq = np.zeros((ln, 2))
        self.seq[:, 0] = 2 * p[0::4] + p[1::4]
        self.seq[:, 1] = 2 * p[2::4] + p[3::4]
        return self

    def __repr__(self):
        if self.bin is None:
            return "Empty"
        else:
            s = self.bin
            st = "Binary: " + '[' + ', '.join(map(str, s)) + ']'
            st_all = st
            s = self.bipolar
            st = "Bipolar: " + '[' + ', '.join(map(str, s)) + ']'
            st_all += '\n' + st

            if self.seq is not None:
                if np.ndim(self.seq) == 1:
                    s = self.seq
                    st = "Sequence: " + '[' + ', '.join(map(str, s)) + ']'
                    st_all += '\n' + st
                else:  # ndim == 2
                    s = self.seq[:, 0]
                    st = "Re(Sequence): " + '[' + ', '.join(map(str, s)) + ']'
                    st_all += '\n' + st
                    s = self.seq[:, 1]
                    st = "Im(Sequence): " + '[' + ', '.join(map(str, s)) + ']'
                    st_all += '\n' + st

            return st_all


def generator_model_bpsk(no_bits_in_a_frame):
    """
    BPSK outputs will be generated by CCN.
    CCN would be 1-2x because x is binary and the output should be bipolar. 
    Also, it is 1-tap processing. For 16-QAM, it will be more compliated. 
    I should consider how to optimize stride or oversampling/max polling 
    in a network. For GANs, hyperparameters can be more well optimized than 
    conventional feedforward networks. 
    While I was watching RNN-LSTM, I realized that many hyperparameters such as
    gating variables are optimized by networks itself. Those values have been optimized 
    by grid search or some other external techniques. However, RNN can do it by itself online. 
    These capability may come from RNN superpower. Similarly, many hyperparameters can be
    easily optimized in GANs. 
    """
    model = Sequential()
    model.add(Convolution1D(
        1, 1,
        input_shape=(no_bits_in_a_frame, 1)))
    return model


def generator_model_qpsk(no_bits_in_a_frame):
    """
    Now the outputs will have 2-d, consisting of real and image parts
    input should be the same output.
    """
    model = Sequential()
    model.add(Conv1D(
        2, 2,
        subsample_length=2,
        input_shape=(no_bits_in_a_frame, 1)))
    return model


def generator_model_16qam(no_bits_in_a_frame):
    """
    Now the outputs will have 2-d, consisting of real and image parts
    input should be the same output.
    """
    model = Sequential()
    model.add(Conv1D(
        2, 4,
        subsample_length=4,
        input_shape=(no_bits_in_a_frame, 1)))
    return model

class DeepGen(GEN):
    """
    This will generate BPSK or other modulation outputs using deep learning Keras.
    """

    def __init__(self):
        super().__init__()

    def to_bpsk(self):
        """
        For 1d conv, 3d data is required. The first dimension represents the number of frames, 
        the second dimension represents the size of a frame, and
        the last dimension represents to indicate real and image parts.
        """
        generator = generator_model_bpsk(len(self.bin))
        generator.compile(loss='binary_crossentropy', optimizer="SGD")

        # Setting weights for BPSK modulation
        w_4d = np.zeros([1, 1, 1, 1])
        b_1d = np.zeros([1, ])
        w_4d[0, 0, 0, 0] = -2
        b_1d[0] = 1
        generator.set_weights([w_4d, b_1d])

        seq_1da = np.array(self.bin)
        frame_rgb_data = seq_1da.reshape(1, -1, 1)
        seq_rgb_data = generator.predict(frame_rgb_data, verbose=0)

        """
        Because of BPSK, the output is 2d array with 2nd dimensio has size 1. 
        Hence, 2d to 1d makes no data loss. 
        """
        self.seq_rgb = seq_rgb_data.reshape(-1, 1)
        self.seq = self.seq_rgb.reshape(-1)

        return self

    def to_qpsk(self):
        """
        For 1d conv, 3d data is required. The first dimension represents the number of frames, 
        the second dimension represents the size of a frame, and
        the last dimension represents to indicate real and image parts.
        """
        generator = generator_model_qpsk(len(self.bin))
        generator.compile(loss='binary_crossentropy', optimizer="SGD")

        # Setting weights for BPSK modulation
        w_4d = np.zeros([2, 1, 1, 2])
        b_1d = np.zeros([2, ])
        w_4d[:, 0, 0, 0] = [0, -2]
        w_4d[:, 0, 0, 1] = [-2, 0]
        b_1d[0] = 1
        b_1d[1] = 1
        generator.set_weights([w_4d, b_1d])

        seq_1da = np.array(self.bin)
        frame_rgb_data = seq_1da.reshape(1, -1, 1)
        seq_rgb_data = generator.predict(frame_rgb_data, verbose=1)

        self.seq = seq_rgb_data.reshape(-1, 2)

        return self

    def to_16qam(self):
        """
        For 1d conv, 3d data is required. The first dimension represents the number of frames, 
        the second dimension represents the size of a frame, and
        the last dimension represents to indicate real and image parts.
        """
        generator = generator_model_16qam(len(self.bin))
        generator.compile(loss='binary_crossentropy', optimizer="SGD")

        # Setting weights for BPSK modulation
        w_4d = np.zeros([4, 1, 1, 2])
        b_1d = np.zeros([2, ])
        w_4d[:, 0, 0, 0] = [0, 0, -2, -4]
        w_4d[:, 0, 0, 1] = [-2, -4, 0, 0]
        b_1d[0] = 2 + 1
        b_1d[1] = 2 + 1
        generator.set_weights([w_4d, b_1d])

        seq_1da = np.array(self.bin)
        frame_rgb_data = seq_1da.reshape(1, -1, 1)
        seq_rgb_data = generator.predict(frame_rgb_data, verbose=1)

        self.seq = seq_rgb_data.reshape(-1, 2)

        return self
