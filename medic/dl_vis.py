from keras.models import Sequential
from keras.layers import Input, Dropout
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, \
    UpSampling2D, BatchNormalization, Activation
import keras


class CNN_r0(Sequential):
    def __init__(model, input_shape, num_classes):
        """
        Score: [1.3923928976058959, 0.22500000000000001]
        """
        super().__init__()
        model.add(Conv2D(1, (2, 2), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2))) # 72

        model.add(Flatten())

        model.add(Dense(10))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))

        model.add(Dense(num_classes, activation='softmax', name='preds'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])


class CNN_r1(Sequential):
    def __init__(model, input_shape, num_classes):
        """
        Params: epochs = 100, batch_size = 100
        Score: [3.8031817436218263, 0.22500000000000001]
        """
        super().__init__()
        model.add(Conv2D(2, (2, 2), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(10))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))

        model.add(Dense(num_classes, activation='softmax', name='preds'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])


class CNN_r2(Sequential):
    def __init__(model, input_shape, num_classes):
        """
        Params: epochs = 100, batch_size = 100
        Score: [0.9332149744033813, 0.63749999999999996]
        Score: [3.5972158432006838, 0.22500000000000001]
        """
        super().__init__()
        model.add(Conv2D(4, (4, 4), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(4, 4)))

        model.add(Flatten())

        model.add(Dense(10))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))

        model.add(Dense(num_classes, activation='softmax', name='preds'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])


class CNN_r3(Sequential):
    def __init__(model, input_shape, num_classes):
        """
        Score: [5.7534255027771, 0.22500000000000001]
        """
        super().__init__()
        model.add(Dropout(0.5, input_shape=input_shape))
        model.add(Conv2D(8, (4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(4, 4)))

        model.add(Flatten())

        model.add(Dense(10))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))

        model.add(Dense(num_classes, activation='softmax', name='preds'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])


class CNN_r4(Sequential):
    def __init__(model, input_shape, num_classes):
        """
        Score: [3.3905357360839843, 0.22500000000000001]
        """
        super().__init__()
        model.add(Dropout(0.5, input_shape=input_shape))
        model.add(Conv2D(8, (4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Dropout(0.5))
        
        model.add(Flatten())

        model.add(Dense(10))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))

        model.add(Dense(num_classes, activation='softmax', name='preds'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])


class CNN_r5(Sequential):
    def __init__(model, input_shape, num_classes):
        """
        Score: [1.7992271423339843, 0.22500000000000001]
        """
        super().__init__()
        model.add(Dropout(0.5, input_shape=input_shape))
        model.add(Conv2D(8, (4, 4), activation='relu'))
        model.add(MaxPooling2D(pool_size=(8, 8)))
        model.add(Dropout(0.5))
        
        model.add(Flatten())

        model.add(Dense(10))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))

        model.add(Dense(num_classes, activation='softmax', name='preds'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])


class CNN_r6(Sequential):
    def __init__(model, input_shape, num_classes):
        """
        Score: [1.4706189155578613, 0.22500000000000001]
        """
        super().__init__()
        model.add(Dropout(0.5, input_shape=input_shape))
        model.add(Conv2D(8, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(5, 5)))
        model.add(Dropout(0.5))

        model.add(Conv2D(8, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.5))        
        
        model.add(Flatten())

        model.add(Dense(10))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))

        model.add(Dense(num_classes, activation='softmax', name='preds'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
