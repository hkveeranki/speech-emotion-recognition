"""
This file contains classes which implement deep neural networks namely CNN and LSTM
"""
import sys

from keras import Sequential
from keras.layers import LSTM as lstm, Dense, Dropout, Conv2D, Flatten, \
    BatchNormalization, Activation, MaxPooling2D
import numpy as np

from . import Model


class DNN(Model):
    """
    This class is parent class for all Deep neural network models
    """

    def __init__(self, input_shape, num_classes, **params):
        """
        Constructor to initialize the deep neural network model
        :param input_shape: shape of the input data
        :param num_classes: number of classes in the data
        """
        super(DNN, self).__init__(**params)
        self.input_shape = input_shape
        self.model = Sequential()
        self.make_default_model()
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                           metrics=['accuracy'])
        print(self.model.summary(), file=sys.stderr)
        self.save_path = self.save_path or self.name + '_best_model.h5'

    def load_model(self, to_load):
        try:
            self.model.load_weights(to_load)
        except:
            sys.stderr.write("Invalid saved file provided")
            sys.exit(-1)

    def save_model(self):
        self.model.save_weights(self.save_path)

    def evaluate(self, x_test, y_test):
        print('Accuracy =', self.model.evaluate(x_test, y_test)[1])

    def train(self, x_train, y_train, x_val=None, y_val=None, n_epochs=50):
        best_acc = 0
        if x_val is None or y_val is None:
            x_val, y_val = x_train, y_train
        for i in range(n_epochs):
            # Shuffle the data for each epoch in unison inspired
            # from https://stackoverflow.com/a/4602224
            p = np.random.permutation(len(x_train))
            x_train = x_train[p]
            y_train = y_train[p]
            self.model.fit(x_train, y_train, batch_size=32, epochs=1)
            loss, acc = self.model.evaluate(x_val, y_val)
            if acc > best_acc:
                best_acc = acc
        self.trained = True

    def make_default_model(self):
        """
        Make the model with default hyper parameters
        """
        raise NotImplementedError()


class CNN(DNN):
    """
    This class handles CNN for speech emotion recognitions
    """

    def __init__(self, input_shape, num_classes, **params):
        params['name'] = 'CNN'
        super(CNN, self).__init__(input_shape, num_classes, **params)

    def make_default_model(self):
        self.model.add(Conv2D(8, (13, 13),
                              input_shape=(
                                  self.input_shape[0], self.input_shape[1], 1)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(8, (13, 13)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 1)))
        self.model.add(Conv2D(8, (13, 13)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(8, (2, 2)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 1)))
        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))


class LSTM(DNN):
    """
    This class handles CNN for speech emotion recognitions
    """

    def __init__(self, input_shape, num_classes, **params):
        params['name'] = 'LSTM'
        super(LSTM, self).__init__(input_shape, num_classes, **params)

    def make_default_model(self):
        self.model.add(
            lstm(128, input_shape=(self.input_shape[0], self.input_shape[1])))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(16, activation='tanh'))
