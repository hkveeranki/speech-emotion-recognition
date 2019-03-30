"""
This file contains classes which implement deep neural networks namely CNN and LSTM
"""
import sys

import numpy as np
from keras import Sequential
from keras.layers import LSTM as KERAS_LSTM, Dense, Dropout, Conv2D, Flatten, \
    BatchNormalization, Activation, MaxPooling2D

from . import Model


class DNN(Model):
    """
    This class is parent class for all Deep neural network models. Any class
    inheriting this class should implement `make_default_model` method which
    creates a model with a set of hyper parameters.
    """

    def __init__(self, input_shape, num_classes, **params):
        """
        Constructor to initialize the deep neural network model. Takes the input
        shape and number of classes and other parameters required for the
        abstract class `Model` as parameters.

        Args:
            input_shape (tuple): shape of the input
            num_classes (int): number of different classes ( labels ) in the data.
            **params: Additional parameters required by the underlying abstract
                class `Model`.

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
        """
        Load the model weights from the given path.

        Args:
            to_load (str): path to the saved model file in h5 format.

        """
        try:
            self.model.load_weights(to_load)
        except:
            sys.stderr.write("Invalid saved file provided")
            sys.exit(-1)

    def save_model(self):
        """
        Save the model weights to `save_path` provided while creating the model.
        """
        self.model.save_weights(self.save_path)

    def train(self, x_train, y_train, x_val=None, y_val=None, n_epochs=50):
        """
        Train the model on the given training data.


        Args:
            x_train (numpy.ndarray): samples of training data.
            y_train (numpy.ndarray): labels for training data.
            x_val (numpy.ndarray): Optional, samples in the validation data.
            y_val (numpy.ndarray): Optional, labels of the validation data.
            n_epochs (int): Number of epochs to be trained.

        """
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

    def predict_one(self, sample):
        if not self.trained:
            sys.stderr.write(
                "Model should be trained or loaded before doing predict\n")
            sys.exit(-1)
        return np.argmax(self.model.predict(np.array([sample])))

    def make_default_model(self) -> None:
        """
        Make the model with default hyper parameters
        """
        # This has to be implemented by child classes. The reason is that the
        # hyper parameters depends on the model.
        raise NotImplementedError()


class CNN(DNN):
    """
    This class handles CNN for speech emotion recognitions
    """

    def __init__(self, **params):
        params['name'] = 'CNN'
        super(CNN, self).__init__(**params)

    def make_default_model(self):
        """
        Makes a CNN keras model with the default hyper parameters.
        """
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

    def __init__(self, **params):
        params['name'] = 'LSTM'
        super(LSTM, self).__init__(**params)

    def make_default_model(self):
        """
        Makes the LSTM model with keras with the default hyper parameters.
        """
        self.model.add(
            KERAS_LSTM(128,
                       input_shape=(self.input_shape[0], self.input_shape[1])))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(16, activation='tanh'))
