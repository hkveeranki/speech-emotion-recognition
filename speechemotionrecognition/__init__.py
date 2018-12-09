"""
speechemotionrecognition module
Provides a library to perform speech emotion recognition on `emodb` dataset
"""
import sys

__author__ = 'harry-7'


class Model(object):
    """
    Model is the abstract class which determines how a model should be
    """

    def __init__(self, save_path=None, name='Not Specified', **params):
        """
        Default constructor
        """
        # Place holder for model
        self.model = None
        # Place holder on where to save the model
        self.save_path = save_path
        # Place holder for name of the model
        self.name = name
        # Model has been trained or not
        self.trained = False

    def train(self, x_train, y_train, x_val=None, y_val=None):
        """
        Trains the model with the given training data
        :param x_train: training samples
        :param y_train:  traning labels
        """
        self.model.fit(x_train, y_train)
        self.trained = True
        if self.save_path:
            self.save_model()

    def predict(self, data):
        """
        Predict labels for given data
        :param data: data for which labels need to be predicted
        :return:
        """
        if not self.trained:
            sys.stderr.write("Model should be trained or loaded before doing predict\n")
            sys.exit(-1)
        return self.model.predict(data)

    def restore_model(self, load_path=None):
        """
        restore the weights to the model
        :param load_path: optional, path to load the weights from a given path
        :return:
        """
        to_load = load_path or self.save_path
        if to_load is None:
            sys.stderr.write("Provide a path to load from or save_path of the model\n")
            sys.exit(-1)
        self.load_model(to_load)
        self.trained = True

    def load_model(self, to_load):
        """
        Load the weights from the given saved model
        :param to_load: path from where to load saved model
        """
        # This will be specific to model so should be implemented by child classes
        raise NotImplementedError()

    def save_model(self):
        """
        Save the model to `save_path`
        """
        # This will be specific to model so should be implemented by child classes
        raise NotImplementedError()

    def evaluate(self, x_test, y_test):
        """
        Evaluate the model with given test data and labels
        :param x_test: test data samples
        :param y_test: test data labels
        :return: Evaluation measures for the model
        """
        # This will be specific to model so should be implemented by child classes
        raise NotImplementedError()
