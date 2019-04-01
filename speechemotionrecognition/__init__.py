"""
speechemotionrecognition module.
Provides a library to perform speech emotion recognition on `emodb` data set
"""
import sys
from typing import Tuple

import numpy
from sklearn.metrics import accuracy_score, confusion_matrix

__author__ = 'harry-7'
__version__ = '1.1'


class Model(object):
    """
    Model is the abstract class which determines how a model should be.
    Any model inheriting this class should do the following.

    1.  Set the model instance variable to the corresponding model class which
        which will provide methods `fit` and `predict`.

    2.  Should implement the following abstract methods `load_model`,
        `save_model` `train` and `evaluate`. These methods provide the
        functionality to save the model to the disk, load the model from the
        disk and train the model and evaluate the model to return appropriate
        measure like accuracy, f1 score, etc.

    Attributes:
        model (Any): instance variable that holds the model.
        save_path (str): path to save the model.
        name (str): name of the model.
        trained (bool): True if model has been trained, false otherwise.
    """

    def __init__(self, save_path: str = '', name: str = 'Not Specified'):
        """
        Default constructor for abstract class Model.

        Args:
            save_path(str): path to save the model to.
            name(str): name of the model given as string.

        """
        # Place holder for model
        self.model = None
        # Place holder on where to save the model
        self.save_path = save_path
        # Place holder for name of the model
        self.name = name
        # Model has been trained or not
        self.trained = False

    def train(self, x_train: numpy.ndarray, y_train: numpy.ndarray,
              x_val: numpy.ndarray = None,
              y_val: numpy.ndarray = None) -> None:
        """
        Trains the model with the given training data.

        Args:
            x_train (numpy.ndarray): samples of training data.
            y_train (numpy.ndarray): labels for training data.
            x_val (numpy.ndarray): Optional, samples in the validation data.
            y_val (numpy.ndarray): Optional, labels of the validation data.

        """
        # This will be specific to model so should be implemented by
        # child classes
        raise NotImplementedError()

    def predict(self, samples: numpy.ndarray) -> Tuple:
        """
        Predict labels for given data.

        Args:
            samples (numpy.ndarray): data for which labels need to be predicted

        Returns:
            list: list of labels predicted for the data.

        """
        results = []
        for _, sample in enumerate(samples):
            results.append(self.predict_one(sample))
        return tuple(results)

    def predict_one(self, sample) -> int:
        """
        Predict label of a single sample. The reason this method exists is
        because often we might want to predict label for a single sample.

        Args:
            sample (numpy.ndarray): Feature vector of the sample that we want to
                                    predict the label for.

        Returns:
            int: returns the label for the sample.
        """
        # This need to be implemented for the child models. The reason is that
        # ML models and DL models predict the labels differently.
        raise NotImplementedError()

    def restore_model(self, load_path: str = None) -> None:
        """
        Restore the weights from a saved model and load them to the model.

        Args:
            load_path (str): Optional, path to load the weights from a given path.

        """
        to_load = load_path or self.save_path
        if to_load is None:
            sys.stderr.write(
                "Provide a path to load from or save_path of the model\n")
            sys.exit(-1)
        self.load_model(to_load)
        self.trained = True

    def load_model(self, to_load: str) -> None:
        """
        Load the weights from the given saved model.

        Args:
            to_load: path containing the saved model.

        """
        # This will be specific to model so should be implemented by
        # child classes
        raise NotImplementedError()

    def save_model(self) -> None:
        """
        Save the model to path denoted by `save_path` instance variable.
        """
        # This will be specific to model so should be implemented by
        # child classes
        raise NotImplementedError()

    def evaluate(self, x_test: numpy.ndarray, y_test: numpy.ndarray) -> None:
        """
        Evaluate the current model on the given test data.

        Predict the labels for test data using the model and print the relevant
        metrics like accuracy and the confusion matrix.

        Args:
            x_test (numpy.ndarray): Numpy nD array or a list like object
                                    containing the samples.
            y_test (numpy.ndarray): Numpy 1D array or list like object
                                    containing the labels for test samples.
        """
        predictions = self.predict(x_test)
        print(y_test)
        print(predictions)
        print('Accuracy:%.3f\n' % accuracy_score(y_pred=predictions,
                                                 y_true=y_test))
        print('Confusion matrix:', confusion_matrix(y_pred=predictions,
                                                    y_true=y_test))
