"""
This file contains all the non deep learning models
"""
import pickle
import sys

import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from . import Model


class MLModel(Model):
    """
    This class is parent class for all Non Deep learning models
    """

    def __init__(self, **params):
        super(MLModel, self).__init__(**params)

    def evaluate(self, x_test:numpy.ndarray, y_test:numpy.ndarray) -> None:
        y_pred = self.predict(x_test)
        print('Accuracy:%.3f\n' % accuracy_score(y_pred=y_pred, y_true=y_test))
        print('Confusion matrix:', confusion_matrix(y_pred=y_pred,
                                                    y_true=y_test))

    def save_model(self):
        pickle.dump(self.model, open(self.save_path, "wb"))

    def load_model(self, to_load:str):
        try:
            self.model = pickle.load(open(self.save_path, "rb"))
        except:
            sys.stderr.write("Invalid saved file provided")
            sys.exit(-1)

    def train(self, x_train, y_train, x_val=None, y_val=None):
        self.model.fit(x_train, y_train)
        self.trained = True
        if self.save_path:
            self.save_model()


class SVM(MLModel):
    """
    SVM implements use of SVM for speech emotion recognition
    """

    def __init__(self, **params):
        params['name'] = 'SVM'
        super(SVM, self).__init__(**params)
        self.model = LinearSVC(multi_class='crammer_singer')


class RF(MLModel):
    """
    RF implements use of Random Forest for speech emotion recognition
    """

    def __init__(self, **params):
        params['name'] = 'Random Forest'
        super(RF, self).__init__(**params)
        self.model = RandomForestClassifier(n_estimators=30)


class NN(MLModel):
    """
    NN implements use of Neural networks for speech emotion recognition
    """

    def __init__(self, **params):
        params['name'] = 'Neural Network'
        super(NN, self).__init__(**params)
        self.model = MLPClassifier(activation='logistic', verbose=True,
                                   hidden_layer_sizes=(512,), batch_size=32)
