"""
Train a model and display its metrics
"""
import sys

from utilities import get_data, display_metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC as SVC

models = ['SVM', 'Random Forest', 'Neural network']


def get_model(model_name):
    """
    Create a model and return it
    :param model_name: name of the model to be created
    :return: created model with fit and predict methods
    """
    if model_name == models[0]:
        return SVC(multi_class='crammer_singer')
    elif model_name == models[1]:
        return RandomForestClassifier(n_estimators=30)
    elif model_name == models[2]:
        return MLPClassifier(activation='logistic', verbose=True,
                             hidden_layer_sizes=(512,), batch_size=32)


def evaluateModel(model_name):
    """
    generate a model train it test it and display its metrics
    :param model_name:
    """
    clf = get_model(model_name)
    x_train, x_test, y_train, y_test = get_data()
    print '------------- Training Started -------------'
    clf.fit(x_train, y_train)
    print '------------- Training Ended -------------'
    y_pred = clf.predict(x_test)
    display_metrics(y_pred, y_test)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        sys.stderr.write('Invalid arguments\n')
        sys.stderr.write('Usage python2 train_DNN.py <model_number>\n')
        sys.stderr.write('1 - SVM\n')
        sys.stderr.write('2 - Random Forest\n')
        sys.stderr.write('3 - Neural Network\n')
        sys.exit(-1)

    n = int(sys.argv[1]) - 1

    if n > len(models):
        sys.stderr.write('Invalid Model number')
        sys.exit(-1)

    print 'model given', models[n]

    evaluateModel(models[n])
