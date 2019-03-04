"""
This example script uses the library `speechemotionrecognition` and do the training and evaluating the models on
"""
from keras.utils import np_utils

from speechemotionrecognition.dnn import LSTM, CNN
from speechemotionrecognition.mlmodel import NN, SVM, RF
from speechemotionrecognition.utilities import get_data

data_path = 'dataset'

class_labels = ["Neutral", "Angry", "Happy", "Sad"]

def dnn_example():
    x_train, x_test, y_train, y_test = get_data(data_path=data_path, flatten=False)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    print('Starting LSTM')
    model = LSTM(input_shape=x_train[0].shape, num_classes=len(class_labels))
    model.train(x_train, y_train, x_test, y_test)
    model.evaluate(x_test, y_test)
    print('LSTM Done\n Starting CNN')
    in_shape = x_train[0].shape
    x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)
    model = CNN(input_shape=x_train[0].shape, num_classes=len(class_labels))
    model.train(x_train, y_train, x_test, y_test)
    model.evaluate(x_test, y_test)
    print('CNN Done')


def ml_example():
    x_train, x_test, y_train, y_test = get_data(data_path=data_path)
    models = [NN, RF, SVM]
    for M in models:
        model = M()
        print('Starting', model.name)
        model.train(x_train, y_train)
        model.evaluate(x_test, y_test)
        print(model.name, 'Done')


if __name__ == "__main__":
    ml_example()
    dnn_example()
