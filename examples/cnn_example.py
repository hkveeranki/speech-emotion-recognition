"""
This example demonstrates how to use `CNN` model from
`speechemotionrecognition` package
"""
from keras.utils import np_utils

from common import extract_data
from speechemotionrecognition.dnn import CNN
from speechemotionrecognition.utilities import get_feature_vector_from_mfcc


def cnn_example():
    to_flatten = False
    x_train, x_test, y_train, y_test, num_labels = extract_data(
        flatten=to_flatten)
    y_train = np_utils.to_categorical(y_train)
    y_test_train = np_utils.to_categorical(y_test)
    in_shape = x_train[0].shape
    x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)
    model = CNN(input_shape=x_train[0].shape,
                num_classes=num_labels)
    model.train(x_train, y_train, x_test, y_test_train)
    model.evaluate(x_test, y_test)
    filename = '../dataset/Sad/09b03Ta.wav'
    print('prediction', model.predict_one(
        get_feature_vector_from_mfcc(filename, flatten=to_flatten)),
          'Actual 3')
    print('CNN Done')


if __name__ == "__main__":
    cnn_example()
