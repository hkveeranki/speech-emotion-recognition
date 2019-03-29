"""
This example script uses the library `speechemotionrecognition` and do the training and evaluating the models on
"""
from keras.utils import np_utils

import common
from speechemotionrecognition.dnn import CNN
from speechemotionrecognition.utilities import get_data


def cnn_example():
    x_train, x_test, y_train, y_test = get_data(
        data_path=common.DATA_PATH, flatten=False)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    in_shape = x_train[0].shape
    x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)
    model = CNN(input_shape=x_train[0].shape,
                num_classes=len(common.CLASS_LABELS))
    model.train(x_train, y_train, x_test, y_test)
    model.evaluate(x_test, y_test)
    print('CNN Done')


if __name__ == "__main__":
    cnn_example()
