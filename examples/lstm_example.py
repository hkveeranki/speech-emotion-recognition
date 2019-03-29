from keras.utils import np_utils

from speechemotionrecognition.dnn import LSTM
from speechemotionrecognition.utilities import get_data

import common


def lstm_example():
    x_train, x_test, y_train, y_test = get_data(
        data_path=common.DATA_PATH, flatten=False)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    print('Starting LSTM')
    model = LSTM(input_shape=x_train[0].shape,
                 num_classes=len(common.CLASS_LABELS))
    model.train(x_train, y_train, x_test, y_test)
    model.evaluate(x_test, y_test)
    print('LSTM done.')


if __name__ == '__main__':
    lstm_example()
