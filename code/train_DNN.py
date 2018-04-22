import numpy as np
import sys
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv2D, Flatten, \
    BatchNormalization, Activation, MaxPooling2D
from keras.utils import np_utils
from tqdm import tqdm

from utilities import get_data, class_labels

models = ["CNN", "LSTM"]


def get_model(model_name, input_shape):
    """
    Generate the required model and return it
    :return: Model created
    """
    # Models are inspired from
    # CNN - https://yashk2810.github.io/Applying-Convolutional-Neural-Network-on-the-MNIST-dataset/
    # LSTM - https://github.com/harry-7/Deep-Sentiment-Analysis/blob/master/code/generatePureLSTM.py
    model = Sequential()
    if model_name == 'CNN':
        model.add(Conv2D(8, (13, 13),
                         input_shape=(input_shape[0], input_shape[1], 1)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(Conv2D(8, (13, 13)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Conv2D(8, (13, 13)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(Conv2D(8, (2, 2)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
    elif model_name == 'LSTM':
        model.add(LSTM(128, input_shape=(input_shape[0], input_shape[1])))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='tanh'))
    model.add(Dense(len(class_labels), activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def evaluateModel(model):
    """
    Train the model and evaluate it
    :param model: model to be evaluted
    """
    # Train the epochs
    best_acc = 0
    global x_train, y_train, x_test, y_test
    for i in tqdm(range(50)):
        # Shuffle the data for each epoch in unison inspired from https://stackoverflow.com/a/4602224
        p = np.random.permutation(len(x_train))
        x_train = x_train[p]
        y_train = y_train[p]
        model.fit(x_train, y_train, batch_size=32, epochs=1)
        loss, acc = model.evaluate(x_test, y_test)
        if acc > best_acc:
            print 'Updated best accuracy', acc
            best_acc = acc
            model.save_weights(best_model_path)
    model.load_weights(best_model_path)
    print 'Accuracy = ', model.evaluate(x_test, y_test)[1]


if __name__ == "__main__":

    if len(sys.argv) != 1:
        sys.stderr.write('Invalid arguments\n')
        sys.stderr.write('Usage python2 train_DNN.py <model_number>\n')
        sys.stderr.write('1 - CNN\n')
        sys.stderr.write('2 - LSTM\n')
        sys.exit(-1)

    n = int(sys.argv[1]) - 1
    print 'model given', models[n]

    # Read data
    global x_train, y_train, x_test, y_test
    x_train, x_test, y_train, y_test = get_data(flatten=False)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    if n == 0:
        # Model is CNN so have to reshape the data
        in_shape = x_train[0].shape
        x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)
    elif n > len(models):
        sys.stderr.write('Model Not Implemented yet')
        sys.exit(-1)

    model = get_model(models[n - 1], x_train[0].shape)

    global best_model_path
    best_model_path = '../models/best_model_' + models[n - 1] + '.h5'

    evaluateModel(model)
