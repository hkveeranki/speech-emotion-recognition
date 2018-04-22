from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Conv2D, AveragePooling1D, AveragePooling2D, Flatten, BatchNormalization, \
    Activation, MaxPooling2D
import numpy as np

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import utilities

from utilities import get_data, display_metrics

models = ["CNN", "LSTM"]
x_train, x_test, y_train, y_test = get_data(False)


def get_model(model_name):
    """
    Generate the required model and return it
    :return: Model created
    """
    if model_name == 'CNN':

        model = Sequential()
        model.add(Conv2D(16, (2, 2),
                         input_shape=(x_train[0].shape[0], x_train[0].shape[1], 1)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (2, 2)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))

        model.add(Conv2D(32, (2, 2)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (2, 2)))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 1)))

        model.add(Flatten())

        # Fully connected layer
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        # model.add(BatchNormalization())
        # model.add(AveragePooling2D(pool_size=(2, 2)))
        # model.add(Conv2D(filters=16, kernel_size=(7, 1)))
        # model.add(Dropout(0.5))
        # model.add(Flatten())
        # model.add(Dense(480, activation='tanh'))
        # model.add(Dropout(0.4))
        # model.add(Dense(120, activation='sigmoid'))
        # model.add(Dropout(0.3))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(len(utilities.class_labels), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif model_name == 'LSTM':
        model = Sequential()
        model.add(LSTM(128, input_shape=()))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(16, activation='tanh'))
        model.add(Dense(len(utilities.class_labels), activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())
    return model


orig_test = np.array(y_test)
encoder = LabelEncoder()
encoder.fit(np.concatenate((y_test, y_train), axis=0))
y_train = np_utils.to_categorical(encoder.transform(y_train))
y_test = np_utils.to_categorical(encoder.transform(y_test))
model = get_model('CNN')
in_shape = x_train[0].shape
x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)
model.fit(x_train, y_train, batch_size=32, epochs=50, verbose=True, validation_data=(x_test, y_test))

y_pred = model.predict(x_test)
pred = [np.argmax(x) for x in y_pred]
display_metrics(pred, orig_test)
