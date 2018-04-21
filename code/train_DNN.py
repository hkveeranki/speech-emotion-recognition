from keras import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np

from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

from utilities import get_data, display_metrics

x_train, x_test, y_train, y_test = get_data(False)
orig_test = np.array(y_test)
encoder = LabelEncoder()
encoder.fit(np.concatenate((y_test, y_train), axis=0))
y_train = np_utils.to_categorical(encoder.transform(y_train))
y_test = np_utils.to_categorical(encoder.transform(y_test))
print(y_train.shape)
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(x_train[0].shape[0], x_train[0].shape[1])))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(x_train, y_train, batch_size=32, epochs=45, verbose=True, validation_data=(x_test, y_test))

y_pred = model.predict(x_test)
pred = [np.argmax(x) for x in y_pred]
display_metrics(pred, orig_test)
