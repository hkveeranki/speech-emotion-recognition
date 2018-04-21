"""

author: harry-7

This file contains functions to read the data files from the given folders and generate the data interms of features
"""
import numpy as np

import scipy.io.wavfile as wav
import os
import speechpy
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

dataset_folder = "/home/harry7/speech/dataset/"

data_folders = ["Neutral", "Angry", "Happy", "Sad"]

rs = 200  # required size to be padded


def read_wav(filename):
    """
    Read the wav file and return corresponding data
    :param filename: name of the file
    :return: return tuple containing sampling frequency and signal
    """
    return wav.read(filename)


def read_data():
    X_data = []
    Y_data = []
    max_fs = 0
    min_sample = int('9' * 10)
    cnt = 0
    os.chdir(dataset_folder)
    print os.getcwd()
    for i, directory in enumerate(data_folders):
        print "started folder", directory
        os.chdir(directory)
        for filename in os.listdir('.'):
            fs, signal = read_wav(filename)
            max_fs = max(max_fs, fs)
            mfcc = speechpy.feature.mfcc(signal, fs)
            mf_len = len(mfcc)
            if mf_len > rs:
                # mean size of mfcc vectors of all samples
                # experimentally calculated
                mfcc = mfcc[:rs, :]
            else:
                pad_len = rs - mf_len
                pad_rem = pad_len % 2
                pad_len /= 2
                mfcc = np.pad(mfcc, ((pad_len, pad_len + pad_rem), (0, 0)), 'constant', constant_values=0)
            min_sample = min(min_sample, len(mfcc))
            mfcc = mfcc.flatten()
            X_data.append(mfcc)
            Y_data.append(i)
            cnt += 1
        os.chdir('..')
    return X_data, Y_data


def get_data():
    """
    Read the files get the data perform the test-train split and return them to the caller
    :return: 4 arrays, x_train x_test y_train y_test
    """
    data, labels = read_data()
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


def display_metrics(y_pred, y_true):
    print accuracy_score(y_pred=y_pred, y_true=y_true)
    print confusion_matrix(y_pred=y_pred, y_true=y_true)


if __name__ == "__main__":
    features, labels = read_data()
    print "number of samples: ", len(labels), "number of features in sample", len(features[0])
