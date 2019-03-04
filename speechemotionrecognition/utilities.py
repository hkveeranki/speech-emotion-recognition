"""

author: harry-7

This file contains functions to read the data files from the given folders and
generate Mel Frequency Cepestral Coefficients features for the given audio
files as training samples.
"""
import numpy as np

import scipy.io.wavfile as wav
import os
import sys
from speechpy.feature import mfcc
from sklearn.model_selection import train_test_split

mean_signal_length = 32000  # Empirically calculated for the given data set


def read_wav(filename):
    """
    Read the wav file and return corresponding data
    :param filename: name of the file
    :return: return tuple containing sampling frequency and signal
    """
    return wav.read(filename)


def get_data(data_path, flatten=True, mfcc_len=39,
             class_labels=("Neutral", "Angry", "Happy", "Sad")):
    """
    Process the data for training and testing.

    Perform the following steps.
    1. Read the files and get the audio frame.
    2. Perform the test-train split

    :param class_labels: class labels that we care about.
    :param data_path: path to the dataset folder
    :param mfcc_len: Number of mfcc features to take for each frame
    :param flatten: Boolean specifying whether to flatten the data or not
    :return: 4 arrays, x_train x_test y_train y_test
    """
    data = []
    labels = []
    max_fs = 0
    cur_dir = os.getcwd()
    sys.stderr.write('curdir: %s\n' % cur_dir)
    os.chdir(data_path)
    for i, directory in enumerate(class_labels):
        sys.stderr.write("started reading folder %s\n" % directory)
        os.chdir(directory)
        for filename in os.listdir('.'):
            fs, signal = read_wav(filename)
            max_fs = max(max_fs, fs)
            s_len = len(signal)
            # pad the signals to have same size if lesser than required
            # else slice them
            if s_len < mean_signal_length:
                pad_len = mean_signal_length - s_len
                pad_rem = pad_len % 2
                pad_len //= 2
                signal = np.pad(signal, (pad_len, pad_len + pad_rem),
                                'constant', constant_values=0)
            else:
                pad_len = s_len - mean_signal_length
                pad_len //= 2
                signal = signal[pad_len:pad_len + mean_signal_length]
            mel_coefficients = mfcc(signal, fs, num_cepstral=mfcc_len)

            if flatten:
                # Flatten the data
                mel_coefficients = np.ravel(mel_coefficients)
            data.append(mel_coefficients)
            labels.append(i)
        sys.stderr.write("ended reading folder %s\n" % directory)
        os.chdir('..')
    os.chdir(cur_dir)
    x_train, x_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=0.2,
                                                        random_state=42)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(
        y_test)
