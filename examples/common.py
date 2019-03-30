import numpy as np
from sklearn.model_selection import train_test_split

from speechemotionrecognition.utilities import get_data, \
    get_feature_vector_from_mfcc

_DATA_PATH = '../dataset'
_CLASS_LABELS = ("Neutral", "Angry", "Happy", "Sad")


def extract_data(flatten):
    data, labels = get_data(_DATA_PATH, class_labels=_CLASS_LABELS,
                            flatten=flatten)
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=42)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(
        y_test), len(_CLASS_LABELS)


def get_feature_vector(file_path, flatten):
    return get_feature_vector_from_mfcc(file_path, flatten, mfcc_len=39)
