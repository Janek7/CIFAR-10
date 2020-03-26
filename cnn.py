import random

import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import mnist
from os import path
import logging
import matplotlib.pyplot as plt

from pre_processing import load_train_labels, get_label_dict
from utils import load_pickle, save_as_pickle, get_config

config = get_config()


def load_data():
    """
    load data and split into test and training data
    :return:
    """
    X = load_pickle(config['image_paths']['train_images_pickle'])
    y = load_train_labels()
    y = to_categorical(y)
    test_indices = np.random.choice(len(X), int(len(X) * float(config['model']['test_size'])), replace=False)
    X_train = np.asarray([e for idx, e in enumerate(X) if idx not in test_indices])
    X_test = np.asarray([e for idx, e in enumerate(X) if idx in test_indices])
    y_train = np.asarray([e for idx, e in enumerate(y) if idx not in test_indices])
    y_test = np.asarray([e for idx, e in enumerate(y) if idx in test_indices])
    # X_train = X[10000:]
    # X_test = X[:10000]
    # y_train = y[10000:]
    # y_test = y[:10000]
    # for i in range(10):
    #     print(y_test[i])
    #     plt.imshow(X_test[i])
    #     plt.show()
    return X_train, y_train, X_test, y_test


def load_data_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 28, 28, 1)
    X_test = X_test.reshape(10000, 28, 28, 1)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return X_train, y_train, X_test, y_test


def get_model(existing_model_path=None):
    """
    loads model from an existing pickle dump
    if no file is given, the dump is invalid or the file does not exists create a new model
    :param existing_model_path: path of existing pickle dump
    :return: model object
    """
    model = None
    if existing_model_path is not None and path.isfile(existing_model_path):
        model = load_pickle(existing_model_path)
        logging.info('loaded model from ' + existing_model_path)
    if not isinstance(model, Sequential):
        logging.info('model is no valid model object')
        model = Sequential()
        logging.info('created new model')
    return model


def train(model, X_train, y_train, X_test, y_test):
    """
    build and train model
    :param model: target model object
    :param X_train: training data
    :param y_train: training labels
    :param X_test: test data
    :param y_test: test labels
    :return: trained model
    """
    # add layers
    nodes = 64
    kernel = 3
    model.add(Conv2D(64, kernel_size=kernel, activation='relu', input_shape=X_train[0].shape))
    model.add(Conv2D(nodes, kernel_size=kernel, activation='relu'))
    model.add(Conv2D(nodes, kernel_size=kernel, activation='relu'))
    model.add(Conv2D(nodes, kernel_size=kernel, activation='relu'))
    model.add(Flatten())
    model.add(Dense(len(get_label_dict()), activation='softmax'))
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
    return model


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    X_train, y_train, X_test, y_test = load_data()
    model = get_model(config['model']['model_path'])
    train(model, X_train, y_train, X_test, y_test)
    # save_as_pickle(model, config['model']['model_path'])
