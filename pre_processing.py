import csv
from os import listdir
from os.path import isfile, join
import numpy as np
from PIL import Image

from utils import get_config, save_as_pickle, load_pickle


def transform_images(dir_path):
    """
    loads images from a given directory and transforms them into a numpy array
    shape -> number of images x pixel length x pixel width x 3 (RGB values)
    :param dir_path: path of directory
    :return: images as four dimensional numpy array
    """
    image_paths = [join(dir_path, file_path) for file_path in listdir(dir_path) if isfile(join(dir_path, file_path))]
    return np.asarray([np.asarray(Image.open(image_path), dtype='int32') for image_path in image_paths])


def get_label_dict():
    """
    returns a dictionary of labels and their numeric values which are used internally
    :return: dict of labels
    """
    config = get_config()
    with open(config['image_paths']['train_labels'], newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',')
        labels = [row['label'] for row in reader]
        return {label: idx for idx, label in enumerate(sorted(list(set(labels))))}


def load_train_labels():
    """
    loads the train labels as a numpy array
    :return:
    """
    config = get_config()
    with open(config['image_paths']['train_labels'], newline='') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',',)
        labels = np.asarray([row['label'] for row in reader])
        label_dict = get_label_dict()
        return np.asarray([label_dict[label_string] for label_string in labels])


if __name__ == '__main__':
    # config = get_config()
    # train_images_np = transform_images(config['image_paths']['train_images_raw'])
    # save_as_pickle(train_images_np, config['image_paths']['train_images_pickle'])
    # test_images_np = transform_images(config['image_paths']['test_images_raw'])
    # save_as_pickle(test_images_np, config['image_paths']['test_images_pickle'])
    load_train_labels()

    # loaded = load_pickle( config['image_paths']['train_images_pickle'] )
    # print(loaded.shape)
