import configparser
import csv
import pickle


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


def get_config():
    """
    create a config object
    :return:
    """
    config = configparser.ConfigParser()
    config.read('configurations.ini')
    return config


def save_as_pickle(obj, filename):
    """
    save an object in a pickle file dump
    :param obj: object to dump
    :param filename: target file
    :return:
    """
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    """
    load an object from a given pickle file
    :param filename: source file
    :return: loaded object
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)
