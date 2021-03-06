"""Loads adult and saves it in multiple small splits. Ready to train from each of the different clients."""
import os
import argparse
import pickle
import numpy as np
import tensorflow as tf


def parse_args():
    """Parses the command line arguments and returns a dict"""
    parser = argparse.ArgumentParser()
    parser.add_argument('datasize', type=int)
    parser.add_argument('datacount', type=int)
    return parser.parse_args()

def load_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def generate_splits(arg_dict):
    """Returns the splits acording to the specified arguments."""
    size = arg_dict.datasize
    count = arg_dict.datacount
    samples = size * count

    x, y = load_data('datasets/extended/train/split1/train_dataset.pickle')

    splits = list()
    for i in range(count):
        pair = x[i * size:(i + 1) * size], y[i * size:(i + 1) * size]
        splits.append(pair)

    return splits

def save_splits(splits):
    """Pickles the splits."""
    for index, split in enumerate(splits):
        save_str = 'datasplit%04d.pickle' % index
        save_path = os.path.join('datasets/extended/train/split3', save_str)
        with open(save_path, 'wb') as f:
            pickle.dump(split, f)


if __name__ == "__main__":
    arg_dict = parse_args()
    splits = generate_splits(arg_dict)
    save_splits(splits)