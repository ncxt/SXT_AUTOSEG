import numpy as np

from math import floor, ceil
import os


def pad_to_shape(img, shape, mode="constant", **kwargs):

    shape_diff = [a - b for a, b in zip(shape, img.shape)]
    pad = tuple((int(floor(x / 2)), int(ceil(x / 2))) for x in shape_diff)

    return np.pad(img, pad, mode, **kwargs)


def ensure_dir(file_path):
    """Ensure the folder exists for file path
    
    Arguments:
        file_path {string} -- full path to file
    """

    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_path(file_path):
    """Ensure the folder exists for file path

    Arguments:
        file_path {string} -- full path to file
    """

    ensure_dir(os.path.dirname(file_path))
