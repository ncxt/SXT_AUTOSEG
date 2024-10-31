import numpy as np
import os
from functools import reduce
from math import ceil
from _blocks import bin_tensor


# from https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
def bin_ndarray(ndarray, new_shape, operation="mean"):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ["sum", "mean"]:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape, new_shape))
    compression_pairs = [(d, c // d) for d, c in zip(new_shape, ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1 * (i + 1))
    return ndarray


import pdb


def bin_label(image, binning):

    labels = np.unique(image)
    image_onehot = np.zeros((len(labels), *image.shape), dtype=image.dtype)

    for i, label in enumerate(labels):
        image_onehot[i, :] = image == label

    binned_onehot = bin_tensor(image_onehot, binning)
    labelindex = np.argmax(binned_onehot, 0)
    retval = np.zeros(labelindex.shape, dtype=image.dtype)

    for i, label in enumerate(labels):
        retval[labelindex == i] = label

    return retval


def upscale(a, scale):
    for dim in range(a.ndim):
        a = a.repeat(scale, axis=dim)
    return a


def upscale_dims(a, dims, scale):
    for dim in range(a.ndim):
        if dim in dims:
            a = a.repeat(scale, axis=dim)
    return a


def window(shape, func, **kwargs):
    vs = [func(l, **kwargs) for l in shape]
    return reduce(np.multiply, np.ix_(*vs))
