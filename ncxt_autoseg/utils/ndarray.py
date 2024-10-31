import numpy as np

from math import floor, ceil
import os


def binned_shape(shape, binning):
    retval = [s // binning for s in shape]
    if len(shape) > 3:
        retval[0] = shape[0]
    return retval


def upscaled_shape(shape, binning):
    retval = [(s // binning) * binning for s in shape]
    if len(shape) > 3:
        retval[0] = shape[0]
    return retval


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


def upscale(a, scale):
    for dim in range(a.ndim):
        a = a.repeat(scale, axis=dim)
    return a


def upscale_dims(a, dims, scale):
    for dim in range(a.ndim):
        if dim in dims:
            a = a.repeat(scale, axis=dim)
    return a


def bin_ndarray_single(ndarray, binning):
    bin_shape = [s // binning for s in ndarray.shape]
    if ndarray.ndim > 3:
        bin_shape[0] = ndarray.shape[0]
    return bin_ndarray(ndarray, bin_shape)


def labels2prob(image):
    n_labels = np.max(image) + 1
    array_label = np.zeros((n_labels, *image.shape))
    for label_index in range(n_labels):
        array_label[label_index, :] = image == label_index
    return array_label


def bin_ndarray_mode(ndarray, binning):
    prob = labels2prob(ndarray)
    bin_shape = [s // binning for s in prob.shape]
    bin_shape[0] = prob.shape[0]
    bin_prob = bin_ndarray(prob, bin_shape)
    return np.argmax(bin_prob, axis=0)

