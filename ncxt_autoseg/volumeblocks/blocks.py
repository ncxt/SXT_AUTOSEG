from math import ceil

import numpy as np
from .utils import bin_ndarray, window


def divide(length, block_size, n_blocks):
    """ Divide a 1d array into evenly spaced
        blocks of size block_size.

    Arguments:
        length {[type]} -- Length of the space
        block_size {[type]} -- Size of the blocks
        n_blocks {[type]} -- Number of blocks

    Returns:
        [type] -- Starting index of the blocks
    """

    x0, xend = 0, length - block_size
    return np.round(np.linspace(x0, xend, n_blocks)).astype(int)


def num_blocks(lenght, size, sampling=1.0):
    """Number of blocks needed to cover the lenght

    Arguments:
        lenght {int} -- Length of the array
        size {int} -- blocks size
        sampling {float} -- sampling of the length
            1.0 gives the minimum required blocks to cover the array (default: {1.0})
    """
    return int(ceil(sampling * lenght / size))


def split_volume(volume, block_shape, sampling=1.0):
    """Split the volume into subvolumes of size block_shape

    Arguments:
        volume {ndarray} -- 3d volume
        block_shape {tuple} -- shape of the blocks

    Keyword Arguments:
        sampling {float} -- (over)Sampling of the data.
        1.0 produces the minimum number of blocks to cover the whole volume(default: {1.0})
    Returns:
        [list] -- List of subvolume Slice views of the original data
    """

    limits = [
        divide(length, size, num_blocks(length, size, sampling))
        for length, size in zip(volume.shape, block_shape)
    ]
    subblocks = []
    for i0 in limits[0]:
        slice_i = slice(i0, i0 + block_shape[0])
        for j0 in limits[1]:
            slice_j = slice(j0, j0 + block_shape[1])
            for k0 in limits[2]:
                slice_k = slice(k0, k0 + block_shape[2])
                subblocks.append(volume[slice_i, slice_j, slice_k])

    return subblocks


def split_tensor(tensor, block_shape, sampling=1.0):
    """Split the volume into subvolumes of size block_shape
        the first dimension is assumed a channel data

    Arguments:
        volume {ndarray} -- (1+3)d volume
        block_shape {tuple} -- shape of the blocks

    Keyword Arguments:
        sampling {float} -- (over)Sampling of the data.
        1.0 produces the minimum number of blocks to cover the whole volume(default: {1.0})
    Returns:
        [list] -- List of subvolume Slice views of the original data
    """
    _, *volume_shape = tensor.shape

    limits = [
        divide(length, size, num_blocks(length, size, sampling))
        for length, size in zip(volume_shape, block_shape)
    ]

    subblocks = []
    for i0 in limits[0]:
        slice_i = slice(i0, i0 + block_shape[0])
        for j0 in limits[1]:
            slice_j = slice(j0, j0 + block_shape[1])
            for k0 in limits[2]:
                slice_k = slice(k0, k0 + block_shape[2])
                subblocks.append(tensor[:, slice_i, slice_j, slice_k])

    return subblocks


def combine_volume(blocklist, shape, sampling=1.0, windowfunc=None):
    """Combine subblocks to one image

    Arguments:
        blocklist {list} -- list of subvolumes
        shape {tuple} -- Original shape of volume

    Keyword Arguments:
        sampling {float} -- Sampling used to produce blocks (default: {1.0})
        windowfunc {function} -- If defined, applies a windowing on the data for smoother blending (default: {None})

    Returns:
        [ndarray] -- Fused volume
    """

    assert len(shape) == 3, f"Array {shape} must be a 3d volume"
    assert blocklist[0].ndim == 3, "Blocks must 3d subvolumes"

    block_shape = blocklist[0].shape
    limits = [
        divide(length, size, num_blocks(length, size, sampling))
        for length, size in zip(shape, block_shape)
    ]

    image = np.zeros(shape, dtype="float32")
    image_n = np.zeros(shape, dtype="float32")

    if windowfunc is None:
        index = 0
        for i0 in limits[0]:
            slice_i = slice(i0, i0 + block_shape[0])
            for j0 in limits[1]:
                slice_j = slice(j0, j0 + block_shape[1])
                for k0 in limits[2]:
                    slice_k = slice(k0, k0 + block_shape[2])

                    image[slice_i, slice_j, slice_k] += blocklist[index]
                    image_n[slice_i, slice_j, slice_k] += 1
                    index += 1
    else:
        window_block = 0.01 + window(block_shape, windowfunc)
        index = 0
        for i0 in limits[0]:
            slice_i = slice(i0, i0 + block_shape[0])
            for j0 in limits[1]:
                slice_j = slice(j0, j0 + block_shape[1])
                for k0 in limits[2]:
                    slice_k = slice(k0, k0 + block_shape[2])

                    image[slice_i, slice_j, slice_k] += window_block * blocklist[index]
                    image_n[slice_i, slice_j, slice_k] += window_block
                    index += 1

    return image / image_n


def combine_tensor(blocklist, tensorshape, sampling=1.0, windowfunc=None):
    """Combine subblocks to one image
        the first dimension is assumed a channel data

    Arguments:
        blocklist {list} -- list of subvolumes
        tensorshape {tuple} -- Original shape of the tensor

    Keyword Arguments:
        sampling {float} -- Sampling used to produce blocks (default: {1.0})
        windowfunc {function} -- If defined, applies a windowing on the data
                                for smoother blending (default: {None})

    Returns:
        [ndarray] -- Fused volume
    """

    assert len(tensorshape) == 4, "Array must be a (1+3)d tensor"
    assert blocklist[0].ndim == 4, "Blocks must (1+3)d subvolumes"
    assert blocklist[0].shape[0] == tensorshape[0], "Tensor dimension doesn't match"

    volumeshape = list(tensorshape[1:])
    block_shape = blocklist[0].shape[-3:]
    print(f"volshape {volumeshape} block shape {block_shape}")
    limits = [
        divide(length, size, num_blocks(length, size, sampling))
        for length, size in zip(volumeshape, block_shape)
    ]

    image = np.zeros(tensorshape, dtype="float32")
    image_n = np.zeros(volumeshape, dtype="float32")

    if windowfunc is None:
        index = 0
        for i0 in limits[0]:
            slice_i = slice(i0, i0 + block_shape[0])
            for j0 in limits[1]:
                slice_j = slice(j0, j0 + block_shape[1])
                for k0 in limits[2]:
                    slice_k = slice(k0, k0 + block_shape[2])
                    image[:, slice_i, slice_j, slice_k] += blocklist[index]
                    image_n[slice_i, slice_j, slice_k] += 1
                    index += 1
    else:
        window_block = 0.01 + window(block_shape, windowfunc)
        index = 0
        print(f"window_bloc {window_block.shape}")
        print(f"blocklist {blocklist[0].shape}")
        for i0 in limits[0]:
            slice_i = slice(i0, i0 + block_shape[0])
            for j0 in limits[1]:
                slice_j = slice(j0, j0 + block_shape[1])
                for k0 in limits[2]:
                    slice_k = slice(k0, k0 + block_shape[2])
                    image[:, slice_i, slice_j, slice_k] += (
                        window_block * blocklist[index]
                    )
                    image_n[slice_i, slice_j, slice_k] += window_block
                    index += 1

    return image / image_n
