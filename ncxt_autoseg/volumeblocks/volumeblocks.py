import numpy as np

import _blocks
from _blocks import bin_volume, bin_tensor
from _blocks import upscale_volume, upscale_tensor

from .blocks import split_volume, split_tensor, combine_volume, combine_tensor
from .utils import bin_ndarray, upscale, upscale_dims
from .utils import bin_label


def split_py(image, block_shape, binning=1, sampling=1.0):
    if image.ndim == 3:
        blocks = split_volume(image, block_shape, sampling)
        block_shape_bin = [s // binning for s in block_shape]

    if image.ndim == 4:
        blocks = split_tensor(image, block_shape, sampling)
        block_shape_bin = [image.shape[0]] + [s // binning for s in block_shape]

    if binning > 1:
        blocks = [bin_ndarray(b, block_shape_bin) for b in blocks]

    return blocks


def fuse_py(blocks, shape, binning=1, sampling=1.0, windowfunc=None):
    if binning > 1:
        if blocks[0].ndim == 3:
            blocks = [upscale(b, binning) for b in blocks]

        if blocks[0].ndim == 4:
            blocks = [upscale_dims(b, [1, 2, 3], binning) for b in blocks]

    if blocks[0].ndim == 3:
        return combine_volume(blocks, shape, sampling, windowfunc=windowfunc)
    if blocks[0].ndim == 4:
        return combine_tensor(blocks, shape, sampling, windowfunc=windowfunc)


def split(image, block_shape, binning=1, sampling=1.0):
    """Split wrapper to split_volume and split_tensor for splitting up volumes
    to subvolumes of shape block_shape.

    Arguments:
        image {ndarray} --  image
        block_shape {tuple} -- shape of the subvolumes

    Keyword Arguments:
        binning {int} -- Optional binning of the subvolumes.
        This enables the original data to  be processed as a binned versions
        without size restrictions of the riginal shape ( s % bin == 0) (default: {1})
        sampling {float} -- (over)Sampling of the data.
            1.0 produces the minimum number of blocks to cover the whole volume. (default: {1.0})
    Returns:
        [type] -- [description]
    """

    if image.ndim == 3:
        blocks = split_volume(image, block_shape, sampling)
        binfunc = bin_volume
    if image.ndim == 4:
        blocks = split_tensor(image, block_shape, sampling)
        binfunc = bin_tensor

    if binning > 1:
        blocks = [binfunc(b, binning) for b in blocks]

    return blocks


def split_label(image, block_shape, binning=1, sampling=1.0):
    """Split wrapper to split_volume and split_tensor for splitting up 
        label images choosing the mode value.

    Arguments:
        image {ndarray} --  image
        block_shape {tuple} -- shape of the subvolumes

    Keyword Arguments:
        binning {int} -- Optional binning of the subvolumes.
        This enables the original data to  be processed as a binned versions
        without size restrictions of the riginal shape ( s % bin == 0) (default: {1})
        sampling {float} -- (over)Sampling of the data.
            1.0 produces the minimum number of blocks to cover the whole volume. (default: {1.0})
        mode {Bool} If mode is True, the binning is done by mode value (label images)
    Returns:
        [type] -- [description]
    """

    if image.ndim == 3:
        blocks = split_volume(image, block_shape, sampling)
        binfunc = bin_label
    if image.ndim == 4:
        assert False, "Not implemented yet"

    if binning > 1:
        blocks = [binfunc(b, binning) for b in blocks]

    return blocks


from scipy.ndimage import gaussian_filter


def fuse(blocks, shape, binning=1, sampling=1.0, smooth=True, windowfunc=None):
    """Fuse wrapper to combine_volume and combine_tensor for fusing processed blocks
    to the original shape

    Arguments:
        blocks {list} --  list of subvolumes
        shape {tuple} -- Original shape of the data

    Keyword Arguments:
        binning {int} -- Optional binning of the subvolumes.
        This enables the original data to  be processed as a binned version
        without size restrictions of the riginal shape ( s % bin == 0) (default: {1})
        sampling {float} -- Sampling of the data used to produce the blocks
        (default: {1.0})

    Returns:
        [type] -- [description]
    """
    if binning > 1:
        if blocks[0].ndim == 3:
            blocks = [upscale_volume(b, binning) for b in blocks]
        if blocks[0].ndim == 4:
            blocks = [upscale_tensor(b, binning) for b in blocks]

    if blocks[0].ndim == 3:
        retval = combine_volume(blocks, shape, sampling, windowfunc=windowfunc)
    if blocks[0].ndim == 4:
        retval = combine_tensor(blocks, shape, sampling, windowfunc=windowfunc)

    if binning > 1 and smooth:
        assert retval.dtype != np.integer, "Smoothing is not valid for label images"
        sigma = binning / 2.355
        if retval.ndim == 4:
            sigma = [0, sigma, sigma, sigma]

        retval = gaussian_filter(retval, sigma)

    return retval


def random_blocks(image, block_shape, max_patches, binning=1, random_state=None):
    """Generate random pathces drawn from image
    Args:
        image (ndarray): input image
        block_shape (tuple): 3D shape of patches
        max_patches (int): number of patches
        binning (int): binning of the pathces
        random_state ({None, int, array_like}, optional):
            Random seed used to initialize the pseudo-random number generator.
    Returns:
        [ndarray]: array of patches, patches along dim 0.
    """

    random_generator = np.random.RandomState(random_state)

    if image.ndim == 3:
        binfunc = bin_volume
    if image.ndim == 4:
        block_shape = (image.shape[0], *block_shape)
        binfunc = bin_tensor

    indecies = [np.arange(1 + l - b) for l, b in zip(image.shape, block_shape)]
    samples = [random_generator.choice(x, size=max_patches) for x in indecies]

    def get_view(i):
        slices = [
            slice(sample[i], sample[i] + b) for sample, b in zip(samples, block_shape)
        ]
        return image[tuple(slices)]

    blocks = [get_view(i) for i in range(max_patches)]

    if binning > 1:
        blocks = [binfunc(b, binning) for b in blocks]

    return blocks


def random_blocks_label(image, block_shape, max_patches, binning=1, random_state=None):
    """Generate random pathces drawn from image
    Args:
        image (ndarray): input image
        block_shape (tuple): 3D shape of patches
        max_patches (int): number of patches
        binning (int): binning of the pathces
        random_state ({None, int, array_like}, optional):
            Random seed used to initialize the pseudo-random number generator.
    Returns:
        [ndarray]: array of patches, patches along dim 0.
    """

    random_generator = np.random.RandomState(random_state)

    if image.ndim == 3:
        binfunc = bin_label
    if image.ndim == 4:
        assert False, "Not implemented yet"

    indecies = [np.arange(1 + l - b) for l, b in zip(image.shape, block_shape)]
    samples = [random_generator.choice(x, size=max_patches) for x in indecies]

    def get_view(i):
        slices = [
            slice(sample[i], sample[i] + b) for sample, b in zip(samples, block_shape)
        ]
        return image[tuple(slices)]

    blocks = [get_view(i) for i in range(max_patches)]

    if binning > 1:
        blocks = [binfunc(b, binning) for b in blocks]
    return blocks
