"""
helper functions to display volumes using matplotlib
"""
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import itertools


def getslice(data, dim, slicenr=None):
    """Get slice of a n-dimensional array

    Arguments:
        data {ndarray} -- Input array.
        dim {int} -- dimension of the slicing

    Keyword Arguments:
        slicenr {int} -- Slice number, if not assigned gets the middle

    Returns:
        [ndarray] -- Output array
    """

    if slicenr is None:
        slicenr = int(data.shape[dim] / 2)
    assert -1 < slicenr < data.shape[dim], f"Index {slicenr} is out of range"

    return np.take(data, slicenr, axis=dim)


def volshow(data, slices=(None, None, None)):
    """Show preview of volume

    Arguments:
        data {ndarray} -- Input array.

    Keyword Arguments:
        slices {tuple} -- Shown slices, default is middle (default: {(-1, -1, -1)})
    """

    ndim = data.ndim
    assert ndim == 3, "Volume must be a 3d ndarray"

    _ = plt.figure(figsize=(13, 5))
    axes = [plt.subplot(gsi) for gsi in gridspec.GridSpec(1, ndim)]
    images = [getslice(data, d, s) for d, s in zip(range(ndim), slices)]

    for axis, image in zip(axes, images):
        axis.imshow(image)


def normalize_ndarray(img, percentiles=(5, 95)):
    """Normalize ndarray to percentiles"""

    lim = np.percentile(img, percentiles[0]), np.percentile(img, percentiles[1])
    retval = (img - lim[0]) / (lim[1] - lim[0])
    retval[retval < 0] = 0.0
    retval[retval > 1] = 1.0

    return 1.0 - retval


def hex2rgb(hexstring):
    """ Convert hex code to RGB value"""
    return tuple([int(hexstring.lstrip("#")[i : i + 2], 16) / 256.0 for i in (0, 2, 4)])


def make_overlay(lac, label, void=0, saturation=0.7, blend=0.6, colors=None):
    """ Make overlay image of lac and label """
    if not isinstance(void, (list, tuple)):
        void = [void]

    rgb_colors = [hex2rgb(h) for h in colors]

    img = normalize_ndarray(lac)
    # Construct a colour image for the labels
    color_mask = np.dstack([img] * 3)
    # replace labels with color
    labelindecies = [i for i in range(np.max(label) + 1) if i not in void]
    for i in labelindecies:
        color_mask[label == i] = rgb_colors[i % len(colors)]

    # Convert the color mask to Hue Saturation Value (HSV)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # decrease the saturation by saturation
    # blend the value with LAC
    color_mask_hsv[..., 1] *= saturation
    color_mask_hsv[..., 2] = (1.0 - blend) * img + color_mask_hsv[..., 2] * blend

    return color.hsv2rgb(color_mask_hsv)


def get_middle_slices(image):
    """ Return slices for all dimensions """
    return [getslice(image, d) for d in range(image.ndim)]


def plot_data(lac, label, name, key):
    """ Plot overlay of data and labels"""

    _ = plt.figure(figsize=(13, 5))
    axes = [plt.subplot(gsi) for gsi in gridspec.GridSpec(1, 4)]

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colorgen = itertools.cycle(prop_cycle.by_key()["color"])
    colors = [next(colorgen) for _ in range(max(key.values()) + 1)]

    void_idx = [v for k, v in key.items() if "void" in k]
    overlay_kwargs = {
        "void": void_idx,
        "saturation": 0.6,
        "blend": 0.6,
        "colors": colors,
    }

    slices_lac = get_middle_slices(lac)
    slices_label = get_middle_slices(label)

    overlay_images = [
        make_overlay(a, b, **overlay_kwargs) for a, b, in zip(slices_lac, slices_label)
    ]

    for axis, image in zip(axes, overlay_images):
        axis.imshow(image)

    labels = list(key.keys())

    im_colors = set()
    for image in slices_label:
        im_colors |= set(np.unique(image))
    void_idx = [v for k, v in key.items() if "void" in k]
    im_colors = [i for i in im_colors if i not in void_idx]

    legend_elements = [
        patches.Patch(
            facecolor=colors[key[label] % len(colors)], edgecolor=None, label=label
        )
        for i, label in enumerate(labels)
        if key[label] in im_colors
    ]

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            # pass
            spine.set_visible(False)

    axes[3].legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(0.9, 0.9),
        prop={"size": 12},
    )

    plt.subplots_adjust(
        left=0.01, right=0.99, hspace=0.05, wspace=0.01, top=0.9, bottom=0.05
    )

    plt.suptitle(name)

    return
