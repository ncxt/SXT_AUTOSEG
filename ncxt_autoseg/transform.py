"""
Submodule for image transforms
Rotate
Translate

"""
from __future__ import absolute_import, division, print_function, unicode_literals

__all__ = ["rotate_complex", "rotate_freq_in_k", "rotate_freq_in_real"]

from scipy.ndimage.interpolation import rotate
import numpy as np

# from simimage import utils
# from simimage.window import window_radial


def rotate_complex(image, rot_angle, axes=None, order=1):
    """
    Extends the scipy rotation function to handle complex numbers
    """
    if len(image.shape) == 2:
        return rotate(np.real(image), rot_angle, reshape=False, order=1) + 1j * rotate(
            np.imag(image), rot_angle, reshape=False, order=order
        )

    return rotate(
        np.real(image), rot_angle, axes=axes, reshape=False, order=1
    ) + 1j * rotate(np.imag(image), rot_angle, axes=axes, reshape=False, order=order)


def rotate_freq_in_k(img, rot, axes=None):
    """
    Rotate a frequency image directly in f-space
    Parameters:
    img: image
    rot: angle of rotation (degrees)
    axes: axes of rotation, if None, assume 2D image
    """

    dims = len(img.shape)

    shifted = np.fft.fftshift(img)
    pad = [(0, (s + 1) % 2) for s in img.shape]
    print(pad)

    shifted = np.pad(shifted, pad, "constant")
    return_image = np.fft.ifftshift(rotate_complex(shifted, rot, axes=axes))

    if dims == 2:
        return return_image[0 : img.shape[0], 0 : img.shape[1]]
    if dims == 3:
        return return_image[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]]


# def rotate_freq_in_real(img, rot, window=None, axes=None):
#     """
#     Rotate a frequency image with interpolation in real space
#     Parameters:
#     img: image
#     rot: angle of rotation (degrees)
#     window: radially symmetric windowing before rotation
#     axes: axes of rotation, if None, assume 2D image

#     Returns:
#     rotated image
#     """
#     if window:
#         weights = window_radial(img.shape, window)
#         return np.fft.fftn(
#             rotate_complex(weights * np.fft.ifftn(img), rot, axes=axes))

#     return np.fft.fftn(rotate_complex(np.fft.ifftn(img), rot, axes=axes))


def translate_int(image, shifts):
    """
    Translate to nearest integer shifts
    shifts in order of dimensions

    Parameters:
    x : nd array
    shifts: tuple of shifts

    Returns:
    tranlated image

    """
    for i, shift in enumerate(shifts):
        image = np.roll(image, int(round(shift)), i)

    return image


def apply_phase_gradient(image, shifts):
    """
    N-D image phase gradients for subpixel translation

    Parameters:
    x : nd array
    shifts: tuple of shifts

    Returns:
    multiplied image
    """
    assert len(shifts) == len(image.shape), "x, shifts must have the same dimensions"

    meshgrid_args = [np.fft.fftfreq(s) for s in image.shape]
    frequencies_grid = np.meshgrid(*meshgrid_args, indexing="ij")
    k = 2 * np.pi

    for shift, frequencies in zip(shifts, frequencies_grid):
        image *= np.exp(-1j * k * (shift * frequencies))

    return image


def translate_phase(image, shifts):
    """
    N-D image translation by multiplication in phase space

    Parameters:
    x : nd array
    shifts: tuple of shifts

    Returns:
    tranlated image
    """
    assert len(shifts) == len(image.shape), "x, shifts must have the same dimensions"

    reciprocal_image = apply_phase_gradient(np.fft.fftn(image), shifts)

    return np.fft.ifftn(reciprocal_image)
