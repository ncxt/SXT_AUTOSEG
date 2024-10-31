import struct
import numpy as np


def read_mrc(filename):

    input_image = open(filename, "rb")

    num_ints = 56
    sizeof_int = 4
    nlines_header = 10
    num_chars = 80

    head1 = input_image.read(num_ints * sizeof_int)  # read 56 long ints
    head2 = input_image.read(nlines_header * num_chars)  # read 10 lines of 80 chars
    byte_pattern = (
        "=" + "l" * num_ints
    )  #'=' required to get machine independent standard size
    dim = struct.unpack(byte_pattern, head1)[:3][::-1]
    imagetype = struct.unpack(byte_pattern, head1)[
        3
    ]  # 0: 8-bit signed, 1:16-bit signed, 2: 32-bit float, 6: unsigned 16-bit (non-std)
    # print("Imagetype = {}".format(imagetype ))

    if imagetype == 0:
        imtype = "b"
    elif imagetype == 1:
        imtype = "h"
    elif imagetype == 2:
        imtype = "f4"
    elif imagetype == 6:
        imtype = "H"
    else:
        imtype = "H"
        print(f"Unknown imagetype {imagetype}")
        type = "unknown"  # should put a fail here

    num_voxels = dim[0] * dim[1] * dim[2]
    image_data = np.fromfile(file=input_image, dtype=imtype, count=num_voxels).reshape(
        dim
    )
    input_image.close()

    return image_data


def write_mrc(filename, im, num_ints=56, sizeof_int=4, num_chars=800):

    type_modes = {"b": 0, "h": 1, "f": 2, "H": 6}

    mode = type_modes[im.dtype.char]
    dims = im.shape[::-1]
    header1 = struct.pack(
        "=" + "l" * num_ints, *(dims + (mode,) + (0,) * (num_ints - len(dims) - 1))
    )
    header2 = struct.pack("=" + "c" * num_chars, *(b" ",) * num_chars)

    with open(filename, "wb") as f:
        f.write(header1)
        f.write(header2)
        f.write(im.tobytes())
