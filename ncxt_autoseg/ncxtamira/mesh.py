"""
Stuff to handle the parsing of the Amira mesh header
"""

import operator
import re
import zlib
from functools import reduce
import numpy as np
from . import LOGGER
from .utils import Matcher


class HeaderDict:
    """
    Parse mesh file header into a nested dictionary
    """

    def __init__(self, inputstring):
        self._inputstring = inputstring
        self._items = []

    def parse(self):
        """ parse the header string """

        depth = -1
        start_pos = []
        stop_pos = []
        descriptions = []
        previous_stop = -1
        char = None
        prev_char = None

        def is_item():
            return depth == -1 and prev_char == "," and char == "\n"

        for pos, char in enumerate(self._inputstring):
            if char == "{":
                depth += 1
                if depth == 0:
                    start_pos.append(pos)
            if char == "}":
                if depth == 0:
                    stop_pos.append(pos)
                    descriptions.append(
                        self._inputstring[previous_stop + 1 : start_pos[-1]].strip()
                    )
                    previous_stop = pos

                depth -= 1
            if is_item():
                item_string = self._inputstring[previous_stop + 1 : pos - 1].strip()
                if item_string:
                    LOGGER.debug("    Found root level item '%s'", item_string)
                    self._items.append(item_string)
                previous_stop = pos
            prev_char = char

        resdict = dict()

        for item in self._items:
            item_split = item.split()
            key, val = item_split[0], " ".join(item_split[1:])
            resdict[key] = val

        for desc, start, stop in zip(descriptions, start_pos, stop_pos):

            nextstr = " " + self._inputstring[start + 1 : stop].rstrip()
            # make sure itemlist ends with a comma
            if nextstr[-1] != ",":
                nextstr += ","

            LOGGER.debug("Found Description %s", desc)
            resdict[desc] = HeaderDict(nextstr + "\n").parse()

        return resdict


def rle_decompress(buf):
    """ rle decoding, from https://github.com/strawlab/py_amira_file_reader """
    result = []
    idx = 0
    bufferlength = len(buf)
    while idx < bufferlength:
        control_byte = ord(buf[idx : idx + 1])
        idx += 1
        if control_byte == 0:
            break
        elif control_byte <= 127:
            repeats = control_byte
            new_byte = buf[idx : idx + 1]
            idx += 1
            result.append(new_byte * repeats)
        else:
            num_bytes = control_byte - 128
            new_bytes = buf[idx : idx + num_bytes]
            idx += num_bytes
            result.append(new_bytes)
    final_result = b"".join(result)
    return final_result


class AmiraMesh:
    """ AmiraMesh, only works for volumetric data"""

    def __init__(self, filepath):
        self.re_bytedata_info = re.compile(r"^@(\d+)(\((\w+),(\d+)\))?$")
        self.is_bytedata_info = Matcher(self.re_bytedata_info)
        self.filepath = filepath
        self.defines = dict()
        self.lattices = dict()

        self._header = []
        self.buffer = []
        self.arrays = dict()
        self.parameters = dict()

        self.supported = [
            "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1",
            "# AmiraMesh BINARY-LITTLE-ENDIAN 3.0",
        ]

    def is_supported(self):
        return self.meshtype in self.supported

    @property
    def meshtype(self):
        with open(self.filepath, mode="rb") as fileobj:
            binaryline = fileobj.readline()
            return binaryline.decode("utf-8").strip()

    def loadmesh(self, headeronly=False):

        msg = f"loadmesh, headeronly {headeronly}"
        LOGGER.info(msg)

        with open(self.filepath, mode="rb") as fileobj:
            binaryline = fileobj.readline()
            assert (
                binaryline.decode("utf-8").strip() in self.supported
            ), f"File format \n{binaryline}\nnot supported"
            count = 0
            while binaryline != b"# Data section follows\n" and binaryline:
                try:
                    line = binaryline.decode("utf-8")
                except UnicodeDecodeError:
                    LOGGER.debug("Cannot decode line\n %s", binaryline)
                    line = binaryline.decode("utf-8", "ignore")
                    LOGGER.debug("Decoded with 'ignore'\n %s", line)

                if line[0] == "#" or line == "\n":
                    pass
                elif self.add_define(line):
                    pass
                elif self.add_lattice(line):
                    pass
                else:
                    self._header.append(line)
                    count += 1

                binaryline = fileobj.readline()
            header = HeaderDict("".join(self._header)).parse()
            self.parameters = header["Parameters"]
            if headeronly:
                return

            self.buffer = fileobj.read()
            lattice_shape = [int(value) for value in self.defines["Lattice"]]

            for key, value in self.lattices.items():
                dtype = value["dtype"]
                matchobj = self.re_bytedata_info.match(value["dataargs"])
                bytedata_id, enc_size, encoding, size = matchobj.groups()

                LOGGER.info("Loading lattice %s", key)
                LOGGER.info(
                    "bytedata_id %s, enc_size %s, encoding %s, size %s",
                    bytedata_id,
                    enc_size,
                    encoding,
                    size,
                )

                # goto next datamarker
                while self.buffer[:1] != b"@":
                    # print(f"Current buffer {self.buffer[:1]}, removing.")
                    self.buffer = self.buffer[1:]

                if size is None:
                    LOGGER.debug("dtype is %s", dtype)
                    size = reduce(operator.mul, lattice_shape, 1)
                    # print(v["dtype"])
                    if dtype == "float":
                        size *= 4
                    if dtype == "ushort":
                        size *= 2
                else:
                    size = int(size)

                token, self.buffer = self.buffer[:3], self.buffer[3:]
                LOGGER.debug("token is %s", token)
                LOGGER.debug("Buffer size %d size %d", len(self.buffer), size)
                binary_buf, self.buffer = self.buffer[:size], self.buffer[size:]

                if encoding == "raw":
                    raw_buf = binary_buf
                elif encoding == "HxZip":
                    raw_buf = zlib.decompress(binary_buf)
                elif encoding == "HxByteRLE":
                    raw_buf = rle_decompress(binary_buf)
                elif encoding is None:
                    raw_buf = binary_buf
                else:
                    raise ValueError("unknown encoding %r" % encoding)
                nptype = {"float": np.float32, "byte": np.uint8, "ushort": np.ushort}
                arr = np.frombuffer(raw_buf, dtype=nptype[dtype])
                arr.shape = lattice_shape[2], lattice_shape[1], lattice_shape[0]
                # arr = np.swapaxes(arr, 0, 2)

                self.arrays[bytedata_id] = np.array(arr, dtype=arr.dtype)
        return self

    def add_lattice(self, line):
        """ check if line is a lattice and add to dictionary of lattices"""
        if "Lattice" in line:
            split = line.split()
            assert split[-1][0] == "@", f"first char of {split[-1]} should be @"
            lattice_id = split[-1][1]
            assert self.is_bytedata_info(split[5])
            self.lattices[lattice_id] = {
                "dtype": split[2],
                "type": split[3],
                "dataargs": split[5],
            }
            LOGGER.info("Added lattice %s", self.lattices[lattice_id])
            return True
        return False

    def add_define(self, line):
        """ check if line is a define argument add to dictionary of defines"""
        if "define" in line:
            split = line.split()
            self.defines[split[1]] = split[2:]
            LOGGER.info("Added define %s with arguments %s", split[1], split[2:])
            return True
        return False

    @property
    def key(self):
        """ return dictionary containing material key,value pairs"""
        if not self.parameters:
            self.loadmesh(headeronly=True)

        if "Materials" not in self.parameters:
            return None

        return {
            name: name_enum
            for name_enum, name in enumerate(self.parameters["Materials"].keys())
        }

    @property
    def arr(self):
        """ return the first lattice as ndarray"""
        if not self.arrays:
            self.loadmesh(headeronly=False)
        return self.arrays["1"]
