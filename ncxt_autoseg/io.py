from pathlib import Path
from .read_write_mrc import read_mrc, write_mrc
from .ncxtamira.project import AmiraProject
import struct
import imageio


def load(filepath):
    filepath = Path(filepath)

    if filepath.suffix == ".hx":
        # amira project
        return AmiraProject(filepath)

    if filepath.suffix in [".mrc", ".rec", ".st"]:
        return read_mrc(filepath)

    return imageio.volread(filepath)

