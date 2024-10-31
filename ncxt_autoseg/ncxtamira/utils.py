"""
misc utilities for ncxtamira
"""
import os


class Matcher:
    """ convenience class for regexp matching """

    def __init__(self, regexp):
        self.regexp = regexp

    def __call__(self, buf):
        matchobj = self.regexp.match(buf)
        return matchobj is not None


def ensure_dir(directory):
    """Ensure the folder exists

    Arguments:
        file_path {string} -- full path to file
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_path(file_path):
    """Ensure the folder exists for file path

    Arguments:
        file_path {string} -- full path to file
    """

    ensure_dir(os.path.dirname(file_path))
