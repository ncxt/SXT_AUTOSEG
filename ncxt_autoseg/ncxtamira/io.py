"""[summary]
Loaders for 3D Amira Mesh files
"""

from .mesh import AmiraMesh
from .project import AmiraProject


def loadfloat(filepath):
    """Load a float valued amira mesh file

    Arguments:
        filepath {string} -- File path for mesh

    Returns:
        [ndarray] -- float32 valued ndarray
    """

    mesh = AmiraMesh(filepath)
    return mesh.arr


def loadlabel(filepath):
    """Load a label file

    Arguments:
        filepath {string} -- File path for mesh

    Returns:
        [ndarray] -- uint8 byte valued ndarray
        [Dict] -- Dectionary containing the material key-value pairs
    """

    mesh = AmiraMesh(filepath).loadmesh(headeronly=False)
    return mesh.arr, mesh.key


def loadproject(filepath):
    """Load an AMIRA .hx pack-and-go project

    Arguments:
        filepath {string} -- File path for project

    Returns:
        [AmiraProject] -- AmiraProject containing the lac, labels and material dictionary
    """

    return AmiraProject(filepath)
