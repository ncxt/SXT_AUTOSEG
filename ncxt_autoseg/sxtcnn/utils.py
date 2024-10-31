import numpy as np
import hashlib
import os
from tqdm import trange, tnrange
from tqdm import tqdm, tqdm_notebook
import torch
import logging
import subprocess
from scipy import ndimage as ndi


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def rangebar(n):
    rangefunc = tnrange if isnotebook() else trange
    return rangefunc(n)


def tqdm_bar(n):
    barfunc = tqdm_notebook if isnotebook() else tqdm
    return barfunc(n)


def kfold(index, k, n_samples, random_seed=1):
    random_idx = np.random.RandomState(random_seed).permutation(n_samples)
    if not k:
        return random_idx, random_idx

    assert index < k, "Index must be less then k"
    assert k > 1, "Fold k must be either larger than 1 or 0 (duplicates all)"

    folds = np.array_split(random_idx, k)

    valid = folds[index]
    folds.pop(index)

    train = [item for sublist in folds for item in sublist]
    return train, valid


def stablehash(*arglist):
    identifier = (".").join([str(x) for x in arglist])
    return int(hashlib.sha256(identifier.encode("utf-8")).hexdigest(), 16) % 2 ** 16


def hashvars(obj):
    varsdict = {"name": type(obj).__name__}
    varsdict.update({k: v for k, v in vars(obj).items() if not k.startswith("_")})
    return varsdict


def path_not_empty(directory):
    if os.path.isdir(directory):
        # directory exists
        if os.listdir(directory):
            return True
    return False


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


def get_slices(image):
    s0, s1, s2 = [int(s / 2) for s in image.shape]
    return image[s0, :, :], image[:, s1, :], image[:, :, s2]


def confusion_matrix(a, b, labels):
    n_labels = len(labels)
    cfm = np.zeros((n_labels, n_labels), dtype=int)
    for i, value_i in enumerate(labels):
        for j, value_j in enumerate(labels):
            sela = a == value_i
            selb = b == value_j
            cfm[i, j] = int(np.sum(sela * selb))
    return cfm


def gpuinfo(gpuid):

    sp = subprocess.Popen(
        ["nvidia-smi", "-q", "-i", str(gpuid), "-d", "MEMORY"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out_str = sp.communicate()
    out_list = out_str[0].decode("utf-8").split("BAR1", 1)[0].split("\n")
    out_dict = {}
    for item in out_list:
        try:
            key, val = item.split(":")
            key, val = key.strip(), val.strip()
            out_dict[key] = val
        except:
            pass
    return out_dict


def getfreegpumem(id):
    return int(gpuinfo(id)["Free"].replace("MiB", "").strip())


def getbestgpu():
    freememlist = []
    for gpu_id in range(torch.cuda.device_count()):
        freemem = getfreegpumem(gpu_id)
        logging.debug("GPU device %d has %d MiB left.", gpu_id, freemem)
        freememlist.append(freemem)
    idbest = freememlist.index(max(freememlist))
    # print("--> GPU device %d was chosen" % idbest)
    return idbest


def get_free_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """

    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
        encoding="utf-8",
    )
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def mask2sdf(mask, spread=32):
    edt_bg = np.clip(ndi.distance_transform_edt(np.bitwise_not(mask))-0.5, 0,np.infty)
    edt_fg = np.clip(ndi.distance_transform_edt(mask)-0.5, 0,np.infty)        
    edt_bg_norm = np.clip(edt_bg / spread, 0, 1)
    edt_fg_norm = np.clip(edt_fg / spread, 0, 1)
    return (1 + edt_fg_norm - edt_bg_norm) / 2
