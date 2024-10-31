import numpy as np
from .utils import hashvars, stablehash, mask2sdf
from pathlib import Path
from scipy.ndimage import gaussian_filter

from .. import ncxtamira
from ..ncxtamira.organelles import Organelles


class MockLoader:
    def __init__(self, shape, in_channels=1, out_channels=2, length=1):
        self.shape = shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        key = {f"material_{n}": n for n in range(self.out_channels)}
        random_lac = np.random.random((self.in_channels, *self.shape))
        random_label = np.random.randint(
            low=0, high=self.out_channels, size=self.shape, dtype="int"
        )

        return {"input": random_lac, "target": random_label.astype(int), "key": key}

    def __call__(self, data):
        retval = data.copy()
        return retval.reshape(1, *retval.shape)


def feature_selector(image, keydict, features, cellmask=False):
    features = features.copy()
    if "*" in features:
        cellmask = True
        features.remove("*")

    keydict = dict((k.lower(), v) for k, v in keydict.items())

    cell_labels = [v for k, v in keydict.items() if "ext" not in k]
    ignore_labels = [v for k, v in keydict.items() if "ignore" in k]

    ignore_mask = np.isin(image, ignore_labels).astype(int)

    retval_key = {"exterior": 0}

    label = (
        np.isin(image, cell_labels).astype(int)
        if cellmask
        else np.zeros(image.shape, dtype=int)
    )
    if cellmask:
        retval_key["cell"] = 1

    for i, keys in enumerate(features):
        index = cellmask + 1 + i
        # print(i, keys, index)
        # make feature iterable
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        for key in keys:
            try:
                value = keydict[key]
                label[image == value] = index
                retval_key[key] = index
            except KeyError:
                pass

    if np.sum(ignore_mask):
        index = max(retval_key.values()) + 1
        retval_key["ignore"] = index
        label[ignore_mask > 0] = index
    return label, retval_key


class FeatureSelector:
    def __init__(self, key, *features):
        self.key = key.copy()
        self.conversion_dict = {"void": 0}
        self.material_dict = {"void": 0}
        self.cellmask = False
        self.features = []
        for i, feature in enumerate(list(*features)):
            if not isinstance(feature, (list, tuple)):
                feature = [feature]
            if "*" in feature:
                self.cellmask = True
                self.material_dict["wildcard"] = 1
            else:
                self.features.append(feature)

        for i, feature in enumerate(self.features):
            index = self.cellmask + 1 + i
            for material in feature:
                matching = [label for label in self.key.keys() if material in label]
                for m in matching:
                    self.conversion_dict[m] = index
                    self.material_dict[material] = index

    def __call__(self, image):
        # print(self.key, "-->", self.material_dict)

        cell_labels = [v for k, v in self.key.items() if "void" not in k]
        ignore_labels = [v for k, v in self.key.items() if "ignore" in k]
        ignore_mask = np.isin(image, ignore_labels).astype(int)

        retlabel = (
            np.isin(image, cell_labels).astype(int)
            if self.cellmask
            else np.zeros(image.shape, dtype=int)
        )
        for k, v in self.conversion_dict.items():
            if k not in ["void", "wildcard"]:
                retlabel[image == self.key[k]] = v

        if np.sum(ignore_mask):
            index = max(self.material_dict.values()) + 1
            self.material_dict["ignore"] = index
            retlabel[ignore_mask > 0] = index

        return retlabel, self.material_dict


class AmiraLoader:
    def __init__(self, files, features, sanitize=False):
        assert isinstance(features, (list, tuple)), "Give features as list of lists"
        self.files = files
        self.features = [f if isinstance(f, (list, tuple)) else [f] for f in features]
        self.sanitize = sanitize

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = ncxtamira.AmiraCell.from_hx(self.files[index], sanitize=self.sanitize)
        lac_input = data.lac
        label_sel, key = FeatureSelector(data.key, self.features)(data.labels)

        return {
            "input": lac_input.reshape(1, *lac_input.shape),
            "target": label_sel.astype(int),
            "key": key,
        }

    def __call__(self, data):
        retval = data.copy()
        return retval.reshape(1, *retval.shape)


 

class AmiraLoaderOrganelle:
    def __init__(
        self, files, organelles, sanitize=False, scale=100, working_directory=None
    ):
        assert isinstance(organelles, (list, tuple)), "Give features as list of lists"
        self.files = files
        self.organelles = organelles
        self.sanitize = sanitize
        self.scale = scale

        self._features = Organelles.organelles_to_features(organelles)
        self._working_directory = working_directory

    def __len__(self):
        return len(self.files)

    def _load_item(self, index):
        data = ncxtamira.AmiraCell.from_hx(self.files[index], sanitize=self.sanitize)
        lac_input = data.lac * self.scale
        label_sel, key = FeatureSelector(data.key, self._features)(data.labels)

        return {
            "input": lac_input.reshape(1, *lac_input.shape),
            "target": label_sel.astype(int),
            "key": key,
        }

    def __getitem__(self, index):
        if self._working_directory is not None:
            path_item = self._cache_dir / f"item_{index}.npy"
            if path_item.exists():
                item = np.load(path_item, allow_pickle=True).item()
            else:
                item = self._load_item(index)
                path_item.parent.mkdir(parents=True, exist_ok=True)
                np.save(path_item, item, allow_pickle=True)
            return item

        return self._load_item(index)

    def __call__(self, data):
        retval = data.copy() * self.scale
        return retval.reshape(1, *retval.shape)

    @property
    def _cache_dir(self):
        if self._working_directory is None:
            raise ValueError(
                'Caching requires passing option "working_directory" in the construction'
            )
        return Path(self._working_directory) / f"{type(self).__name__}_{self._hash}"

    @property
    def _hash(self):
        modelvars = hashvars(self)
        return stablehash(
            type(self).__name__,
            modelvars,
        )


class AmiraLoaderClahe:
    def __init__(
        self, files, features, sanitize=False, block_shape=(32, 32, 32), clip_limit=0.01
    ):
        assert isinstance(features, (list, tuple)), "Give features as list of lists"
        self.files = files
        self.features = [f if isinstance(f, (list, tuple)) else [f] for f in features]

        self.block_shape = block_shape
        self.clip_limit = clip_limit

        # todo assert features in CellProject

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = ncxtamira.project.AmiraCell.from_hx(self.files[index])
        lac_input = ncxtutils.exposure.clahe_blocks(
            data.lac, block_shape=self.block_shape, clip_limit=self.clip_limit
        )
        label_sel, key = FeatureSelector(data.key, self.features)(data.labels)

        return {
            "input": lac_input.reshape(1, *lac_input.shape),
            "target": label_sel.astype(int),
            "key": key,
        }

    def __call__(self, data):
        retval = data.copy() * 100
        return retval.reshape(1, *retval.shape)




class CascadeAmiraLoader:
    def __init__(self, segmenter, files, features):
        self.files = files
        self.features = features
        self.sigma = 2
        self._segmenter = segmenter
        self._loader = copy.deepcopy(segmenter.loader)
        self._loader.files = files
        self._loader.features = features

    def __len__(self):
        return len(self.files)

    def cascade_input(self, data):
        model_prediction = self._segmenter.model_probability(data)
        for index, _ in enumerate(model_prediction):
            model_prediction[index] = gaussian_filter(
                model_prediction[index], sigma=self.sigma
            )
        model_prediction[0] = data
        return model_prediction

    def __getitem__(self, index):
        sample = self._loader[index]
        sample["input"] = self.cascade_input(sample["input"])
        return sample

    def __call__(self, volume):
        return self.cascade_input(self._loader(volume))


class CascadeAmiraLoaderOrganelles:
    def __init__(self, segmenter, files, organelles):
        self.files = files
        self.organelles = organelles
        self.sigma = 2
        self._segmenter = segmenter
        self._loader = copy.deepcopy(segmenter.loader)
        self._loader.files = files
        self._loader.organelles = organelles

    def __len__(self):
        return len(self.files)

    def cascade_input(self, data):
        model_prediction = self._segmenter.model_probability(data)
        for index, _ in enumerate(model_prediction):
            model_prediction[index] = gaussian_filter(
                model_prediction[index], sigma=self.sigma
            )
        model_prediction[0] = data
        return model_prediction

    def __getitem__(self, index):
        sample = self._loader[index]
        sample["input"] = self.cascade_input(sample["input"])
        return sample

    def __call__(self, volume):
        return self.cascade_input(self._loader(volume))


import copy


class OneHotCascadeAmiraLoader:
    def __init__(self, segmenter, files, features):
        self.files = files
        self.features = features
        self.sigma = 2
        self._segmenter = segmenter
        self._loader = copy.deepcopy(segmenter.loader)
        self._loader.files = files
        self._loader.features = features

    def __len__(self):
        return len(self.files)

    def cascade_input(self, data):
        model_prediction = self._segmenter.model_prediction(data)
        n_values = self._segmenter.model.num_classes
        onehot = np.transpose(np.eye(n_values)[model_prediction], (3, 0, 1, 2))
        onehot[0] = data
        return onehot

    def __getitem__(self, index):
        sample = self._loader[index]
        sample["input"] = self.cascade_input(sample["input"])
        return sample

    def __call__(self, volume):
        return self.cascade_input(self._loader(volume))


class OneHotCascadeAmiraLoaderOrganelles:
    def __init__(self, segmenter, files, organelles):
        self.files = files
        self.organelles = organelles
        self.sigma = 2
        self._segmenter = segmenter
        self._loader = copy.deepcopy(segmenter.loader)
        self._loader.files = files
        self._loader.organelles = organelles

    def __len__(self):
        return len(self.files)

    def cascade_input(self, data):
        model_prediction = self._segmenter.model_prediction(data)
        n_values = self._segmenter.model.num_classes
        onehot = np.transpose(np.eye(n_values)[model_prediction], (3, 0, 1, 2))
        onehot[0] = data
        return onehot

    def __getitem__(self, index):
        sample = self._loader[index]
        sample["input"] = self.cascade_input(sample["input"])
        return sample

    def __call__(self, volume):
        return self.cascade_input(self._loader(volume))


class AmiraSDFLoader:
    def __init__(
        self,
        files,
        organelles,
        sanitize=False,
        scale=100,
        spread=32,
        working_directory=None,
    ):
        assert isinstance(organelles, (list, tuple)), "Give features as list of lists"
        self.files = files
        self.organelles = organelles
        self.sanitize = sanitize
        self.scale = scale
        self.spread = spread

        self._features = Organelles.organelles_to_features(organelles)
        self._working_directory = working_directory

    def __len__(self):
        return len(self.files)

    def _load_item(self, index):
        data = ncxtamira.AmiraCell.from_hx(self.files[index], sanitize=self.sanitize)
        lac_input = data.lac * self.scale
        label_sel, key = FeatureSelector(data.key, self._features)(data.labels)
        sdf = mask2sdf(label_sel > 0, spread=self.spread)

        return {
            "input": lac_input.reshape(1, *lac_input.shape),
            "target": sdf,
            "key": key,
        }

    def __getitem__(self, index):
        if self._working_directory is not None:
            path_item = self._cache_dir / f"item_{index}.npy"
            if path_item.exists():
                item = np.load(path_item, allow_pickle=True).item()
            else:
                item = self._load_item(index)
                path_item.parent.mkdir(parents=True, exist_ok=True)
                np.save(path_item, item, allow_pickle=True)
            return item

        return self._load_item(index)

    def __call__(self, data):
        retval = data.copy() * self.scale
        return retval.reshape(1, *retval.shape)

    @property
    def _cache_dir(self):
        if self._working_directory is None:
            raise ValueError(
                'Caching requires passing option "working_directory" in the construction'
            )
        return Path(self._working_directory) / f"{type(self).__name__}_{self._hash}"

    @property
    def _hash(self):
        modelvars = hashvars(self)
        return stablehash(
            type(self).__name__,
            modelvars,
        )
