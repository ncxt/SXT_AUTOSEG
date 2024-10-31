from abc import ABC, abstractmethod

import numpy as np
import volumeblocks
from scipy.signal.windows import triang

from ..utils.ndarray import bin_ndarray_mode, bin_ndarray_single
from ..utils.ndarray import binned_shape, upscaled_shape
from ..utils.ndarray import upscale, upscale_dims
from .utils import tqdm_bar

from volumeblocks.utils import window


class DataProcessor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, data):
        pass

    @abstractmethod
    def backward(self, data):
        pass

    @abstractmethod
    def init_data(self, loader, folder, indices, seed):
        """Abstact method for the data initialization

        Arguments:
            loader {NCXTDBLoader} -- Data loader
            folder {string} -- working directory for the data
            indices {list} -- list of indices for loader
        """


class DataBinner:
    def __init__(self):
        self._shape = None
        self._binning = None

    def forward(self, data, binning, mode=False):
        if mode:
            assert issubclass(
                data.dtype.type, np.integer
            ), f"Array {data.dtype} must be integer if mode ==True"

        if data.ndim == 3:
            self._shape = data.shape
        if data.ndim == 4:
            _, self._shape = data.shape[0], data.shape[1:]

        self._binning = binning
        binfunc = bin_ndarray_mode if mode else bin_ndarray_single
        slices = tuple([slice(0, s) for s in upscaled_shape(data.shape, self._binning)])
        return binfunc(data[slices], binning)

    def backward(self, data):
        assert self._shape is not None, "Backward needs a defined shape"

        if data.ndim == 3:
            upscaled = upscale(data, self._binning)
            slices = tuple([slice(0, s) for s in upscaled.shape])
            retval = np.zeros(self._shape, dtype=data.dtype)
            retval[slices] = upscaled
            return retval
        if data.ndim == 4:
            n_channels = data.shape[0]
            upscaled = upscale_dims(data, [1, 2, 3], self._binning)
            slices = tuple([slice(0, s) for s in upscaled.shape])
            retval = np.zeros((n_channels, *self._shape), dtype=data.dtype)
            retval[slices] = upscaled[slices]
            return retval
        assert False, "Dimensions other than 3 and 1+3 not supported"


class DataSampler:
    def __init__(self, shape):
        self.roi_shape = shape
        self._shape = None
        self._seed = None

    def set_seed(self, seed):
        self._seed = seed

    def data_slices(self):
        # print(f"self shape {self._shape} roi {self.roi_shape}")

        valid_length = np.minimum(self.roi_shape, self._shape)
        input_offset = [max(0, s - l) for l, s in zip(valid_length, self._shape)]
        out_offset = [max(0, s - l) for l, s in zip(valid_length, self.roi_shape)]

        if self._seed is not None and self._seed < 0:
            input_offset = [x // 2 if x else x for x in input_offset]
            out_offset = [x // 2 if x else x for x in out_offset]
        else:
            random_generator = np.random.RandomState(self._seed)
            input_offset = [
                random_generator.randint(x) if x else x for x in input_offset
            ]
            out_offset = [random_generator.randint(x) if x else x for x in out_offset]

        slices_inp = tuple(
            [slice(off, off + l) for off, l in zip(input_offset, valid_length)]
        )
        slices_out = tuple(
            [slice(off, off + l) for off, l in zip(out_offset, valid_length)]
        )
        # print(f"Seed = {self._seed} offset = {input_offset} {out_offset}")

        return slices_inp, slices_out

    def forward(self, data):
        if data.ndim == 3:
            self._shape = data.shape
        if data.ndim == 4:
            n_channels, self._shape = data.shape[0], data.shape[1:]

        slices_data, slices_sample = self.data_slices()

        if data.ndim == 3:
            retval = np.zeros(self.roi_shape, dtype=data.dtype)
        if data.ndim == 4:
            retval = np.zeros((n_channels, *self.roi_shape), dtype=data.dtype)
            slice_dim = slice(0, n_channels)
            slices_data = tuple([slice_dim, *slices_data])
            slices_sample = tuple([slice_dim, *slices_sample])

        retval[slices_sample] = data[slices_data]
        return retval

    def backward(self, sample):
        assert self._shape is not None, "Backward needs a defined shape"
        slices_data, slices_sample = self.data_slices()
        if sample.ndim == 5:
            assert sample.shape[0] == 1, "backward only supports single images"
            sample = sample.reshape(sample.shape[1:])

        if sample.ndim == 3:
            retval = np.zeros(self._shape, dtype=sample.dtype)
        if sample.ndim == 4:
            n_channels = sample.shape[0]
            retval = np.zeros((n_channels, *self._shape), dtype=sample.dtype)
            slice_dim = slice(0, n_channels)
            slices_data = tuple([slice_dim, *slices_data])
            slices_sample = tuple([slice_dim, *slices_sample])

        retval[slices_data] = sample[slices_sample]
        return retval


class RandomSingleBlockProcessor(DataProcessor):
    def __init__(self, block_shape=None, binning=1, n_blocks=1):
        super().__init__()
        self.block_shape = tuple(block_shape)
        self.binning = binning
        self.n_blocks = n_blocks

        self._shape = None
        self._loader = None
        self._seed = None
        self._binner = DataBinner()
        self._sampler = DataSampler(block_shape)

    def setloader(self, loader):
        self._loader = loader

    def forward_sample(self, data, mode=False, seed=None):
        if seed is None:
            seed = self._seed
        self._sampler.set_seed(seed)
        binned = self._binner.forward(data, self.binning, mode)
        sampled = self._sampler.forward(binned)
        return sampled

    def forward(self, data, mode=False, seed=None):
        assert data.ndim == 4, "Full forward only on (1+3)d data"
        block_height = self.block_shape[1] * self.binning

        self._sampler.set_seed(-1)
        if data.shape[2] > block_height:
            # print(f"Data shape {data.shape} splitting")
            self._shape = data.shape[-3:]
            sample1 = data[:, :, :block_height, :]
            sample2 = data[:, :, -block_height:, :]

            binned1 = self._binner.forward(sample1, self.binning, mode)
            sampled1 = self._sampler.forward(binned1)
            binned2 = self._binner.forward(sample2, self.binning, mode)
            sampled2 = self._sampler.forward(binned2)
            return [sampled1, sampled2]

        binned = self._binner.forward(data, self.binning, mode)
        sampled = self._sampler.forward(binned)
        return sampled

    def backward(self, sample):
        if sample.ndim == 5 and sample.shape[0] > 1:
            # print(f"Data shape {sample.shape} fusing")
            block_height = self.block_shape[0] * self.binning

            b_sampled1 = self._sampler.backward(sample[0])
            b_sampled2 = self._sampler.backward(sample[1])
            b_binned1 = self._binner.backward(b_sampled1)
            b_binned2 = self._binner.backward(b_sampled2)

            retval = np.zeros((sample.shape[1], *self._shape))

            norm = np.ones((sample.shape[1], *self._shape)) * 1e-6
            window1 = window(b_binned1.shape, triang)
            window2 = window(b_binned2.shape, triang)
            retval[:, :, :block_height, :] += window1 * b_binned1
            retval[:, :, -block_height:, :] += window1 * b_binned2
            norm[:, :, :block_height, :] += window1
            norm[:, :, -block_height:, :] += window2
            return retval / norm

        b_sampled = self._sampler.backward(sample)
        b_binned = self._binner.backward(b_sampled)
        return b_binned

    def __getitem__(self, index):
        sample = self._loader[index]
        data = sample["input"]
        labels = sample["target"]
        seed = self._seed + index if self._seed else np.random.randint(1000)
        return self.forward(data, seed=seed), self.forward(labels, mode=True, seed=seed)

    def init_data(self, loader, folder, indices, seed):
        self.setloader(loader)

        fileindex = 0
        for ind in tqdm_bar(indices):
            sample = self._loader[ind]
            data = sample["input"]
            labels = sample["target"]

            for si in range(self.n_blocks):
                dataseed = seed + ind * self.n_blocks + si
                x = self.forward_sample(data, seed=dataseed)
                y = self.forward_sample(labels, mode=True, seed=dataseed)

                data_dict = {"x": x, "y": y}
                np.save(folder / f"data{fileindex}", data_dict)
                fileindex += 1


class NonZeroSplitProcessor(DataProcessor):
    def __init__(
        self,
        block_shape=(32, 32, 32),
        binning=1,
        pad=0,
        sampling=1.1,
        rel_threshold=0.1,
    ):
        super().__init__()
        self.block_shape = tuple(block_shape)
        self.binning = binning
        self.pad = pad
        self.sampling = sampling
        self.rel_threshold = rel_threshold

        self._sampling = 1.5
        self._loader = None
        self._seed = None
        self._sampler = None

    def setloader(self, loader):
        self._loader = loader

    def forward(self, data):
        block_shape_big = [s * self.binning for s in self.block_shape]
        sampler_size = [d + self.pad for d in block_shape_big]

        self._sampler = DataSampler(np.maximum(sampler_size, data.shape[-3:]))
        self._sampler.set_seed(-1)

        lac_blocks = volumeblocks.split(
            self._sampler.forward(data),
            block_shape_big,
            binning=self.binning,
            sampling=self._sampling,
        )
        return lac_blocks

    def backward(self, data):
        ret_dim = data.shape[1]
        retval_shape = [ret_dim] + list(self._sampler.roi_shape)
        return self._sampler.backward(
            volumeblocks.fuse(
                data,
                retval_shape,
                binning=self.binning,
                sampling=self._sampling,
                windowfunc=triang,
            )
        )

    def get_data(self, index, seed):

        sample = self._loader[index]
        data = sample["input"]
        labels = sample["target"]

        block_shape_big = [s * self.binning for s in self.block_shape]
        sampler_size = [d + self.pad for d in block_shape_big]
        data_size = [d + self.pad for d in labels.shape]

        self._sampler = DataSampler(np.maximum(sampler_size, data_size))
        self._sampler.set_seed(seed)

        blocks_x = volumeblocks.split(
            self._sampler.forward(data),
            block_shape_big,
            binning=self.binning,
            sampling=self.sampling,
        )

        blocks_y = volumeblocks.split_label(
            self._sampler.forward(labels),
            block_shape_big,
            binning=self.binning,
            sampling=self.sampling,
        )
        nonzero = [np.sum(v > 0) for v in blocks_y]
        labelled = [np.prod(v.shape) for v in blocks_y]
        print(nonzero[:5])
        print(labelled[:5])

        if sample["key"].get("ignore"):
            ignore_val = sample["key"]["ignore"]
            nonzero = [np.sum((v > 0) * (v != ignore_val)) for v in blocks_y]
            labelled = [np.sum(v != ignore_val) for v in blocks_y]
            print(nonzero[:5])
            print(labelled[:5])

        ind = [
            i
            for i, n in enumerate(zip(nonzero, labelled))
            if n[0] / n[1] > self.rel_threshold
        ]
        print(
            f"Split has {len(blocks_y)} blocks  of shape {blocks_y[0].shape} of which {len(ind)} are nonzero in volume {labels.shape}"
        )
        return [blocks_x[i] for i in ind], [blocks_y[i] for i in ind]

    def init_data(self, loader, folder, indices, seed):
        self.setloader(loader)
        self._seed = seed

        fileindex = 0
        for ind in tqdm_bar(indices):
            seed = self._seed + ind if self._seed else np.random.randint(1000)
            blocks_x, blocks_y = self.get_data(ind, seed)

            for bx, by in zip(blocks_x, blocks_y):
                data_dict = {"x": bx, "y": by}
                np.save(folder / f"data{fileindex}", data_dict)
                fileindex += 1

    def __getitem__(self, index):
        seed = self._seed + index if self._seed else np.random.randint(1000)
        return self.get_data(index, seed)


class RandomBlockProcessor(DataProcessor):
    def __init__(self, block_shape=(32, 32, 32), binning=1, pad=0, n_blocks=10):
        super().__init__()
        self.block_shape = tuple(block_shape)
        self.binning = binning
        self.n_blocks = n_blocks
        self.pad = pad

        self._sampling = 1.5
        self._loader = None
        self._seed = None
        self._sampler = None

    def setloader(self, loader):
        self._loader = loader

    def forward(self, data):
        block_shape_big = [s * self.binning for s in self.block_shape]
        sampler_size = [d + self.pad for d in block_shape_big]

        self._sampler = DataSampler(np.maximum(sampler_size, data.shape[-3:]))
        self._sampler.set_seed(-1)

        lac_blocks = volumeblocks.split(
            self._sampler.forward(data),
            block_shape_big,
            binning=self.binning,
            sampling=self._sampling,
        )
        return lac_blocks

    def backward(self, data):
        ret_dim = data.shape[1]
        retval_shape = [ret_dim] + list(self._sampler.roi_shape)
        return self._sampler.backward(
            volumeblocks.fuse(
                data,
                retval_shape,
                binning=self.binning,
                sampling=self._sampling,
                windowfunc=triang,
            )
        )

    def init_data(self, loader, folder, indices, seed):
        self.setloader(loader)
        self._seed = seed
        block_shape_big = [s * self.binning for s in self.block_shape]
        sampler_size = [d + self.pad for d in block_shape_big]

        fileindex = 0
        for ind in indices:
            sample = self._loader[ind]
            data = sample["input"]
            labels = sample["target"]

            seed = self._seed + ind if self._seed else np.random.randint(1000)
            self._sampler = DataSampler(np.maximum(sampler_size, labels.shape))
            self._sampler.set_seed(seed)

            blocks_x = volumeblocks.random_blocks(
                self._sampler.forward(data),
                block_shape_big,
                max_patches=self.n_blocks,
                binning=self.binning,
                random_state=seed + ind,
            )

            blocks_y = volumeblocks.random_blocks_label(
                self._sampler.forward(labels),
                block_shape_big,
                max_patches=self.n_blocks,
                binning=self.binning,
                random_state=seed + ind,
            )

            for bx, by in zip(blocks_x, blocks_y):
                data_dict = {"x": bx, "y": by}
                np.save(folder / f"data{fileindex}", data_dict)
                fileindex += 1

    def __getitem__(self, index):
        block_shape_big = [s * self.binning for s in self.block_shape]

        sample = self._loader[index]
        data = sample["input"]
        labels = sample["target"]
        seed = self._seed + index if self._seed else np.random.randint(1000)

        self._sampler = DataSampler(np.maximum(block_shape_big, labels.shape))
        self._sampler.set_seed(seed)
        print(f"seed {seed}")

        blocks_x = volumeblocks.random_blocks(
            self._sampler.forward(data),
            block_shape_big,
            max_patches=self.n_blocks,
            binning=self.binning,
            random_state=seed,
        )

        blocks_y = volumeblocks.random_blocks_label(
            self._sampler.forward(labels),
            block_shape_big,
            max_patches=self.n_blocks,
            binning=self.binning,
            random_state=seed,
        )
        return blocks_x, blocks_y


class RandomFloatBlockProcessor(DataProcessor):
    def __init__(self, block_shape=(32, 32, 32), binning=1, pad=0, n_blocks=10):
        super().__init__()
        self.block_shape = tuple(block_shape)
        self.binning = binning
        self.n_blocks = n_blocks
        self.pad = pad

        self._sampling = 1.5
        self._loader = None
        self._seed = None
        self._sampler = None

    def setloader(self, loader):
        self._loader = loader

    def forward(self, data):
        block_shape_big = [s * self.binning for s in self.block_shape]
        sampler_size = [d + self.pad for d in block_shape_big]

        self._sampler = DataSampler(np.maximum(sampler_size, data.shape[-3:]))
        self._sampler.set_seed(-1)

        lac_blocks = volumeblocks.split(
            self._sampler.forward(data),
            block_shape_big,
            binning=self.binning,
            sampling=self._sampling,
        )
        return lac_blocks

    def backward(self, data):
        ret_dim = data.shape[1]
        retval_shape = [ret_dim] + list(self._sampler.roi_shape)
        return self._sampler.backward(
            volumeblocks.fuse(
                data,
                retval_shape,
                binning=self.binning,
                sampling=self._sampling,
                windowfunc=triang,
            )
        )

    def init_data(self, loader, folder, indices, seed):
        self.setloader(loader)
        self._seed = seed
        block_shape_big = [s * self.binning for s in self.block_shape]
        sampler_size = [d + self.pad for d in block_shape_big]

        fileindex = 0
        for ind in indices:
            sample = self._loader[ind]
            data = sample["input"]
            labels = sample["target"]

            seed = self._seed + ind if self._seed else np.random.randint(1000)
            self._sampler = DataSampler(np.maximum(sampler_size, labels.shape))
            self._sampler.set_seed(seed)

            blocks_x = volumeblocks.random_blocks(
                self._sampler.forward(data),
                block_shape_big,
                max_patches=self.n_blocks,
                binning=self.binning,
                random_state=seed + ind,
            )

            blocks_y = volumeblocks.random_blocks(
                self._sampler.forward(labels),
                block_shape_big,
                max_patches=self.n_blocks,
                binning=self.binning,
                random_state=seed + ind,
            )

            for bx, by in zip(blocks_x, blocks_y):
                data_dict = {"x": bx, "y": by}
                np.save(folder / f"data{fileindex}", data_dict)
                fileindex += 1

    def __getitem__(self, index):
        block_shape_big = [s * self.binning for s in self.block_shape]

        sample = self._loader[index]
        data = sample["input"]
        labels = sample["target"]
        seed = self._seed + index if self._seed else np.random.randint(1000)

        self._sampler = DataSampler(np.maximum(block_shape_big, labels.shape))
        self._sampler.set_seed(seed)

        blocks_x = volumeblocks.random_blocks(
            self._sampler.forward(data),
            block_shape_big,
            max_patches=self.n_blocks,
            binning=self.binning,
            random_state=seed,
        )

        blocks_y = volumeblocks.random_blocks(
            self._sampler.forward(labels),
            block_shape_big,
            max_patches=self.n_blocks,
            binning=self.binning,
            random_state=seed,
        )
        return blocks_x, blocks_y
