"""
wrapper for the pythorch CNN framework
"""
import os
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

# import logging
# logger = logging.getLogger(__name__)

from . import logger


from .cnnutils import (
    confusion_matrix_metric,
    model_confusion_matrix,
    plot_confusion_matrix,
    CFMMetrics,
)
from .datasets import TorchList, TrainBlocks
from .utils import (
    confusion_matrix,
    ensure_dir,
    get_slices,
    hashvars,
    kfold,
    rangebar,
    tqdm_bar,
    stablehash,
    getbestgpu,
    get_free_gpu_memory_map,
)
from .models.torchsummary import size_mb

_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


class Settings:
    """Settings manager for the SXTCNN wrapper"""

    def __init__(self, conf=None):
        """
        Keyword Arguments:
            conf {[dict]} -- [configuration dictionary assigned to
                                            class attributes] (default: {None})

        Raises:
            AttributeError: [If conf dictionary contains non-member attributes]
        """

        self.ignore = -1
        self.learning_rate = 3e-4
        self.learning_rate_window = 10
        self.weight_decay = 3e-5
        self.learning_rate_decay = 0.8
        self.max_without_change = 50
        self.maximum_iterations = 5
        self.augment_linear = 0
        self.augment_elastic = 0

        # undesrscore for members not hashed via properties
        self._batch_size = 2
        self._max_batch_size = 16
        self._num_workers = 4
        self._reset = False
        self._parallel = False
        self._device = "cuda"
        self._cfm_step = 3

        if conf:
            for key, value in conf.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise AttributeError(
                        f'Settings attribute "{key}" not a valid attribute.'
                    )

        self._current_learning_rate = self.learning_rate

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, val):
        self._batch_size = val

    @property
    def max_batch_size(self):
        return self._max_batch_size

    @max_batch_size.setter
    def max_batch_size(self, val):
        self._max_batch_size = val

    @property
    def num_workers(self):
        return self._num_workers

    @num_workers.setter
    def num_workers(self, val):
        self._num_workers = val

    @property
    def reset(self):
        return self._reset

    @reset.setter
    def reset(self, val):
        self._reset = val

    @property
    def parallel(self):
        return self._parallel

    @parallel.setter
    def parallel(self, val):
        self._parallel = val

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, val):
        self._device = val

    @property
    def cfm_step(self):
        return self._cfm_step

    @cfm_step.setter
    def cfm_step(self, val):
        self._cfm_step = val

    def __str__(self):
        header = "SXTCNN settings\n" + "=" * 20
        varlist = [
            f"{k:<20}: {v}" for k, v in vars(self).items() if not k.startswith("_")
        ]
        return "\n".join([header] + varlist)


class TrainLogger:
    """
    Trainer class for the training progrss of the cnn model
    """

    def __init__(self, state=None, cfm_step=5):
        """Initializes ot empty container

        Keyword Arguments:
            cfm_step {int} -- interval for calcvulating the confusion matrix (default: {5})
        """

        self.train_res = [[] for i in range(3)]
        self.valid_res = [[] for i in range(3)]
        self.cfm_step = cfm_step
        self.train_cfm = []
        self.valid_cfm = []

        if state:
            for key, value in state.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def reset(self):
        self.train_res = [[] for i in range(3)]
        self.valid_res = [[] for i in range(3)]
        self.train_cfm = []
        self.valid_cfm = []
        
    def x(self):
        return np.arange(len(self.train_res[0]))

    def plot(self):
        """
        Plot the loss function and CFM for the training
        """
        x = self.x()
        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 5))
        ax1.plot(x, self.train_res[0])
        ax1.plot(x, self.valid_res[0])
        ax1.fill_between(x, self.train_res[1], self.train_res[2], alpha=0.2)
        ax1.fill_between(x, self.valid_res[1], self.valid_res[2], alpha=0.2)

        props = dict(boxstyle="round", facecolor=_COLORS[1], alpha=0.2)
        ax1.text(
            0.95,
            0.95,
            f"Best Validation\n({np.argmin(self.valid_res[0])}/{len(self.valid_res[0])})",
            transform=ax1.transAxes,
            fontsize=14,
            horizontalalignment="right",
            verticalalignment="top",
            bbox=props,
        )

        train_cfm = self.train_cfm
        valid_cfm = self.valid_cfm

        x = np.arange(len(train_cfm)) * self.cfm_step
        n_labels = train_cfm[0].shape[0]

        def plot_cfm(axis, cfmlist):
            for index in range(n_labels):
                cfm_metric = confusion_matrix_metric
                y_prec = [cfm_metric(m, index, "precision") for m in cfmlist]
                y_rec = [cfm_metric(m, index, "recall") for m in cfmlist]
                y_dice = [cfm_metric(m, index, "dice") for m in cfmlist]
                axis.plot(
                    x,
                    y_dice,
                    label=f"Label {index} {y_prec[-1]:.2f} / {y_rec[-1]:.2f}",
                    color=_COLORS[index],
                    linewidth=2,
                )
                axis.plot(x, y_prec, color=_COLORS[index], linestyle="--")
                axis.plot(x, y_rec, color=_COLORS[index], linestyle="-.")
            axis.legend(loc=4)

        plot_cfm(ax2, train_cfm)
        plot_cfm(ax3, valid_cfm)

        ax1.set_xlabel("Loss function")
        ax2.set_xlabel("CFM: Training")
        ax3.set_xlabel("CFM: Validation")


def allvars(obj):
    varsdict = {"name": type(obj).__name__}
    varsdict.update({k: v for k, v in vars(obj).items() if not k.startswith("___")})
    return varsdict


class SXTCNN:
    """CNN annotation using pytoprch"""

    def __init__(
        self,
        loader,
        processor,
        model,
        criterion,
        working_directory,
        conf=None,
        device=None,
    ):

        self.working_directory = Path(working_directory)
        self.settings = Settings(conf)

        self.loader = loader
        self.processor = processor
        self.model = model
        self.criterion = criterion
        self.device = device

        if self.settings.ignore == -1:
            self.settings.ignore = self.model.num_classes

        if self.settings.parallel:
            logger.info("Eanbling DataParallel")
            self.model = torch.nn.DataParallel(self.model)

        self.optimizer = None

        self.train_idx = []
        self.valid_idx = []

        self.epoch = 0
        self.logger = TrainLogger(cfm_step=self.settings.cfm_step)

        self.statedata = ["epoch", "train_idx", "valid_idx"]
        self.iter_callback = lambda x: None
        if self.device:
            self.set_device()

        self._model_size = None

    @property
    def model_size(self):
        if not self._model_size:
            block_shape = (self.model.in_channels, *self.processor.block_shape)
            self._model_size = size_mb(self.model, block_shape, device=self.device)
        return self._model_size

    @property
    def data_folder_train(self):
        return self._data_folder / "train"

    @property
    def data_folder_validation(self):
        return self._data_folder / "validation"

    def logtest(self):
        print(f"logger from {__name__}")
        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warn message")
        logger.error("error message")
        logger.critical("critical message")

    @property
    def num_classes(self):
        if self.settings.parallel:
            return self.model.module.num_classes
        else:
            return self.model.num_classes

    @property
    def file_state(self):
        """Hashed filename for current state"""
        return f"{self._data_folder}/{self._model_hash}_state.npy"

    @property
    def file_weights(self):
        """Hashed filename for current wieghts"""
        return f"{self._data_folder}/{self._model_hash}_{self.epoch}.weights"

    @property
    def file_weights_best(self):
        """Hashed filename for bhest wieghts"""
        return f"{self._data_folder}/{self._model_hash}_best.weights"

    def state_dict(self):
        """Collect state into a dict"""

        # pytorch initializes Models in training mode
        # forcing saved files aas train so hashing works
        self.model.train()
        retval = dict()
        for key in self.statedata:
            retval[key] = self.__getattribute__(key)

        retval.update(hashvars(self.logger))
        return retval

    def load_state_dict(self, state):
        """Collect state into a dict"""

        for key in self.statedata:
            self.__setattr__(key, state[key])

        self.logger = TrainLogger(state, cfm_step=self.settings.cfm_step)

    def load(self):
        """load latest saved state"""
        self.load_state_dict(np.load(self.file_state, allow_pickle=True).item())
        print(f"Loading state {self.epoch} with last weights")
        self.model.load_state_dict(torch.load(self.file_weights))

    def load_best(self):
        """load best saved state"""
        self.load_state_dict(np.load(self.file_state, allow_pickle=True).item())
        print(f"Loading state {self.epoch} with best weights")
        self.model.load_state_dict(torch.load(self.file_weights_best))

    def save(self):
        """save current state and weights"""
        self.model.train()
        np.save(self.file_state, self.state_dict())
        torch.save(self.model.state_dict(), self.file_weights)

    def save_if_best(self):
        """save current state and weights if they have the smallest observed loss"""
        if np.argmin(self.logger.valid_res[0]) == self.epoch - 1:
            self.model.training = False
            np.save(self.file_state, self.state_dict())
            torch.save(self.model.state_dict(), self.file_weights_best)

    @property
    def _data_hash(self):
        return stablehash(
            self.train_idx, hashvars(self.loader), hashvars(self.processor)
        )

    @property
    def _model_hash(self):
        modelvars = hashvars(self.model)
        _ = modelvars.pop("training")

        return stablehash(
            type(self.criterion).__name__, modelvars, hashvars(self.settings)
        )

    @property
    def _data_folder(self):
        return self.working_directory / f"data{self._data_hash}"

    def set_device(self, device=None):
        """Change current device and initialize optimizer

        Keyword Arguments:
            device {string} -- Device can be either cpu, cuda or cuda:n (default: use current device)
        """

        if device is not None:
            self.device = device
        else:
            self.device = self.settings.device
        if self.device == "cuda":
            self.device = f"cuda:{getbestgpu()}"
        logger.info("Setting device to %s", self.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.settings._current_learning_rate,
            weight_decay=self.settings.weight_decay,
        )
        return self.device

    def _needs_initialization(self, mode):
        directory = self._data_folder / mode
        if (
            os.path.isdir(directory)
            and os.listdir(directory)
            and not self.settings.reset
        ):
            return False
        else:
            ensure_dir(directory)
            return True

    def _init_data(self):
        logger.info("Initializing data: %s", self._data_folder)
        logger.info("Training: %s", self.train_idx)
        logger.info("Validation: %s", self.valid_idx)
        if self._needs_initialization("train"):
            self.processor.init_data(
                self.loader,
                folder=self.data_folder_train,
                indices=self.train_idx,
                seed=0,
            )
        if self._needs_initialization("validation"):
            self.processor.init_data(
                self.loader,
                folder=self.data_folder_validation,
                indices=self.valid_idx,
                seed=2**16 - 1,
            )

    def init_data(self, train_idx, valid_idx):
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self._init_data()

    def init_kfold(self, index=0, k=1):
        self.train_idx, self.valid_idx = kfold(index, k, len(self.loader))
        self._init_data()

    def reinit_data(self, mode, seed):
        if mode == "train":
            self.processor.init_data(
                self.loader,
                folder=self.data_folder_train,
                indices=self.train_idx,
                seed=seed,
            )
        elif mode == "validation":
            self.processor.init_data(
                self.loader,
                folder=self.data_folder_validation,
                indices=self.valid_idx,
                seed=seed,
            )

    def epoch_step(self):
        self.epoch += 1
        calc_cfm = (self.epoch % self.settings.cfm_step) == 0

        for step in ["train", "validation"]:
            folder = self._data_folder / step

            loader = torch.utils.data.DataLoader(
                TrainBlocks(
                    folder,
                    random_flip=True,
                    augment_linear=self.settings.augment_linear,
                    augment_elastic=self.settings.augment_elastic,
                ),
                batch_size=self.settings.batch_size,
                num_workers=self.settings.num_workers,
            )

            if step == "train":
                self.model.train()
                logs_cfm = self.logger.train_cfm
                logs_loss = self.logger.train_res
            else:
                self.model.eval()
                logs_cfm = self.logger.valid_cfm
                logs_loss = self.logger.valid_res

            batch_loss = []

            cfm_accumulate = np.zeros((self.num_classes, self.num_classes))

            for sample_batched in loader:
                inputs, labels = sample_batched

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                model_out = self.model(inputs)
                loss = self.criterion(model_out, labels)
                batch_loss.append(loss.item())

                if step == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if calc_cfm:
                    if self.num_classes > 1:
                        cfm_accumulate += model_confusion_matrix(
                            model_out, labels, self.num_classes
                        )

            if calc_cfm:
                logs_cfm.append(cfm_accumulate)

            lossres = (np.mean(batch_loss), np.min(batch_loss), np.max(batch_loss))

            for i, loss in enumerate(lossres):
                logs_loss[i].append(loss)
        # switch back to train, so saving works
        self.model.train()

    def run(self, n_epoch=None, learning_rate=None, show_progress=True):
        self.check_run()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = (
                learning_rate if learning_rate else self.settings._current_learning_rate
            )

        if n_epoch is None:
            n_epoch = self.settings.maximum_iterations

        t = rangebar(n_epoch) if show_progress else range(n_epoch)
        for _ in t:
            self.reinit_data("train", self.epoch)  # TODO: flag as parameter?
            self.epoch_step()
            self.save_if_best()
            self.change_learning_rate()
            self.increase_batch()

            if self.stopping_criterion():
                self.save()
                break

            if show_progress:
                t.set_description(
                    f"loss ({self.logger.train_res[0][-1]:.1e}/{self.logger.valid_res[0][-1]:.1e})"
                )
            self.iter_callback(self)

    def check_run(self):
        if not len(self.train_idx):
            raise AttributeError("train_idx must be initialized")
        if not len(self.valid_idx):
            raise AttributeError("valid_idx must be initialized")
        if self.device is None:
            self.set_device()

    def change_learning_rate(self):
        window = self.settings.learning_rate_window
        factor = self.settings.learning_rate_decay

        x = self.logger.train_res[0]
        if len(x) > window:
            x_moving = np.mean(x[-window:])
            x_past = np.mean(x[-window - 1 : -1])

            if ~(x_moving < x_past):
                self.settings._current_learning_rate *= factor
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.settings._current_learning_rate

    def stopping_criterion(self):
        if self.epoch < 1:
            return False

        if (
            self.epoch - np.argmin(self.logger.valid_res[0])
            > self.settings.max_without_change
        ):
            print(f"No change within {self.settings.max_without_change} iterations")
            return True
        if self.epoch > self.settings.maximum_iterations:
            print(
                f"Maximum number of iteration ({self.settings.maximum_iterations}) reached"
            )
            return True

    def model_probability(self, image):
        assert self.processor is not None, "model_probability needs a defined processor"

        eval_loader = torch.utils.data.DataLoader(
            TorchList(self.processor.forward(image)),
            batch_size=self.settings.batch_size,
        )
        self.model.eval()
        model_res = []
        with torch.no_grad():
            for sample_batched in eval_loader:
                sample_batched = sample_batched.to(self.device)
                output = torch.softmax(self.model(sample_batched), dim=1)
                model_res.append(output.cpu().numpy())

        self.model.train()
        return self.processor.backward(np.concatenate(model_res, axis=0))

    def model_features(self, image):
        """Return the last feature layer of the CNN before the classification layer"""
        assert self.processor is not None, "model_features needs a defined processor"

        eval_loader = torch.utils.data.DataLoader(
            TorchList(self.processor.forward(image)),
            batch_size=self.settings.batch_size,
        )
        self.model.eval()
        model_res = []
        with torch.no_grad():
            for sample_batched in eval_loader:
                sample_batched = sample_batched.to(self.device)
                output = self.model.features(sample_batched)
                model_res.append(output.cpu().numpy())

        return self.processor.backward(np.concatenate(model_res, axis=0))

    def model_prediction(self, image):
        return np.argmax(self.model_probability(image), axis=0)

    def __call__(self, data):
        return self.model_prediction(self.loader(data))

    def train_blocks(self, mode="train", totorch=False):
        assert mode in [
            "train",
            "validation",
        ], "Valid modes are 'train' and 'validation'"
        folder = self._data_folder / mode
        return TrainBlocks(folder, totorch=totorch, random_flip=False)

    def get_training_data(self, index=0, mode="train"):
        assert mode in [
            "train",
            "validation",
        ], "Valid modes are 'train' and 'validation'"

        folder = self._data_folder / mode
        loader = TrainBlocks(folder, random_flip=False)

        self.model.eval()
        with torch.no_grad():
            inputs, labels = loader[index]

            inputs = inputs.view(1, *inputs.shape).to(self.device)
            labels = labels.view(1, *labels.shape).to(self.device)
            output = self.model(inputs)
            if self.num_classes > 1:
                output = torch.softmax(self.model(inputs), dim=1)

        inputs = inputs.cpu().numpy()[0]
        labels = labels.cpu().numpy()[0]
        if self.num_classes > 1:
            print("segmentation")
            output_label = torch.argmax(output, dim=1).cpu().numpy()[0]
        else:
            print("match")
            output_label = (output > 0.5).cpu().numpy()[0, 0]

        output = output.cpu().numpy()[0]
        print(output.shape)
        print(output_label.shape)
        return inputs, labels, output_label, output

    def show_training_data(self, index=0, mode="train"):
        assert mode in [
            "train",
            "validation",
        ], "Valid modes are 'train' and 'validation'"

        inputs, labels, output_label, output = self.get_training_data(index, mode)

        images = []
        names = []
        clims = []

        for i in range(inputs.shape[0]):
            image = np.concatenate(get_slices(inputs[i, :]), 0)
            images.append(image)
            names.append(f"input{i}")
            if i == 0:
                clim = (np.percentile(image, 5), np.percentile(image, 97))
            else:
                clim = (0, 1)
            clims.append(clim)

        images.append(np.concatenate(get_slices(labels), 0))
        names.append(f"labels")
        clims.append((0, self.settings.ignore))

        images.append(np.concatenate(get_slices(output_label), 0))
        names.append(f"result")
        clims.append((0, self.settings.ignore))

        for i in range(output.shape[0]):
            images.append(np.concatenate(get_slices(output[i, :]), 0))
            names.append(f"output{i}")
            clims.append((0, 1))

        f, axes = plt.subplots(1, len(images), figsize=(8, 11))
        for i, el in enumerate(zip(axes.ravel(), images)):
            axis, image = el
            axis.imshow(image, clim=clims[i])
            axis.set_axis_off()
            axis.set_title(names[i])

    def plot_cfm_evaluation(self, data, target, model_prediction):
        labels = set(np.unique(target)) | set(np.unique(model_prediction))
        if self.settings.ignore in labels:
            labels.remove(self.settings.ignore)

        imgs = [*get_slices(data), *get_slices(target), *get_slices(model_prediction)]

        cnf_matrix = confusion_matrix(target, model_prediction, labels=labels)

        plt.figure(figsize=(13, 8))
        gs1 = gridspec.GridSpec(3, 3)
        gs1.update(left=0.05, right=0.55, wspace=0.05)
        axis = [plt.subplot(g) for i, g in enumerate(gs1)]

        gs2 = gridspec.GridSpec(1, 1)
        gs2.update(left=0.60, right=0.98, hspace=0.05)
        print((np.percentile(data, 5), np.percentile(data, 97)))
        for i, (axes, image) in enumerate(zip(axis, imgs)):
            if i < 3:
                clim = (np.percentile(data, 5), np.percentile(data, 97))
            else:
                clim = (0, self.settings.ignore)
            axes.imshow(image, clim=clim)
            axes.axis("off")

        class_names = [int(x) for x in labels]

        ax_cfm = plt.subplot(gs2[0])
        plot_confusion_matrix(
            ax_cfm,
            cnf_matrix,
            classes=class_names,
            title="Normalized(prec) confusion matrix ",
        )
        metrics = ["precision", "recall", "dice"]
        # TODO: fix index
        res = [confusion_matrix_metric(cnf_matrix, 1, m) for m in metrics]
        msg = [f"{m}: {val:.3f}" for m, val in zip(metrics, res)]
        plt.suptitle(msg)

    def evaluate_training_data(self, index=0, mode="train", augment=False):
        assert mode in [
            "train",
            "validation",
        ], "Valid modes are 'train' and 'validation'"

        folder = self._data_folder / mode
        loader = TrainBlocks(folder, random_flip=False)
        if augment:
            loader = TrainBlocks(
                folder,
                random_flip=True,
                augment_affine=self.settings.augment_affine,
                augment_linear=self.settings.augment_linear,
            )

        self.model.eval()
        with torch.no_grad():
            inputs, labels = loader[index]
            labels = labels.long()
            inputs = inputs.view(1, *inputs.shape).to(self.device)
            labels = labels.view(1, *labels.shape).to(self.device)
            output = torch.softmax(self.model(inputs), dim=1)

        inputs = inputs.cpu().numpy()[0]
        labels = labels.cpu().numpy()[0]
        output_label = torch.argmax(output, dim=1).cpu().numpy()[0]

        self.plot_cfm_evaluation(inputs[0], labels, output_label)

    def show_receptive_field(self, index=0, mode="train", threshold=0.5, augment=False):
        assert mode in [
            "train",
            "validation",
        ], "Valid modes are 'train' and 'validation'"

        folder = self._data_folder / mode
        loader = TrainBlocks(folder, random_flip=False)
        if augment:
            loader = TrainBlocks(
                folder,
                random_flip=True,
                augment_affine=self.settings.augment_affine,
                augment_linear=self.settings.augment_linear,
            )

        self.model.eval()
        with torch.no_grad():
            inputs, labels = loader[index]
            labels = labels.long()
            inputs = inputs.view(1, *inputs.shape).to(self.device)
            labels = labels.view(1, *labels.shape).to(self.device)

            tensor = torch.zeros_like(inputs)
            r = [l // 2 for l in tensor.shape]
            tensor[tuple(r)] = 1

            rf = self.model.receptive_fied(tensor)

        inputs = inputs.cpu().numpy()[0]
        labels = labels.cpu().numpy()[0]
        rf = rf.cpu().numpy()
        rf = np.clip(rf, 0, threshold) / threshold

        print(inputs.shape)
        print(labels.shape)
        print(rf.shape)
        imgs = [
            *get_slices(inputs[0]),
            *get_slices(labels),
            *get_slices(inputs[0] * rf),
        ]

        plt.figure(figsize=(13, 8))
        gs1 = gridspec.GridSpec(3, 3)
        gs1.update(left=0.05, right=0.55, wspace=0.05)
        axis = [plt.subplot(g) for i, g in enumerate(gs1)]

        gs2 = gridspec.GridSpec(1, 1)
        gs2.update(left=0.60, right=0.98, hspace=0.05)

        for i, (axes, image) in enumerate(zip(axis, imgs)):
            if i == 1:
                clim = (0, self.settings.ignore)
            else:
                clim = (np.percentile(inputs, 5), np.percentile(inputs, 97))
            axes.imshow(image, clim=clim)
            axes.axis("off")

    def evaluate_sample(self, index, loader=None, plot=False):
        if loader is None:
            loader = self.loader

        sample = loader[index]
        data = sample["input"]
        target = sample["target"]
        model_prediction = self.model_prediction(data)

        model_prediction_masked = np.copy(model_prediction)

        model_prediction_masked[target == self.settings.ignore] = target[
            target == self.settings.ignore
        ]
        labels = np.arange(self.model.num_classes)
        cnf_matrix = confusion_matrix(target, model_prediction_masked, labels=labels)

        if plot == False:
            return cnf_matrix

        imgs = [
            *get_slices(data[0]),
            *get_slices(target),
            *get_slices(model_prediction),
        ]

        plt.figure(figsize=(13, 8))
        gs1 = gridspec.GridSpec(3, 3)
        gs1.update(left=0.05, right=0.55, wspace=0.05)
        axis = [plt.subplot(g) for i, g in enumerate(gs1)]

        gs2 = gridspec.GridSpec(1, 1)
        gs2.update(left=0.60, right=0.98, hspace=0.05)

        for i, (axes, image) in enumerate(zip(axis, imgs)):
            if i < 3:
                clim = (np.percentile(data[0], 5), np.percentile(data[0], 97))
            else:
                clim = (0, self.settings.ignore)
            axes.imshow(image, clim=clim)
            axes.axis("off")

        class_names = [int(x) for x in labels]

        ax_cfm = plt.subplot(gs2[0])
        plot_confusion_matrix(
            ax_cfm,
            cnf_matrix,
            classes=class_names,
            title="Normalized(prec) confusion matrix ",
        )
        metrics = ["precision", "recall", "dice"]
        # TODO: fix index
        res = [confusion_matrix_metric(cnf_matrix, 1, m) for m in metrics]
        msg = [f"{m}: {val:.3f}" for m, val in zip(metrics, res)]
        plt.suptitle(msg)

        return cnf_matrix

    def validation_metrics(self):
        print(f"validation_metrics on {self.valid_idx}")
        return [
            CFMMetrics(i, self.evaluate_sample(i)) for i in tqdm_bar(self.valid_idx)
        ]

    def load_trained(self):
        logstr = f"loading model model {self._model_hash} {hashvars(self.model)}"
        logger.info(logstr)

        try:
            self.check_run()
            self.load()
            logstr = f"State at epoch {self.epoch} found"
            logger.info(logstr)
        except FileNotFoundError as not_found:
            print(f"Missing file {not_found.filename}")
            print("State not found, training")

            self.run(n_epoch=self.settings.maximum_iterations)
            print(f"model {self._model_hash} {hashvars(self.model)}")
            self.save()
            print(f"model {self._model_hash} {hashvars(self.model)}")

    def increase_batch(self, extra_buffer=2):
        cuda_index = int(self.device.split(":")[1])
        memory_avail = get_free_gpu_memory_map()[cuda_index]

        if (memory_avail - self.model_size * extra_buffer) > self.model_size:
            logger.info(f"model_size {self.model_size} memory_avail {memory_avail}")
            new_size = min(self.settings.max_batch_size, self.settings.batch_size + 1)
            logger.info(f"Increasing batch to {new_size}")
            self.settings.batch_size = new_size
