import copy
import hashlib
import importlib
import json
import multiprocessing

import numpy as np

from .sxtcnn import SXTCNN
from .sxtcnn.utils import hashvars, stablehash, getbestgpu
import tempfile
from pathlib import Path


class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs["context"] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)


def class_from_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def get_loader(name):
    return class_from_name("ncxt_sxtcnn.sxtcnn.loaders", name)


def get_processor(name):
    return class_from_name("ncxt_sxtcnn.sxtcnn.processors", name)


def get_model(name):
    return class_from_name("ncxt_sxtcnn.sxtcnn.models", name)


def get_criterion(name):
    return class_from_name("ncxt_sxtcnn.sxtcnn.criteria", name)


def arg_as_str(cls):
    return [f"  {k:<20}: {v}" for k, v in vars(cls).items() if not k.startswith("_")]


def arg_as_dict(cls):
    return dict([(k, v) for k, v in vars(cls).items() if not k.startswith("_")])


class Segmenter:
    def __init__(
        self,
        loader,
        processor,
        model,
        criterion,
        loader_args=None,
        processor_args=None,
        model_args=None,
        criterion_args=None,
        settings=None,
        fold=0,
        working_directory=None,
    ):

        if loader_args is not None:
            self._loader_args = loader_args
        else:
            self._loader_args = dict()

        if processor_args is not None:
            self._processor_args = processor_args
        else:
            self._processor_args = dict()

        if model_args is not None:
            self._model_args = model_args
        else:
            self._model_args = dict()

        if criterion_args is not None:
            self._criterion_args = criterion_args
        else:
            self._criterion_args = dict()

        if settings is not None:
            self._settings = settings
        else:
            self._settings = dict()

        self._folder = (
            Path(tempfile.gettempdir())
            if working_directory is None
            else working_directory
        )
        self._fold = fold
        self._device = None

        # TODO: ignore as argument
        self._criterion_args["ignore_index"] = self._model_args["num_classes"]

        self._loader = loader(**self._loader_args)
        self._processor = processor(**self._processor_args)
        self._model = model(**self._model_args)
        self._criterion = criterion(**self._criterion_args)

        self._validation_metrics = None
        self._seg = None

    @property
    def fold(self):
        return self._fold

    @fold.setter
    def fold(self, val):
        self._fold = val

    @property
    def folder(self):
        return self._folder

    @folder.setter
    def folder(self, value):
        self._folder = Path(value)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        if value == "cuda":
            self._device = f"cuda:{getbestgpu()}"
        else:
            self._device = value

    @property
    def hash_variables(self):
        loader = [type(self._loader).__name__, arg_as_dict(self._loader)]
        processor = [type(self._processor).__name__, arg_as_dict(self._processor)]
        model = [type(self._model).__name__, arg_as_dict(self._model)]
        criterion = [type(self._criterion).__name__]
        settings = arg_as_dict(self._seg.settings)

        return {
            "loader": loader,
            "processor": processor,
            "model": model,
            "criterion": criterion,
            "settings": settings,
        }

    @property
    def hash(self):
        return stablehash(self.hash_variables)

    def setup(self):
        self._seg = SXTCNN(
            self._loader,
            self._processor,
            self._model,
            self._criterion,
            self._folder,
            self._settings,
        )
        if self._device:
            self._seg.set_device(self._device)

    @property
    def validation_metrics(self):
        if self._validation_metrics:
            return self._validation_metrics
        self.eval_validation_metrics()
        return self._validation_metrics

    def sxtcnn(self, index):
        self.setup()
        self._seg.init_kfold(index, self._fold)
        return self._seg

    def load_trained(self, index):
        self.setup()
        self._seg.init_kfold(index, self._fold)
        self._seg.load_trained()
        return self._seg

    def data(self, fold, mode="train"):
        self.setup()
        self._seg.init_kfold(fold, self._fold)
        return self._seg.train_blocks(mode)

    def eval_validation_metrics(self):
        self._validation_metrics = []

        if self._fold == 0:
            self.setup()
            print(f"{__name__}.eval_metrics {self._fold} ")
            self._seg.init_kfold(0, self._fold)
            self._seg.load_trained()
            self._validation_metrics = self._seg.validation_metrics()

        for i in range(self._fold):
            self.setup()
            print(f"{__name__}.eval_metrics {i}/{self._fold} ")
            self._seg.init_kfold(i, self._fold)
            self._seg.load_trained()
            self._validation_metrics.extend(self._seg.validation_metrics())

        self._validation_metrics = sorted(self._validation_metrics, key=lambda x: x.id)

    # def train_kfold(self, i, folder="c:\\Users\\axela\\Documents\\2019\\wdcnn\\"):
    #     print(f"k_fold_validation_metrics {i}")
    #     _loader = get_loader(self.loader)(self.files, *self.features)
    #     _processor = get_processor(self.processor)(**self.processor_args)
    #     _model = get_model(self.model)(**self.model_args)
    #     _criterion = get_criterion(self.criterion)(**self._criterion_args)
    #     seg = SXTCNN(_loader, _processor, _model, _criterion, folder, self.cnn_args)
    #     seg.init_kfold(i, self.fold)
    #     seg.set_device(i)
    #     seg.load_trained(self.n_iterations)
    #     return

    # def eval_metrics_mp(self):
    #     print("Evaluating k-fold metrics")
    #     pool = MyPool(processes=3)
    #     args = [i for i in range(self.fold)]
    #     _ = pool.map(self.train_kfold, args)

    def metrics(self):
        metrics = self.validation_metrics
        hamming = [el.hamming_loss() for el in metrics]
        f1_micro = [el.f1_micro() for el in metrics]
        f1_macro = [el.f1_macro() for el in metrics]
        recall_micro = [el.recall_micro() for el in metrics]
        recall_macro = [el.recall_macro() for el in metrics]
        precision_micro = [el.precision_micro() for el in metrics]
        precision_macro = [el.precision_macro() for el in metrics]
        labeldice = [el.labeldice() for el in metrics]

        return {
            "hamming": [np.mean(hamming), hamming],
            "f1_micro": [np.mean(f1_micro), f1_micro],
            "f1_macro": [np.mean(f1_macro), f1_macro],
            "recall_micro": [np.mean(recall_micro), recall_micro],
            "recall_macro": [np.mean(recall_macro), recall_macro],
            "precision_micro": [np.mean(precision_micro), precision_micro],
            "precision_macro": [np.mean(precision_macro), precision_macro],
            "precision_macro": [np.mean(precision_macro), precision_macro],
            "labeldice": [
                np.mean(labeldice, 0).tolist(),
                [el.tolist() for el in labeldice],
            ],
        }

    @property
    def jsonpath(self):
        return self._folder / f"segmenter_{self.hash}_{self._fold}.json"

    def kfold_result(self, kfold):
        self._fold = kfold
        jsondict = self.export_dict()
        jsondict["metrics"] = self.metrics()
        jsondump = json.dumps(jsondict, indent=2)
        with open(self.jsonpath, "w") as outfile:
            json.dump(jsondict, outfile, indent=2)

        return jsondump

    def export_dict(self):
        retval = dict()
        # retval["Segmenter"] = arg_as_dict(self)
        for name in ["loader", "processor", "model", "criterion"]:
            member = getattr(self, "_" + name)
            retval[name] = [type(member).__name__, arg_as_dict(member)]
        segdict = arg_as_dict(self._seg.settings)
        retval["settings"] = segdict
        return retval

    @classmethod
    def from_dict(cls, dictionary, fold=None, working_directory=None):

        # init_args = dictionary["Segmenter"]
        loader, loader_args = dictionary["loader"]
        processor, processor_args = dictionary["processor"]
        model, model_args = dictionary["model"]
        criterion, criterion_args = dictionary["criterion"]
        settings_args = dictionary["settings"]
        model_args.pop("training", None)
        criterion_args.pop("training", None)

        retval = cls(
            loader=get_loader(loader),
            processor=get_processor(processor),
            model=get_model(model),
            criterion=get_criterion(criterion),
            loader_args=loader_args,
            processor_args=processor_args,
            model_args=model_args,
            criterion_args=criterion_args,
            settings=settings_args,
            fold=fold,
            working_directory=working_directory,
        )

        retval.setup()
        return retval

    @classmethod
    def from_json(cls, filename, fold=None):
        with open(filename, "r") as read_file:
            jsondict = json.load(read_file)
            return cls.from_dict(jsondict, fold=fold, working_directory=filename.parent)

    def k_fold_ensamble(self, data):
        p_list = []
        for i in range(3):
            cnn = self.load_trained(i)
            p_list.append(cnn.model_probability(cnn.loader(data)))

        p_ensamble = np.mean(p_list, 0)
        return np.argmax(p_ensamble, 0)

