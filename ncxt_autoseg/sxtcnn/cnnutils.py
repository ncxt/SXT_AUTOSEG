"""
various utilities for the cnn
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


class CFMMetrics:
    def __init__(self, id, cfm):
        self.id = id
        self.cfm = cfm
        self._tp = np.diag(cfm)
        self._pred = np.sum(cfm, 0)
        self._label = np.sum(cfm, 1)

    def labeldice(self):
        return 2 * self._tp / (self._pred + self._label)

    def hamming_loss(self):
        return (np.sum(self.cfm) - np.sum(np.diag(self.cfm))) / np.sum(self.cfm)

    def f1_micro(self):
        return 2 * np.sum(self._tp) / (np.sum(self._pred) + np.sum(self._label))

    def f1_macro(self):
        return np.nanmean(2 * self._tp / (self._label + self._pred))

    def recall_micro(self):
        return np.sum(self._tp) / np.sum(self._pred)

    def recall_macro(self):
        return np.nanmean(self._tp / self._pred)

    def precision_micro(self):
        return np.sum(self._tp) / np.sum(self._label)

    def precision_macro(self):
        return np.nanmean(self._tp / self._label)

    def __str__(self):
        header = f"CFMMetrics [{self.id}]\n" + "=" * 20
        attributes = [getattr(self, f) for f in self.__dir__() if not f.startswith("_")]
        metrics = [f"{f.__name__:<20}: {f():.3g}" for f in attributes if callable(f)]
        return "\n".join([header] + metrics)


def plot_confusion_matrix(
    axis,
    matrix,
    classes,
    normalize=True,
    title="Confusion matrix",
    cmap=plt.cm.get_cmap("Blues"),
):
    """
    This function prints and plots the confusion matrix.
      Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        matrix = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]

    axis.imshow(matrix, interpolation="nearest", cmap=cmap)
    axis.set_title(title)
    tick_marks = np.arange(len(classes))
    axis.set_xticks(tick_marks)
    axis.set_xticklabels(classes, rotation=45)
    axis.set_yticks(tick_marks)
    axis.set_yticklabels(classes)
    axis.set_xlim(-0.5, len(classes) - 0.5)
    axis.set_ylim(-0.5, len(classes) - 0.5)

    fmt = ".2f" if normalize else "d"
    thresh = matrix.max() / 2.0
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        axis.text(
            j,
            i,
            format(matrix[i, j], fmt),
            horizontalalignment="center",
            color="white" if matrix[i, j] > thresh else "black",
        )

    axis.set_ylabel("True label")
    axis.set_xlabel("Predicted label")


def confusion_matrix_metric(cfm, index, mode):
    """extract metric from confusion amtrix

    Arguments:
        cfm {[type]} -- confusion matrix
        index {int} -- index of label
        mode {string} -- mode of metric (recall, precision,dice)

    Raises:
        ValueError: If passed mode is invalid   

    Returns:
        [float] -- Metric
    """
    eps = 1e-6

    if mode == "recall":
        return cfm[index, index] / (eps + np.sum(cfm[index, :]))
    if mode == "precision":
        return cfm[index, index] / (eps + np.sum(cfm[:, index]))
    if mode == "dice":
        return (
            2
            * cfm[index, index]
            / (eps + (np.sum(cfm[:, index])) + np.sum(cfm[index, :]))
        )

    raise ValueError("mode must be precision, recall or dice")


def model_confusion_matrix(model_out, labels, num_classes, ignore_index=None):
    """calculate confusion matric from model
    
    Arguments:
        model_out {tensor} -- output from model
        labels {tensor} -- reference
        num_classes {int} -- number of classes
    
    Returns:
        ndarray -- confusion matrix
    """
    output_labels = torch.argmax(model_out, dim=1)

    if output_labels.is_cuda:
        output_labels = output_labels.cpu()
    if labels.is_cuda:
        labels = labels.cpu()

    if ignore_index is not None:
        output_labels[labels == ignore_index] = labels[labels == ignore_index]

    cfm = confusion_matrix(
        labels.numpy().ravel(),
        output_labels.numpy().ravel(),
        labels=np.arange(num_classes),
    )
    return cfm

