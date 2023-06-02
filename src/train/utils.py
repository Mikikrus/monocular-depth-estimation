"""Utility functions for training."""
import os

import matplotlib.pyplot as plt
import numpy as np


def get_lr_scheduler_kwargs(data_dir: str, batch_size: int, accumulate_grad_batches: int) -> dict:
    """Calculates the learning rate scheduler kwargs.

    :param data_dir: Path to the data directory.
    :type data_dir: str
    :param batch_size: Batch size.
    :type batch_size: int
    :param accumulate_grad_batches: Accumulate gradient batches.
    :type accumulate_grad_batches: int
    :return: Learning rate scheduler kwargs.
    :rtype: dict
    """
    number_of_train_samples = len(os.listdir(data_dir))
    t_0 = number_of_train_samples // (batch_size * accumulate_grad_batches)
    lr_scheduler_kwargs = {"T_0": t_0, "T_mult": 3, "eta_min": 1e-07}
    return lr_scheduler_kwargs


def create_color_map(arr: np.array) -> np.ndarray:
    """Creates a color map.

    :param arr: Array.
    :type arr: np.array
    :return: Color map.
    :rtype: np.ndarray
    """
    color_map = []
    for y in arr:
        temp = []
        for x in y:
            temp.append(colors()[int(x)])
        color_map.append(temp)
    color_map = np.array(color_map).astype(np.uint8)
    return color_map


def show(arr: np.ndarray, cmap: str = "tab10") -> None:
    """Shows the image.

    :param arr: Image array.
    :type arr: np.ndarray
    :param cmap: Color map, defaults to "tab10"
    :type cmap: str, optional
    :return: None
    :rtype: None
    """
    figsize = 10, 10
    fig = plt.figure(figsize=figsize)
    plt.imshow(arr, interpolation="none", cmap=cmap)
    plt.colorbar().set_label("cost", labelpad=-45, y=1.025, rotation=0)
    plt.show()


def colors():
    colors = [
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
    ]
    return colors
