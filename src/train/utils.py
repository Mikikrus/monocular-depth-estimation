"""Utility functions for training."""
import os


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
