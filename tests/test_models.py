"""Tests for models.py module."""
from typing import List

import pytest
import torch

from src import models

MODELS = [models.Unet, models.Unet3Plus]
ENCODERS: List[str] = ["tf_efficientnetv2_m.in21k_ft_in1k", "maxxvit_rmlp_nano_rw_256.sw_in1k"]
INPUTS: List[torch.Tensor] = [
    torch.ones([2, 3, 256, 256], dtype=torch.float),
    torch.ones([2, 3, 128, 256], dtype=torch.float),
]
NUMBER_OF_CLASSES: List[int] = [19]


def _test_forward(model: torch.nn.Module, sample: torch.Tensor, classes: int, test_shape: bool = False) -> None:
    """Tests forward method

    :param model: Model object
    :rtype model: torch.nn.Module
    :param sample: Tensor containing sample data
    :rtype sample: torch.Tensor
    :param test_shape: Check whether output of the model is of the same shape as input
    :rtype test_shape: bool
    """
    with torch.no_grad():
        out = model(sample)
    if test_shape:
        assert out["depth_mask"].shape[2:] == sample.shape[2:]
        assert out["seg_mask"].shape[1:] == (classes, *sample.shape[2:])


@pytest.mark.parametrize("model_class", MODELS)
@pytest.mark.parametrize("sample", INPUTS)
@pytest.mark.parametrize("encoder_name", ENCODERS)
@pytest.mark.parametrize("classes", NUMBER_OF_CLASSES)
def test_model(model_class: torch.nn.Module, sample: torch.Tensor, encoder_name: str, classes: int) -> None:
    """Test whether forward method works for a given model.

    :param model_class: Model class
    :type model_class: torch.nn.Module
    :param sample: Tensor containing sample data
    :type sample: torch.Tensor
    :param encoder_name: Name of the model used as an encoder
    :type encoder_name: str
    :param classes: Number of classes
    :type classes: int
    """
    model = model_class(encoder_name=encoder_name, decoder_attention_type="scse", classes=classes)
    model.eval()
    _test_forward(model, sample, classes, test_shape=True)


if __name__ == "__main__":
    pytest.main([__file__])
