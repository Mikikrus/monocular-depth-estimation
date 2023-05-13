from typing import List

import pytest
import torch

from src import models

MODELS = [models.Unet3Plus]
ENCODERS: List[str] = ["tf_efficientnetv2_m.in21k_ft_in1k"]


def _test_forward(model: torch.nn.Module, sample: torch.Tensor, test_shape: bool = False) -> None:
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
        assert out.shape[2:] == sample.shape[2:]


@pytest.mark.parametrize("model_class", MODELS)
@pytest.mark.parametrize("encoder_name", ENCODERS)
def test_model(model_class: torch.nn.Module, encoder_name: str) -> None:
    """Test whether forward method works for a given model.
    :param model_class: Model class
    :rtype model_class: torch.nn.Module
    :param encoder_name: Name of the model used as an encoder
    :rtype encoder_name: str
    """
    sample = torch.ones([2, 3, 128, 256], dtype=torch.float)
    model = model_class(encoder_name=encoder_name,decoder_attention_type="scse")
    model.eval()
    _test_forward(model, sample, test_shape=True)


if __name__ == "__main__":
    pytest.main([__file__])
