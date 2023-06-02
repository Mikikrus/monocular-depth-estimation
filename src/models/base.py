"""Base classes for all models and heads"""
from typing import Callable, Dict, Optional, Union

import torch
from torch import nn

from .modules import Activation


def initialize_decoder(module) -> None:
    """Sets initial weights for the decoder module.

    :param module: module to initialize
    :type module: nn.Module
    :return: None
    :rtype: None
    """
    for _module in module.modules():
        if isinstance(_module, nn.Conv2d):
            nn.init.kaiming_uniform_(_module.weight, mode="fan_in", nonlinearity="relu")
            if _module.bias is not None:
                nn.init.constant_(_module.bias, 0)

        elif isinstance(_module, nn.BatchNorm2d):
            nn.init.constant_(_module.weight, 1)
            nn.init.constant_(_module.bias, 0)

        elif isinstance(_module, nn.Linear):
            nn.init.xavier_uniform_(_module.weight)
            if _module.bias is not None:
                nn.init.constant_(_module.bias, 0)


def initialize_head(module) -> None:
    """Sets initial weights for the head module.

    :param module: module to initialize
    :type module: nn.Module
    :return: None
    :rtype: None
    """
    for _module in module.modules():
        if isinstance(_module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(_module.weight)
            if _module.bias is not None:
                nn.init.constant_(_module.bias, 0)


class BaseModel(torch.nn.Module):
    """Base class for all models"""

    def initialize(self) -> None:
        """Initializes model with default parameters

        :return: None
        :rtype: None
        """
        initialize_decoder(self.depth_decoder)
        initialize_decoder(self.seg_decoder)
        initialize_head(self.seg_head)
        initialize_head(self.depth_head)

    def check_input_shape(self, sample: torch.Tensor) -> None:
        """Check if input image height and width are divisible by `output_stride`

        :param sample: 4D torch tensor with shape (batch_size, channels, height, width)
        :type sample: torch.Tensor
        :raises RuntimeError: if input image height or width are not divisible by `output_stride`
        :return: None
        :rtype: None
        """
        height, width = sample.shape[-2:]
        output_stride: int = self.encoder.output_stride
        if height % output_stride != 0 or width % output_stride != 0:
            new_h = (height // output_stride + 1) * output_stride if height % output_stride != 0 else height
            new_w = (width // output_stride + 1) * output_stride if width % output_stride != 0 else width
            raise RuntimeError(
                f"Wrong input shape height={height}, width={width}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, sample: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Sequentially pass `x` trough model`s encoder, decoder and heads.

        :param sample: 4D torch tensor with shape (batch_size, channels, height, width)
        :type sample: torch.Tensor
        :return: model output
        :rtype: Dict[str,torch.Tensor]
        """

        self.check_input_shape(sample)

        features = self.encoder(sample)
        depth_decoder_output = self.depth_decoder(*features)
        seg_decoder_output = self.seg_decoder(*features)

        depth_mask = self.depth_head(depth_decoder_output)
        seg_mask = self.seg_head(seg_decoder_output)
        return {"depth_mask": depth_mask, "seg_mask": seg_mask}

    @torch.no_grad()
    def predict(self, sample: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        :param sample: 4D torch tensor with shape (batch_size, channels, height, width)
        :type sample: torch.Tensor
        :return: prediction
        :rtype: Dict[str,torch.Tensor]
        """
        if self.training:
            self.eval()

        output = self.forward(sample)

        return output


class BaseHead(nn.Sequential):
    """Base class for all heads"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation_name: Optional[Union[str, Callable]] = None,
        upsampling: int = 1,
    ) -> None:
        """Initialize head module.

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param kernel_size: convolution kernel size, defaults to 3
        :type kernel_size: int, optional
        :param activation_name: name of the activation function, defaults to None
        :type activation_name: Optional[str], optional
        :param upsampling: upsampling factor, defaults to 1
        :type upsampling: int, optional
        :return: None
        :rtype: None
        """
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling_block = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation_name = Activation(activation_name)
        super().__init__(conv2d, upsampling_block, activation_name)
