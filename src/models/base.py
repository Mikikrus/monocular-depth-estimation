"""Base classes for all models and heads"""
from typing import Optional

import torch
import torch.nn as nn

from .modules import Activation


def initialize_decoder(module) -> None:
    """Sets initial weights for the decoder module.

    :param module: module to initialize
    :type module: nn.Module
    :return: None
    :rtype: None
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def initialize_head(module) -> None:
    """Sets initial weights for the head module.

    :param module: module to initialize
    :type module: nn.Module
    :return: None
    :rtype: None
    """
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class BaseModel(torch.nn.Module):
    """Base class for all models"""

    def initialize(self) -> None:
        """Initializes model with default parameters

        :return: None
        :rtype: None
        """
        initialize_decoder(self.decoder)
        initialize_head(self.head)

    def check_input_shape(self, x) -> None:
        """Check if input image height and width are divisible by `output_stride`

        :param x: 4D torch tensor with shape (batch_size, channels, height, width)
        :type x: torch.Tensor
        :raises RuntimeError: if input image height or width are not divisible by `output_stride`
        :return: None
        :rtype: None
        """
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
            new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sequentially pass `x` trough model`s encoder, decoder and heads.

        :param x: 4D torch tensor with shape (batch_size, channels, height, width)
        :type x: torch.Tensor
        :return: model output
        :rtype: torch.Tensor
        """

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.head(decoder_output)
        return masks

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        :param x: 4D torch tensor with shape (batch_size, channels, height, width)
        :type x: torch.Tensor
        :return: prediction
        :rtype: torch.Tensor
        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x


class BaseHead(nn.Sequential):
    """Base class for all heads"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation_name: Optional[str] = None,
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
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation_name = Activation(activation_name)
        super().__init__(conv2d, upsampling, activation_name)
