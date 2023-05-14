"""Modules for building neural network models."""
from typing import Optional, Union

import torch
import torch.nn as nn


class Conv2dReLU(nn.Sequential):
    """Sequentially applies Convolution and ReLU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        use_batchnorm: bool = True,
    ):
        """Initialize Conv2dReLU module.
        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param kernel_size: convolution kernel size
        :type kernel_size: int
        :param padding: convolution padding, defaults to 0
        :type padding: int, optional
        :param stride: convolution stride, defaults to 1
        :type stride: int, optional
        :param use_batchnorm: if True, use BatchNorm2d before ReLU activation, defaults to True
        :type use_batchnorm: bool, optional
        :return: None
        :rtype: None"""
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SCSEModule(nn.Module):
    """Squeeze and Excitation module for 2D inputs. Described in https://arxiv.org/pdf/1803.02579.pdf"""

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        """Initialize SCSEModule.
        :param in_channels: number of input channels
        :type in_channels: int
        :param reduction: reduction ratio, defaults to 16
        :type reduction: int, optional
        :return: None
        :rtype: None
        """
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        :param x: 4D torch tensor with shape (batch_size, channels, height, width)
        :type x: torch.Tensor
        :return: Features multiplied by channel and spatial attention tensors.
        :rtype: torch.Tensor
        """
        return x * self.cSE(x) + x * self.sSE(x)


class ArgMax(nn.Module):
    """ArgMax module."""

    def __init__(self, dim: Optional[int] = None) -> None:
        """Initialize ArgMax module.
        :param dim: dimension to apply argmax, defaults to None
        :type dim: Optional[int], optional
        :return: None
        :rtype: None
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        :param x: 4D torch tensor with shape (batch_size, channels, height, width)
        :type x: torch.Tensor
        :return: Features with applied argmax along the specified dimension
        :rtype: torch.Tensor
        """
        return torch.argmax(x, dim=self.dim)


class Clamp(nn.Module):
    """Clamp module."""

    def __init__(self, min: Union[int, float] = 0, max: Union[int, float] = 1) -> None:
        """Initialize Clamp module.
        :param min: minimum value, defaults to 0
        :type min: Union[int, float], optional
        :param max: maximum value, defaults to 1
        :type max: Union[int, float], optional
        :return: None
        :rtype: None
        """
        super().__init__()
        self.min, self.max = min, max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        :param x: 4D torch tensor with shape (batch_size, channels, height, width)
        :type x: torch.Tensor
        :return: Features with applied clamp
        :rtype: torch.Tensor
        """
        return torch.clamp(x, self.min, self.max)


class Activation(nn.Module):
    """Activation module."""

    def __init__(self, name: str, **params) -> None:
        """Initialize Activation module.
        :param name: name of the activation function
        :type name: str
        :param params: parameters for the activation function
        :type params: dict
        :return: None
        :rtype: None
        """
        super().__init__()

        if name is None or name == "identity":
            self.activation = nn.Identity(**params)
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax2d":
            self.activation = nn.Softmax(dim=1, **params)
        elif name == "softmax":
            self.activation = nn.Softmax(**params)
        elif name == "logsoftmax":
            self.activation = nn.LogSoftmax(**params)
        elif name == "tanh":
            self.activation = nn.Tanh()
        elif name == "argmax":
            self.activation = ArgMax(**params)
        elif name == "argmax2d":
            self.activation = ArgMax(dim=1, **params)
        elif name == "clamp":
            self.activation = Clamp(**params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError(
                f"Activation should be callable/sigmoid/softmax/logsoftmax/tanh/"
                f"argmax/argmax2d/clamp/None; got {name}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        :param x: 4D torch tensor with shape (batch_size, channels, height, width)
        :type x: torch.Tensor
        :return: Features with applied activation function
        :rtype: torch.Tensor
        """
        return self.activation(x)


class Attention(nn.Module):
    """Attention module."""

    def __init__(self, name: str, **params) -> None:
        """Initialize Attention module.
        :param name: name of the attention module
        :type name: str
        :param params: parameters for the attention module
        :type params: dict
        :return: None
        :rtype: None
        """
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        :param x: 4D torch tensor with shape (batch_size, channels, height, width)
        :type x: torch.Tensor
        :return: Features with applied attention module
        :rtype: torch.Tensor
        """
        return self.attention(x)
