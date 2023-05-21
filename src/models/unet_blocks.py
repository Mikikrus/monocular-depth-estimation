"""Module containing building blocks used for model definition."""

from typing import Callable, Optional, Union

import torch
from torch import nn

import src.models.modules as md


class DownBlock(nn.Module):
    """Downscaling with max pooling followed by the 3x3 convolution and attention block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: int,
        use_batchnorm: bool = True,
        attention_type: Optional[str] = None,
    ) -> None:
        """Initialize DownBlock.

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param scale: downscaling factor
        :type scale: int
        :param use_batchnorm: if True, use BatchNorm2d before ReLU activation, defaults to True
        :type use_batchnorm: bool, optional
        :param attention_type: attention block type, defaults to None
        :type attention_type: Optional[str], optional
        :return: None
        :rtype: None
        """
        super().__init__()
        self.maxpool = nn.MaxPool2d(scale, scale, ceil_mode=True)

        self.conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param features: 4D torch tensor with shape (batch_size, channels, height, width)
        :type features: torch.Tensor
        :return: output tensor
        :rtype: torch.Tensor
        """
        output = self.maxpool(features)
        output = self.conv1(output)
        output = self.attention1(output)
        return output


class ConstBlock(nn.Module):
    """Convolutional block with kernel size 3x3 with padding 1."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: Optional[int] = None,
        use_batchnorm: bool = True,
        attention_type: Optional[Union[str, Callable]] = None,
    ):
        """Initialize ConstBlock.

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param scale: downscaling factor, defaults to None
        :type scale: Optional[int], optional
        :param use_batchnorm: if True, use BatchNorm2d before ReLU activation, defaults to True
        :type use_batchnorm: bool, optional
        :param attention_type: attention block type, defaults to None
        :type attention_type: Optional[Union[str, Callable]], optional
        :return: None
        :rtype: None
        """
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param features: 4D torch tensor with shape (batch_size, channels, height, width)
        :type features: torch.Tensor
        :return: output tensor
        :rtype: torch.Tensor
        """
        output = self.conv1(features)
        return output


class UpBlock(nn.Module):
    """Up-scaling with bi-linear interpolation followed by the 3x3 convolution and attention block."""

    def __init__(
        self,
        in_channels,
        out_channels,
        scale,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale, mode="bilinear")

        self.conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param features: input tensor
        :type features: torch.Tensor
        :return: output tensor
        :rtype: torch.Tensor
        """
        output = self.upsample(features)
        output = self.conv1(output)
        output = self.attention1(output)
        return output


class CenterBlock(nn.Sequential):
    """Center block. This block does not change the size of the signal, only the number of features is changed."""

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool = True) -> None:
        """Initialize CenterBlock.

        :param in_channels: number of input channels
        :type in_channels: int
        :param out_channels: number of output channels
        :type out_channels: int
        :param use_batchnorm: if True, use BatchNorm2d before ReLU activation, defaults to True
        :type use_batchnorm: bool, optional
        :return: None
        :rtype: None
        """
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)
