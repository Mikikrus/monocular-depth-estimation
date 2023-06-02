"""Module containing UNet model definition."""

from typing import Callable, List, Optional, Union

import torch
import torch.nn.functional as f
from torch import nn

import src.models.modules as md
from src.models.base import BaseHead, BaseModel
from src.models.unet_blocks import CenterBlock
from src.models.utils import get_encoder


class DecoderBlock(nn.Module):
    """Decoder block. Performs up-scaling, concatenation with encoder output, 3x3 convolution and attention block."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        attention_type: Optional[str] = None,
    ) -> None:
        """Initialize DecoderBlock.

        :param in_channels: Encoder's output channel size
        :type in_channels: int
        :param skip_channels: Encoder's skip connection channel size
        :type skip_channels: int
        :param out_channels: Decoder's output channel size
        :type out_channels: int
        :param use_batchnorm: if True, use BatchNorm2d before ReLU activation, defaults to True
        :type use_batchnorm: bool, optional
        :param attention_type: attention block type, defaults to None
        :type attention_type: Optional[str], optional
        :return: None
        :rtype: None
        """
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, features: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        :param features: 4D torch tensor with shape (batch_size, channels, height, width)
        :type features: torch.Tensor
        :param skip: 4D torch tensor with shape (batch_size, channels, height, width), defaults to None
        :type skip: Optional[torch.Tensor], optional
        :return: output tensor
        :rtype: torch.Tensor
        """
        output = f.interpolate(features, scale_factor=2, mode="nearest")
        if skip is not None:
            output = torch.cat([output, skip], dim=1)
            output = self.attention1(output)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.attention2(output)
        return output


class UnetDecoder(nn.Module):
    """Decoder block implementing decoder with skip connections."""

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        n_blocks: int = 5,
        use_batchnorm: bool = True,
        attention_type: Optional[str] = None,
        center: bool = False,
    ) -> None:
        """Initialize UnetDecoder.

        :param encoder_channels: list of encoder's output channel sizes
        :type encoder_channels: List[int]
        :param decoder_channels: list of decoders output channel sizes
        :type decoder_channels: List[int]
        :param n_blocks: number of blocks to be concatenated, defaults to 5
        :type n_blocks: int, optional
        :param use_batchnorm: if True, use BatchNorm2d before ReLU activation, defaults to True
        :type use_batchnorm: bool, optional
        :param attention_type: attention block type, defaults to None
        :type attention_type: Optional[str], optional
        :param center: if True, use CenterBlock, defaults to False
        :type center: bool, optional
        :return: None
        :rtype: None
        """
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"Model depth is {n_blocks}, but you provide `decoder_channels` for {len(decoder_channels)} blocks."
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, use_batchnorm=use_batchnorm, attention_type=attention_type)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        :param features: input tensors
        :type features: List[torch.Tensor]
        :return: output tensor
        :rtype: torch.Tensor
        """
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class Unet(BaseModel):
    """Implementation of UNet model."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        decoder_use_batchnorm: bool = True,
        decoder_channels: tuple = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 19,
        activation: Optional[Union[str, Callable]] = None,
    ) -> None:
        """Initialize Unet.

        :param encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build final model.
        :type encoder_name: str
        :param encoder_depth: number of stages used in decoder, larger depth - more features are generated.
        :type encoder_depth: int
        :param decoder_use_batchnorm: if ``True``, ``BatchNormalisation`` layer between ``Conv2D`` and ``Activation``
            layers is used, ``False`` otherwise.
        :type decoder_use_batchnorm: bool
        :param decoder_channels: a number of convolution layer filters in decoder blocks
        :type decoder_channels: tuple
        :param decoder_attention_type: attention block type, one of ``None``, ``scse``
        :type decoder_attention_type: str
        :param in_channels: number of input channels for model, default is 3
        :type in_channels: int
        :param classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        :type classes: int
        :param activation: activation function used in the model's head
        :type activation: str or callable
        :return: None
        :rtype: None
        """

        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            pretrained=True,
        )

        decoder_kwargs = {
            "encoder_channels": self.encoder.out_channels,
            "decoder_channels": list(decoder_channels),
            "n_blocks": encoder_depth,
            "use_batchnorm": decoder_use_batchnorm,
            "attention_type": decoder_attention_type,
        }

        self.depth_decoder = UnetDecoder(**decoder_kwargs)
        self.seg_decoder = UnetDecoder(**decoder_kwargs)

        self.depth_head = BaseHead(
            in_channels=decoder_channels[encoder_depth - 1],
            out_channels=1,
            activation_name=activation,
            kernel_size=3,
        )
        self.seg_head = BaseHead(
            in_channels=decoder_channels[encoder_depth - 1],
            out_channels=classes,
            activation_name=None,
            kernel_size=3,
        )
        self.name = f"unet-{encoder_name}"
        self.initialize()
