"""Module containing UNet3Plus model definition."""

from typing import Callable, List, Optional, Union

import torch
from torch import nn

import src.models.modules as md
from src.models.base import BaseHead, BaseModel
from src.models.utils import get_encoder


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: 4D torch tensor with shape (batch_size, channels, height, width)
        :rtype x: torch.Tensor
        :return: output tensor
        :rtype: torch.Tensor
        """
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.attention1(x)
        return x


class ConstBlock(nn.Module):
    """Convolutional block with kernel size 3x3 with padding 1."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: Optional[int] = None,
        use_batchnorm: bool = True,
        attention_type: Optional[str] = None,
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
        :type attention_type: Optional[str], optional
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: 4D torch tensor with shape (batch_size, channels, height, width)
        :rtype x: torch.Tensor
        :return: output tensor
        :rtype: torch.Tensor
        """
        x = self.conv1(x)
        return x


class UpBlock(nn.Module):
    """Upscaling with bilinear interpolation followed by the 3x3 convolution and attention block."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param x: input tensor
        :rtype x: torch.Tensor
        :return: output tensor
        :rtype: torch.Tensor
        """
        x = self.upsample(x)
        x = self.conv1(x)
        x = self.attention1(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block. Performs upscaling, concatenation with encoder output, 3x3 convolution and attention block."""

    def __init__(
        self,
        in_channels: List[int],
        out_channels: List[int],
        previous_channels: list,
        max_down_scale: int,
        num_concat_blocks: int,
        output_index: int,
        use_batchnorm: bool = True,
        attention_type: Optional[str] = None,
    ) -> None:
        """Initialize DecoderBlock.

        :param in_channels: list of encoder's output channel sizes
        :type in_channels: List[int]
        :param out_channels: list of decoders output channel sizes
        :type out_channels: List[int]
        :param previous_channels: list of previous output channels
        :type previous_channels: list
        :param max_down_scale: maximum down scaling to be done
        :type max_down_scale: int
        :param num_concat_blocks: no. of blocks to be concatenated
        :type num_concat_blocks: int
        :param output_index: index of output channels for each block
        :type output_index: int
        :param use_batchnorm: if True, use BatchNorm2d before ReLU activation, defaults to True
        :type use_batchnorm: bool, optional
        :param attention_type: attention block type, defaults to None
        :type attention_type: Optional[str], optional
        :return: None
        :rtype: None
        """
        super().__init__()

        scale = max_down_scale
        blocks = []
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        self.num_concat_blocks = num_concat_blocks
        self.up_channel = out_channels[output_index] * self.num_concat_blocks

        # pos = len(in_channels) - scale - 1

        prev_ind = 1  # used for calculating index of input channels for UpBlock

        # creating blocks with respect to the scale
        for i in range(self.num_concat_blocks):
            if scale > 0:
                block = DownBlock(in_channels[i], out_channels[output_index], pow(2, scale), **kwargs)
            elif scale == 0:
                block = ConstBlock(in_channels[i], out_channels[output_index], 0, **kwargs)
            else:
                block = UpBlock(
                    previous_channels[output_index - prev_ind], out_channels[output_index], pow(2, abs(scale)), **kwargs
                )
                prev_ind += 1

            blocks.append(block)
            scale = scale - 1

        self.blocks = nn.ModuleList(blocks)

        # concatenation block
        self.cat_block = md.Conv2dReLU(
            self.up_channel,
            self.up_channel,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention_cat = md.Attention(attention_type, in_channels=self.up_channel)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        :param feature: 4D torch tensor with shape (batch_size, channels, height, width)
        :rtype feature: torch.Tensor
        :return: output tensor
        :rtype: torch.Tensor
        """
        result_list = []
        for i, block in enumerate(self.blocks):
            result = block(feature[i])
            result_list.append(result)

        concat_tensor = torch.cat(result_list, 1)  # concatenation of every tensor
        final = self.cat_block(concat_tensor)
        final = self.attention_cat(final)

        return final


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


class Unet3PlusDecoder(nn.Module):
    """Decoder block implementing decoder with full-scale skip connections."""

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        n_blocks: int = 5,
        use_batchnorm: bool = True,
        attention_type: Optional[str] = None,
        center: bool = False,
    ) -> None:
        """Initialize Unet3PlusDecoder.

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

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = encoder_channels[1:]
        in_channels = in_channels[::-1]  # reversing the list

        out_channels = [head_channels] + list(decoder_channels)
        previous_channels = [
            i * n_blocks for i in out_channels
        ]  # calculating previous channels i.e. concatenated output channels of previous block
        previous_channels[0] = int(
            previous_channels[0] / n_blocks
        )  # first tensor as it comes from CenterBlock, hence no concatenation

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        stages = []
        self.scale = n_blocks - 2

        # Blocks for every stage is generated
        for i in range(n_blocks):
            stage = DecoderBlock(
                in_channels, out_channels, previous_channels, self.scale - i, n_blocks, i + 1, **kwargs
            )
            stages.append(stage)

        self.stages = nn.ModuleList(stages)

    def forward(self, *features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        :param features: input tensors
        :rtype features: List[torch.Tensor]
        :return: output tensor
        :rtype: torch.Tensor
        """
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        current_features = features[1:]
        current_features = current_features[::-1]
        features_in_scope = list(current_features)

        x = self.center(head)  # first tensor is input into center block

        decoded_features = [x]

        # tensors are calculated for each stage
        for i, stage in enumerate(self.stages):
            total_features = features_in_scope.copy()
            total_features.extend(decoded_features)
            x = stage(total_features)

            features_in_scope = features_in_scope[:-1]

            decoded_features = [x] + decoded_features

        return x


class Unet3Plus(BaseModel):
    """Implementation of UNet 3+ model."""

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: tuple = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, Callable]] = None,  # type: ignore
    ) -> None:
        """Initialize Unet3Plus.

        :param encoder_name: name of classification model (without last dense layers) used as feature
            extractor to build final model.
        :type encoder_name: str
        :param encoder_depth: number of stages used in decoder, larger depth - more features are generated.
        :type encoder_depth: int
        :param encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        :type encoder_weights: str
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
            weights=encoder_weights,
        )

        self.decoder = Unet3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=bool(encoder_name.startswith("vgg")),
            attention_type=decoder_attention_type,
        )

        self.head = BaseHead(
            in_channels=decoder_channels[encoder_depth - 1] * encoder_depth,
            out_channels=classes,
            activation_name=activation,
            kernel_size=3,
        )

        self.name = f"unet3plus-{encoder_name}"
        self.initialize()
