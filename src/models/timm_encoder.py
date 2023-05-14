"""Timm universal encoder. Loads encoder by name from timm library."""
import timm
import torch
import torch.nn as nn


class TimmUniversalEncoder(nn.Module):
    """Timm universal encoder. Loads encoder by name from timm library."""

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        in_channels: int = 3,
        depth: int = 5,
        output_stride: int = 32,
    ) -> None:
        """Initialize TimmUniversalEncoder.

        :param name: name of timm model
        :type name: str
        :param pretrained: load pretrained weights, defaults to True
        :type pretrained: bool, optional
        :param in_channels: number of input channels, defaults to 3
        :type in_channels: int, optional
        :param depth: number of output feature maps, defaults to 5
        :type depth: int, optional
        :param output_stride: output stride, defaults to 32
        :type output_stride: int, optional
        :return: None
        :rtype: None
        """
        super().__init__()
        kwargs = dict(
            in_chans=in_channels,
            features_only=True,
            output_stride=output_stride,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
        )

        # not all models support output stride argument, drop it by default
        if output_stride == 32:
            kwargs.pop("output_stride")

        self.model = timm.create_model(name, **kwargs)

        self._in_channels = in_channels
        self._out_channels = [
            in_channels,
        ] + self.model.feature_info.channels()
        self._depth = depth
        self._output_stride = output_stride

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass.

        :param x: Input tensor.
        :rtype x: torch.Tensor
        :return: Features.
        :rtype: list[torch.Tensor]"""
        features = self.model(x)
        features = [
            x,
        ] + features
        return features

    @property
    def out_channels(self) -> list[int]:
        """Output channels.

        :return: Output channels.
        :rtype: list[int]
        """
        return self._out_channels

    @property
    def output_stride(self) -> int:
        """Output stride.

        :return: Output stride.
        :rtype: int
        """
        return min(self._output_stride, 2**self._depth)
