"""Utils for models."""
from typing import Optional

from .timm_encoder import TimmUniversalEncoder


def get_encoder(
    name: str, in_channels: int = 3, depth: int = 5, weights: Optional[str] = None, output_stride: int = 32
):
    """Loads encoder by name from timm library.

    :param name: Name of the encoder.
    :type name: str
    :param in_channels: Number of input channels.
    :type in_channels: int
    :param depth: Depth of the encoder.
    :type depth: int
    :param weights: Pretrained weights.
    :type weights: str
    :param output_stride: Output stride of the encoder.
    :type output_stride: int
    :return: Encoder.
    :rtype: nn.Module
    """
    encoder = TimmUniversalEncoder(
        name=name,
        in_channels=in_channels,
        depth=depth,
        output_stride=output_stride,
        pretrained=weights is not None,
    )
    return encoder
