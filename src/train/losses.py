"""Loss functions for training."""
from typing import Union

import torch
from torch import nn


class L1Loss(nn.Module):
    """
    L1 loss with ignore regions.
    normalize: normalization for surface normals
    """

    def __init__(self, normalize: bool = False, ignore_values: Union[int, float] = 0, reduction: str = "mean") -> None:
        """Initialize L1Loss.
        :param normalize: Normalize surface normals, defaults to False
        :type normalize: bool, optional
        :param ignore_values: Values to ignore, defaults to 0
        :type ignore_values: Union[int,float], optional
        :param reduction: Reduction type, defaults to 'mean'
        :type reduction: str, optional
        :return: None
        :rtype: None
        """
        super().__init__()
        self.normalize = normalize
        self.ignore_values = ignore_values
        self.reduction = reduction

    def forward(self, out: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Forward step of the loss.
        :param out: Output of the model.
        :rtype: torch.Tensor
        :param label: Ground truth.
        :rtype: torch.Tensor
        :return: Loss value.
        :rtype: torch.Tensor
        """

        if self.normalize:
            out = nn.functional.normalize(out, p=2, dim=1)

        mask = (label != self.ignore_values).all(dim=1, keepdim=True)
        n_valid = torch.sum(mask).item()
        masked_out = torch.masked_select(out, mask)
        masked_label = torch.masked_select(label, mask)
        if self.reduction == "mean":
            return nn.functional.l1_loss(masked_out, masked_label, reduction=self.reduction) / max(n_valid, 1)
        elif self.reduction == "sum":
            return nn.functional.l1_loss(masked_out, masked_label, reduction=self.reduction)
        elif self.reduction == "none":
            return nn.functional.l1_loss(masked_out, masked_label, reduction=self.reduction)
