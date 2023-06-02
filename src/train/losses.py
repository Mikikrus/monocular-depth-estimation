"""Loss functions for training."""
from functools import partial
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import functional as f


class L1Loss(nn.Module):
    """
    L1 loss with ignore regions.
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
        :type out: torch.Tensor
        :param label: Ground truth.
        :type label: torch.Tensor
        :return: Loss value.
        :rtype: torch.Tensor
        """

        if self.normalize:
            out = nn.functional.normalize(out, p=2, dim=1)

        mask = label != self.ignore_values
        masked_out = torch.masked_select(out, mask)
        masked_label = torch.masked_select(label, mask)
        return nn.functional.l1_loss(masked_out, masked_label, reduction=self.reduction)


def focal_loss_with_logits(
    output: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
    reduction: str = "mean",
    normalized: bool = False,
    reduced_threshold: Optional[float] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute binary focal loss between target and output logits.

    :param output: Tensor of arbitrary shape (predictions of the model)
    :type output: torch.Tensor
    :param target: Tensor of the same shape as input
    :type target: torch.Tensor
    :param gamma: Focal loss power factor, defaults to 2.0
    :type gamma: float, optional
    :param alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
        high values will give more weight to positive class, defaults to 0.25
    :type alpha: Optional[float], optional
    :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum' | 'batchwise_mean'.
        'none': no reduction will be applied, 'mean': the sum of the output will be divided by the number of
        elements in the output, 'sum': the output will be summed.
    :type reduction: str, optional
    :param normalized: If True, losses are normalized by the number of positive pixels in target
    :type normalized: bool, optional
    :param reduced_threshold: Threshold for reduced focal loss
    :type reduced_threshold: Optional[float], optional
    :param eps: Epsilon to avoid log(0)
    :type eps: float, optional
    :return: Loss value
    :rtype: torch.Tensor
    """

    target = target.type(output.type())

    logpt = f.binary_cross_entropy_with_logits(output, target, reduction="none")
    pt = torch.exp(-logpt)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * logpt

    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)

    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss /= norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: Optional[float] = 2.0,
        ignore_values: Optional[int] = None,
        reduction: Optional[str] = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
    ):
        """Compute focal loss between target and output logits.

        :param alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
            high values will give more weight to positive class.
        :type alpha: Optional[float], optional
        :param gamma: Focal loss power factor, defaults to 2.0
        :type gamma: Optional[float], optional
        :param ignore_values: Values to ignore, defaults to None
        :type ignore_values: Optional[int], optional
        :param reduction: Reduction type, defaults to 'mean'
        :type reduction: Optional[str], optional
        :param normalized: Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf), defaults to False
        :type normalized: bool, optional
        :param reduced_threshold: Compute reduced focal loss (https://arxiv.org/abs/1903.01347), defaults to None
        :type reduced_threshold: Optional[float], optional
        :return: None
        :rtype: None
        """
        super().__init__()

        self.ignore_values = ignore_values
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Forward propagation method for the focal loss.

        :param y_pred: Predicted logits
        :type y_pred: torch.Tensor
        :param y_true: Ground truth labels
        :type y_true: torch.Tensor
        :return: Focal loss
        :rtype: torch.Tensor
        """
        num_classes = y_pred.size(1)
        loss = 0

        # Filter anchors with -1 label from loss computation
        if self.ignore_values is not None:
            not_ignored = y_true != self.ignore_values
            not_ignored = not_ignored.squeeze(1)

        for cls in range(num_classes):
            cls_y_true = (y_true == cls).long().squeeze(1)
            cls_y_pred = y_pred[:, cls, ...]

            if self.ignore_values is not None:
                cls_y_true = cls_y_true[not_ignored]
                cls_y_pred = cls_y_pred[not_ignored]

            loss += self.focal_loss_fn(cls_y_pred, cls_y_true)

        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        ignore_values: Optional[int] = -1,
        dim: int = 1,
    ):
        """Cross entropy loss.

        :param reduction: Reduction type, defaults to 'mean'
        :type reduction: str, optional
        :param ignore_values: Values to ignore, defaults to -1
        :type ignore_values: Optional[int], optional
        :param dim: Dimension to reduce, defaults to 1
        :type dim: int, optional
        :return: None
        :rtype: None
        """
        super().__init__()
        self.ignore_values = ignore_values
        self.reduction = reduction
        self.dim = dim

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Forward propagation method for the cross entropy loss.

        :param y_pred: Predicted logits
        :type y_pred: torch.Tensor
        :param y_true: Ground truth labels
        :type y_true: torch.Tensor
        :return: Cross entropy loss
        :rtype: torch.Tensor
        """

        if y_true.dim() == y_pred.dim():
            y_true = y_true.squeeze(self.dim)

        if self.ignore_values is not None:
            pad_mask = y_true.eq(self.ignore_values)

            y_true = y_true.masked_fill(pad_mask, 0)

            y_pred = y_pred.masked_fill(pad_mask, 0)
        loss = f.cross_entropy(y_pred, y_true, reduction=self.reduction)
        return loss


def dice_loss(true: torch.Tensor, logits: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Computes the Sørensen–Dice loss.

    :param true: a tensor of shape [B, 1, H, W].
    :type true: torch.Tensor
    :param logits: a tensor of shape [B, C, H, W]. Corresponds to
        the raw output or logits of the model.
    :type logits: torch.Tensor
    :param eps: added to the denominator for numerical stability.
    :type eps: float, optional
    :return: dice_loss: the Sørensen–Dice loss.
    :rtype: torch.Tensor
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes).to(logits.device)
        true_1_hot = true_1_hot[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = f.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    _dice_loss = (2.0 * intersection / (cardinality + eps)).mean()
    return 1 - _dice_loss


class DiceLoss(nn.Module):
    def __init__(self, ignore_values: Optional[int] = -1):
        """Dice loss.

        :param ignore_values: Values to ignore, defaults to -1
        :type ignore_values: Optional[int], optional
        :return: None
        :rtype: None
        """
        super().__init__()
        self.ignore_values = ignore_values

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Forward propagation method for the dice loss.

        :param y_pred: Predicted logits
        :type y_pred: torch.Tensor
        :param y_true: Ground truth labels
        :type y_true: torch.Tensor
        :return: Dice loss
        :rtype: torch.Tensor
        """
        if y_pred.dim() != y_true.dim():
            y_true = y_true.unsqueeze(1)
        pad_mask = y_true.eq(self.ignore_values)

        true = y_true.masked_fill(pad_mask, 0)
        pred = y_pred.masked_fill(pad_mask, 0)
        return dice_loss(true, pred)
