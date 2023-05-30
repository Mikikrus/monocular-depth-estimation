"""Loss functions for training."""
from typing import Optional, Union
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F
from functools import partial
import torch
from torch import nn


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
        n_valid = torch.sum(mask).item()
        masked_out = torch.masked_select(out, mask)
        masked_label = torch.masked_select(label, mask)
        if self.reduction == "mean":
            return nn.functional.l1_loss(masked_out, masked_label, reduction=self.reduction) / max(n_valid, 1)
        elif self.reduction == "sum":
            return nn.functional.l1_loss(masked_out, masked_label, reduction=self.reduction)
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
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.

    Args:
        output: Tensor of arbitrary shape (predictions of the model)
        target: Tensor of the same shape as input
        gamma: Focal loss power factor
        alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
            high values will give more weight to positive class.
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).

    References:
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type(output.type())

    logpt = F.binary_cross_entropy_with_logits(output, target, reduction="none")
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


class FocalLoss(_Loss):
    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: Optional[float] = 2.0,
        ignore_index: Optional[int] = None,
        reduction: Optional[str] = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
    ):
        """Compute Focal loss

        Args:
            alpha: Prior probability of having positive value in target.
            gamma: Power factor for dampening weight (focal strength).
            ignore_index: If not None, targets may contain values to be ignored.
                Target values equal to ignore_index will be ignored from loss computation.
            normalized: Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
            reduced_threshold: Switch to reduced focal loss. Note, when using this mode you
                should use `reduction="sum"`.

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt

        """
        super().__init__()

        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        num_classes = y_pred.size(1)
        loss = 0

        # Filter anchors with -1 label from loss computation
        if self.ignore_index is not None:
            not_ignored = y_true != self.ignore_index

        for cls in range(num_classes):
            cls_y_true = (y_true == cls).long()
            cls_y_pred = y_pred[:, cls, ...]

            if self.ignore_index is not None:
                cls_y_true = cls_y_true[not_ignored]
                cls_y_pred = cls_y_pred[not_ignored]

            loss += self.focal_loss_fn(cls_y_pred, cls_y_true)

        return loss

def label_smoothed_nll_loss(
    lprobs: torch.Tensor,
    target: torch.Tensor,
    epsilon: float,
    ignore_index=None,
    reduction="mean",
    dim=-1,
) -> torch.Tensor:
    """NLL loss with label smoothing

    References:
        https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py

    Args:
        lprobs (torch.Tensor): Log-probabilities of predictions (e.g after log_softmax)

    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(dim)
    # print('=======================')
    # print(target.shape, lprobs.shape)
    # print('=======================')

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0)
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        # nll_loss.masked_fill_(pad_mask, 0.0)
        # smooth_loss.masked_fill_(pad_mask, 0.0)
        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)

    if reduction == "sum":
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == "mean":
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()

    eps_i = epsilon / lprobs.size(dim)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss

class SoftCrossEntropyLoss(nn.Module):

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(
        self,
        reduction: str = "mean",
        smooth_factor: Optional[float] = 0.1,
        ignore_index: Optional[int] = -100,
        dim: int = 1,
    ):
        """Drop-in replacement for torch.nn.CrossEntropyLoss with label_smoothing

        Args:
            smooth_factor: Factor to smooth target (e.g. if smooth_factor=0.1 then [1, 0, 0] -> [0.9, 0.05, 0.05])

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(y_pred, dim=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            y_true,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )
