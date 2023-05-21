"""Training module. It contains LightningModel class that is a wrapper around the model that handles forward step,
loss calculation, optimizer and learning rate scheduler."""
from typing import Dict, Union

import lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.utils import CallableObjectProtocol


class LightningModel(pl.LightningModule):
    """Lightning model class. It is a wrapper around the model that handles forward step, loss calculation, optimizer
    and learning rate scheduler."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: str,
        learning_rate: float,
        depth_loss: CallableObjectProtocol,
        segmentation_loss: CallableObjectProtocol,
        lr_scheduler_params: Union[dict, None],
    ) -> None:
        """Initialize LightningModel.

        :param model: Model.
        :type model: nn.Module
        :param optimizer: Name of one of the torch optimizers.
        :type optimizer: str
        :param learning_rate: Learning rate.
        :type learning_rate: float
        :param depth_loss: Loss function used in depth estimation head.
        :type depth_loss: CallableObjectProtocol
        :param segmentation_loss: Loss function used in segmentation head.
        :type segmentation_loss: CallableObjectProtocol
        :param lr_scheduler_params: Parameters for the learning rate scheduler, defaults to None
        :type lr_scheduler_params: Union[dict, None], optional
        :return: None
        :rtype: None
        """
        super().__init__()
        self.model = model
        self.optimizer = getattr(torch.optim, optimizer)
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=["model", "loss"])
        self.lr_scheduler = None
        self.depth_loss = depth_loss
        self.segmentation_loss = segmentation_loss
        self.lr_scheduler_params = lr_scheduler_params

    def calculate_loss(
        self,
        prediction: Dict[str, torch.Tensor],
        depth_ground_truth: torch.Tensor,
        segmentation_ground_truth: torch.Tensor,
        state: str = "Train",
    ) -> torch.FloatTensor:
        """Calculates loss between prediction and ground truth on the pixels.

        :param prediction: Prediction of the model.
        :type prediction: Dict[str,torch.Tensor]
        :param depth_ground_truth: Depth ground truth.
        :type depth_ground_truth: torch.Tensor
        :param segmentation_ground_truth: Segmentation ground truth.
        :type segmentation_ground_truth: torch.Tensor
        :param state: State of the model, defaults to "Train"
        :type state: str, optional
        :return: Loss value.
        :rtype: torch.FloatTensor
        """
        _depth_loss = self.depth_loss(prediction["depth_mask"], depth_ground_truth)
        _segmentation_loss = self.segmentation_loss(prediction["seg_mask"], torch.squeeze(segmentation_ground_truth).long())
        self.log(f"{state}/{self.depth_loss.__class__.__name__}", _depth_loss)
        self.log(f"{state}/{self.segmentation_loss.__class__.__name__}", _segmentation_loss)
        return  _segmentation_loss

    def forward_step(self, batch, batch_idx, state="Train") -> torch.Tensor:
        """Forward step of the model.

        :param batch: Batch of data.
        :type batch: Dict[str, torch.Tensor]
        :param batch_idx: Index of the batch.
        :type batch_idx: int
        :param state: State of the model.
        :type state: str
        :return: Loss value.
        :rtype: torch.Tensor
        """
        images = batch["image"]
        depth_images = batch["depth_image"]
        label = batch["label"]
        outputs = self.model(images)
        loss = self.calculate_loss(
            prediction=outputs, depth_ground_truth=depth_images, segmentation_ground_truth=label, state=state
        )
        return loss

    def training_step(self, batch, batch_idx):
        """Training step of the model.

        :param batch: Batch of data.
        :type batch: Dict[str, torch.Tensor]
        :param batch_idx: Index of the batch.
        :type batch_idx: int
        :return: Loss value.
        :rtype: torch.Tensor
        """
        return self.forward_step(batch, batch_idx, state="Train")

    def on_fit_start(self):
        """Sets seed for reproducibility."""
        pl.seed_everything(0)

    def validation_step(self, batch, batch_idx):
        """Validation step of the model.

        :param batch: Batch of data.
        :type batch: Dict[str, torch.Tensor]
        :param batch_idx: Index of the batch.
        :type batch_idx: int
        :return: Loss value.
        :rtype: torch.Tensor
        """
        return self.forward_step(batch, batch_idx, state="Validation")

    def configure_optimizers(self):
        """Configures optimizers and learning rate schedulers.

        :return: Optimizers and learning rate schedulers.
        :rtype: List[torch.optim.Optimizer], List[Dict[str, Any]]
        """
        optimizer = self.optimizer(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        self.lr_scheduler = CosineAnnealingWarmRestarts(optimizer, **self.lr_scheduler_params)
        return [optimizer], [{"scheduler": self.lr_scheduler, "interval": "step"}]
