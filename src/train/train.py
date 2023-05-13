"""Training module. It contains LightningModel class that is a wrapper around the model that handles forward step,
loss calculation, optimizer and learning rate scheduler."""
from typing import Any, Protocol, Union

import lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


class CallableObjectProtocol(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


class LightningModel(pl.LightningModule):
    """Lightning model class. It is a wrapper around the model that handles forward step, loss calculation, optimizer
    and learning rate scheduler."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: str,
        learning_rate,
        loss: CallableObjectProtocol,
        lr_scheduler_params: Union[dict, None],
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = getattr(torch.optim, optimizer)
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=["model", "loss"])
        self.lr_scheduler = None
        self.loss = loss
        self.lr_scheduler_params = lr_scheduler_params

    def calculate_loss(self, prediction: torch.Tensor, ground_truth: torch.Tensor) -> torch.FloatTensor:
        """Calculates loss between prediction and ground truth on the pixels where ground truth is greater than zero.
        :param prediction: Prediction of the model.
        :rtype: torch.Tensor
        :param ground_truth: Ground truth.
        :rtype: torch.Tensor
        :return: Loss value.
        """
        mask = ground_truth > 0
        masked_predictions = torch.masked_select(prediction, mask)
        masked_ground_truth = torch.masked_select(ground_truth, mask)
        loss = self.loss(masked_predictions, masked_ground_truth)
        return loss

    def forward_step(self, batch, batch_idx, state="Train") -> torch.Tensor:
        """Forward step of the model.
        :param batch: Batch of data.
        :rtype: Dict[str, torch.Tensor]
        :param batch_idx: Index of the batch.
        :rtype: int
        :param state: State of the model.
        :rtype: str
        :return: Loss value.
        :rtype: torch.Tensor
        """
        images = batch["image"]
        depth_images = batch["depth_image"]
        outputs = self.model(images)
        loss = self.calculate_loss(prediction=outputs, ground_truth=depth_images)
        self.log(f"{state}/{self.loss.__class__.__name__}", loss)
        return loss

    def training_step(self, batch, batch_idx):
        """Training step of the model.
        :param batch: Batch of data.
        :rtype: Dict[str, torch.Tensor]
        :param batch_idx: Index of the batch.
        :rtype: int
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
        :rtype: Dict[str, torch.Tensor]
        :param batch_idx: Index of the batch.
        :rtype: int
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
