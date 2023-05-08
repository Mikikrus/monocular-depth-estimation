from typing import Union

import lightning as pl
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchmetrics import MeanAbsoluteError


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        optimizer: str = "Adam",
        learning_rate=1e-4,
        lr_scheduler_params: Union[dict, None] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = getattr(torch.optim, optimizer)
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        self.lr_scheduler = None
        self.mean_absolute_error = MeanAbsoluteError()
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
        loss = self.mean_absolute_error(masked_predictions, masked_ground_truth)
        return loss

    def forward_step(self, batch, batch_idx, state="Train") -> torch.Tensor:
        images = batch["image"]
        depth_images = batch["depth_image"]
        outputs = self.model(images)
        loss = self.calculate_loss(prediction=outputs, ground_truth=depth_images)
        self.log(f"{state}/Mean absolute error", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.forward_step(batch, batch_idx, state="Train")

    def on_fit_start(self):
        pl.seed_everything(0)

    def validation_step(self, batch, batch_idx):
        return self.forward_step(batch, batch_idx, state="Validation")

    def configure_optimizers(self):
        optimizer = self.optimizer(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        self.lr_scheduler = CosineAnnealingWarmRestarts(optimizer, **self.lr_scheduler_params)
        return [optimizer], [{"scheduler": self.lr_scheduler, "interval": "step"}]
