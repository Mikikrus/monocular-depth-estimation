"""PyTorch Lightning callbacks. Callbacks are used to perform actions at various events during training and inference.
For example, ModelCheckpoint callback saves the best model based on the validation loss."""
import lightning as pl
import matplotlib.pyplot as plt
import torch
from lightning.pytorch import callbacks

from src import DEVICE


class ModelCheckpoint(callbacks.ModelCheckpoint):
    """Model checkpoint callback. Saves the best model based on the validation loss."""
    def __init__(
        self,
        save_top_k: int = 2,
        monitor: str = "Validation/Mean absolute error",
        mode: str = "min",
        dir_path: str = "/content/checkpoints",
        filename: str = "base-{epoch:02d}-{Validation/Mean absolute error:.2f}",
        **kwargs,
    ) -> None:
        super().__init__(
            save_top_k=save_top_k, monitor=monitor, mode=mode, dirpath=dir_path, filename=filename, **kwargs
        )


class VisualizePrediction(callbacks.Callback):
    """Visualize the prediction and ground truth for the first num_samples in the validation dataset."""

    def __init__(self, num_samples: int = 3) -> None:
        super().__init__()
        self.num_samples = num_samples

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Visualize the prediction and ground truth for the first num_samples in the validation dataset. The images are
        logged to the wandb logger.
        :param trainer: PyTorch Lightning trainer.
        :rtype: pl.Trainer
        :param pl_module: PyTorch Lightning module.
        :rtype: pl.LightningModule
        """
        for prediction_id in range(self.num_samples):
            sample_data = trainer.datamodule.val_dataset[prediction_id]
            image = torch.unsqueeze(sample_data["image"], 0).float().to(DEVICE)
            depth_image = torch.unsqueeze(sample_data["depth_image"], 0).float().to(DEVICE)
            with torch.no_grad():
                prediction = trainer.model.model(image).detach().cpu()
            fig, ax = plt.subplots(1, ncols=2, figsize=(15, 5))
            ax[0].imshow(prediction.squeeze().cpu(), cmap="hot")
            ax[1].imshow(depth_image.squeeze().cpu(), cmap="hot")
            trainer.logger.experiment.log({f"prediction {prediction_id}": fig})
