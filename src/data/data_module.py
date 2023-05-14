"""Pytorch lightning data module for depth estimation."""
from typing import Any, Optional

import lightning as pl
from torch.utils.data import DataLoader

from .dataset import DepthEstimationDataset


class DepthEstimationDataModule(pl.LightningDataModule):
    """Data module for depth estimation."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int = 2,
        persistent_workers: bool = False,
        transforms: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.transforms = transforms
        self.train_subset: Optional[DepthEstimationDataset] = None
        self.val_subset: Optional[DepthEstimationDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup data module.

        :param stage: stage of the data module
        :type stage: Optional[str]
        :return: None
        :rtype: None
        """
        self.train_subset = DepthEstimationDataset(self.data_dir, split="train", transforms=self.transforms)
        self.val_subset = DepthEstimationDataset(self.data_dir, split="val", transforms=self.transforms)

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader.

        :return: train dataloader
        :rtype: DataLoader
        """
        return DataLoader(
            self.train_subset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader.

        :return: validation dataloader
        :rtype: DataLoader
        """
        return DataLoader(
            self.val_subset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )
