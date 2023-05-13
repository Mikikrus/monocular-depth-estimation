import os

import numpy as np
import torch
from torch.utils.data import Dataset


class DepthEstimationDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", transforms=None) -> None:
        self.data_dir = data_dir
        self.split = split
        self.transforms = transforms
        self.image_dir = os.path.join(self.data_dir, self.split, "image")
        self.depth_dir = os.path.join(self.data_dir, self.split, "depth")
        self.label_dir = os.path.join(self.data_dir, self.split, "label")
        self.image_files = sorted(os.listdir(self.image_dir))
        self.depth_files = sorted(os.listdir(self.depth_dir))
        self.label_files = sorted(os.listdir(self.label_dir))

    def check_ordering(self) -> None:
        """
        Check that the ordering of the images and labels is the same.
        :raises AssertionError: if the ordering is not the same
        :return: None
        :rtype: None
        """
        for i in range(len(self.image_files)):
            assert self.image_files[i] == self.label_files[i]
            assert self.image_files[i] == self.depth_files[i]
            assert self.label_files[i] == self.depth_files[i]

    def __len__(self) -> int:
        """Return the length of the dataset.
        :return: length of the dataset
        :rtype: int
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        label_path = os.path.join(self.label_dir, self.label_files[idx])

        image = np.load(img_path).astype(np.float32)
        depth = np.load(depth_path).astype(np.float32)
        label = np.load(label_path).astype(np.float32)

        if self.transforms is not None:
            transformed_data = self.transforms(image=image, depth_image=depth, label=label)
            image = transformed_data["image"]
            depth = transformed_data["depth_image"]
            label = transformed_data["label"]

        return {
            "image": image,
            "depth_image": depth,
            "label": label,
        }
