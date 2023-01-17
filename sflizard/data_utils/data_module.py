"""Copyright (C) SquareFactory SA - All Rights Reserved.

This source code is protected under international copyright law. All rights 
reserved and protected by the copyright holders.
This file is confidential and only available to authorized individuals with the
permission of the copyright holders. If you encounter this file and do not have
permission, please contact the copyright holders and delete this file.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from sflizard.data_utils import get_stardist_data


class LizardDataset(Dataset):
    """Dataset object for the Lizard."""

    def __init__(
        self,
        df: pd.DataFrame,
        data: np.ndarray,
        tf_base: A.Compose,
        tf_augment: A.Compose,
        annotation_target: str,
        aditional_args: Optional[dict] = None,
        test: bool = False,
    ):
        """Initialize dataset."""
        self.df = df
        self.data = data
        self.tf_base = tf_base
        self.tf_augment = tf_augment
        self.test = test
        self.annotation_target = annotation_target
        self.aditional_args = aditional_args

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """Get images and transform them if needed."""
        # retriev inputs
        image = np.array(self.data[self.df.iloc[idx].id])

        # retrieve masks
        inst_map = self.df.iloc[idx].inst_map
        masks = [inst_map]
        if self.annotation_target == "stardist_class":
            class_map = self.df.iloc[idx].class_map
            masks.append(class_map)

        # augmentations
        if self.tf_augment is not None:
            transformed = self.tf_augment(image=image, masks=masks)
            image = transformed["image"]
            inst_map = transformed["masks"][0]
            if self.annotation_target == "stardist_class":
                class_map = transformed["masks"][1]
        if self.tf_base is not None:
            image = self.tf_base(image=image)["image"]

        # get targets in stardist form
        if self.annotation_target == "stardist":
            obj_probabilities, distances = get_stardist_data(
                inst_map, self.aditional_args
            )
            return image, obj_probabilities, distances
        elif self.annotation_target == "stardist_class":
            obj_probabilities, distances, classes = get_stardist_data(
                inst_map,
                self.aditional_args,
                class_map,
            )
            return image, obj_probabilities, distances, classes
        else:
            raise ValueError(
                f"Annotation target {self.annotation_target} not supported."
            )


class LizardDataModule(pl.LightningDataModule):
    """DataModule that returns the correct dataloaders for the Lizard dataset."""

    def __init__(
        self,
        train_data_path: str,
        valid_data_path: str,
        test_data_path: str,
        annotation_target: str = "inst",
        batch_size: int = 4,
        num_workers: int = 4,
        input_size=540,
        seed: int = 303,
        aditional_args: Optional[dict] = None,
    ):
        """Initialize the dataloaders with batch size and targets."""
        super().__init__()

        if train_data_path is not None:
            train_data_path = Path(train_data_path)
            with train_data_path.open("rb") as f:
                self.train_data = pickle.load(f)
        else:
            self.train_data = None
        valid_data_path = Path(valid_data_path)
        with valid_data_path.open("rb") as f:
            self.valid_data = pickle.load(f)
        test_data_path = Path(test_data_path)
        with test_data_path.open("rb") as f:
            self.test_data = pickle.load(f)

        self.annotation_target = annotation_target

        self.dataloader_arguments = {
            "batch_size": batch_size,
            "num_workers": num_workers,
        }
        self.input_size = input_size
        self.seed = seed
        self.aditional_args = aditional_args

    def setup(self, stage: Optional[str] = None):
        """Data setup for training."""

        tf_base = A.Compose(
            [
                A.Resize(self.input_size, self.input_size),
                A.Normalize(mean=0, std=1),
                ToTensorV2(),
            ]
        )
        tf_augment = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # A.RandomBrightnessContrast(p=0.2),
                A.RandomSizedCrop(
                    min_max_height=[self.input_size / 2, self.input_size],
                    height=self.input_size,
                    width=self.input_size,
                    p=0.75,
                ),
            ]
        )

        if self.train_data is not None:
            train_df = self.train_data["annotations"]
            train_df.reset_index(drop=True, inplace=True)
            print(f"Training with {len(train_df)} examples")
            self.train_ds = LizardDataset(
                train_df,
                self.train_data["images"],
                tf_base,
                tf_augment,
                self.annotation_target,
                self.aditional_args,
            )

        valid_df = self.valid_data["annotations"]
        test_df = self.test_data["annotations"]

        valid_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        self.valid_ds = LizardDataset(
            valid_df,
            self.valid_data["images"],
            tf_base,
            None,
            self.annotation_target,
            self.aditional_args,
        )
        self.test_ds = LizardDataset(
            test_df,
            self.test_data["images"],
            tf_base,
            None,
            self.annotation_target,
            self.aditional_args,
            test=True,
        )

    def train_dataloader(self):
        """Return the training dataloader."""
        if self.train_data is None:
            return None
        return DataLoader(self.train_ds, **self.dataloader_arguments)

    def val_dataloader(self):
        """Return the validation dataloader."""
        return DataLoader(self.valid_ds, **self.dataloader_arguments)

    def test_dataloader(self):
        """Return the test dataloader."""
        return DataLoader(self.test_ds, **self.dataloader_arguments)
