"""Copyright (C) SquareFactory SA - All Rights Reserved.

This source code is protected under international copyright law. All rights 
reserved and protected by the copyright holders.
This file is confidential and only available to authorized individuals with the
permission of the copyright holders. If you encounter this file and do not have
permission, please contact the copyright holders and delete this file.
"""
from __future__ import annotations
from typing import Optional
from pathlib import Path
import pickle

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from stardist import star_dist, edt_prob

class LizardDataset(Dataset):
    """Dataset object for the Lizard."""

    def __init__(
        self,
        df: pd.DataFrame,
        data: np.ndarray,
        tf: A.Compose,
        annotation_target: str,
        aditional_args: Optional[dict] = None,
        test: bool = False,
    ):
        """Initialize dataset."""
        self.df = df
        self.data = data
        self.tf = tf
        self.test = test
        self.annotation_target = annotation_target
        self.aditional_args = aditional_args

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """Get images and transform them if needed."""
        image = np.array(self.data[self.df.iloc[idx].id])
        if self.tf is not None:
            image = self.tf(image=image)["image"]

        annotation = self.df.iloc[idx].annotation

        if self.annotation_target == "stardist":
            if "n_rays" not in self.aditional_args.keys():
                raise ValueError("n_rays not in aditional_args. Mandatory for stardist model.") 
            distances = star_dist(annotation, self.aditional_args["n_rays"])
            distances = torch.from_numpy(np.transpose(distances, (2, 0, 1)))
            obj_probabilities = edt_prob(annotation)
            obj_probabilities = torch.from_numpy(np.expand_dims(obj_probabilities, 0))
            return image, obj_probabilities, distances

        # for testing, return image id for reporting
        if self.test:
            return image, annotation, self.df.iloc[idx].id
        else:
            return image, annotation


class LizardDataModule(pl.LightningDataModule):
    """DataModule that returns the correct dataloaders for the Lizard dataset."""

    def __init__(
        self,
        data_path: str,
        annotation_target: str = "inst",
        batch_size: int = 32,
        num_workers: int = 4,
        input_size=540,
        seed: int = 303,
        aditional_args: Optional[dict] = None,
    ):
        """Initialize the dataloaders with batch size and targets."""
        super().__init__()

        data_path = Path(data_path)
        with data_path.open("rb") as f:
            data = pickle.load(f)
        self.data = data

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
        annotations = self.data["annotations"]
        if self.annotation_target in ["inst", "stardist"]:
            # keep only the nuclei instance
            annotations = annotations.drop(
                ["class_map", "nuclei_id", "classes", "bboxs", "centroids"], axis=1
            )
            annotations.rename(columns={"inst_map": "annotation"}, inplace=True)
        elif self.annotation_target == "class":
            # keep only the class
            annotations = annotations.drop(
                ["inst_map", "nuclei_id", "classes", "bboxs", "centroids"], axis=1
            )
            annotations.rename(columns={"class_map": "annotation"}, inplace=True)
            self.num_classes = len(
                np.unique(np.concatenate([np.unique(a) for a in annotations.class_map]))
            )
        elif self.annotation_target == "full":
            print("not implemented yet")
        else:
            raise ValueError(f"Invalid annotation target: {self.annotation_target}")

        train_df, test_df = train_test_split(
            annotations,
            test_size=0.2,
            random_state=self.seed,
        )

        train_df, valid_df = train_test_split(
            train_df,
            test_size=0.2,
            random_state=self.seed,
        )

        train_df.reset_index(drop=True, inplace=True)
        valid_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        tf_base = A.Compose(
            [
                A.Resize(self.input_size, self.input_size),
                A.Normalize(mean=0, std=1),
                ToTensorV2(),
            ]
        )
        tf_augment = A.Compose(
            [
                A.Resize(self.input_size, self.input_size),
                A.Normalize(mean=0, std=1),
                A.HorizontalFlip(),
                A.RandomBrightnessContrast(p=0.2),
                ToTensorV2(),
            ]
        )
        data = self.data["images"]
        print(f"Training with {len(train_df)} examples")
        self.train_ds = LizardDataset(
            train_df, data, tf_augment, self.annotation_target, self.aditional_args
        )
        self.valid_ds = LizardDataset(valid_df, data, tf_base, self.annotation_target, self.aditional_args)
        self.test_ds = LizardDataset(
            test_df, data, tf_base, self.annotation_target, self.aditional_args, test=True
        )

    def train_dataloader(self):
        """Return the training dataloader."""
        return DataLoader(self.train_ds, **self.dataloader_arguments)

    def val_dataloader(self):
        """Return the validation dataloader."""
        return DataLoader(self.valid_ds, **self.dataloader_arguments)

    def test_dataloader(self):
        """Return the test dataloader."""
        return DataLoader(self.test_ds, **self.dataloader_arguments)
