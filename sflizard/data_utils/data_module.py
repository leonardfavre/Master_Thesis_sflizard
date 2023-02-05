from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Union

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
    ) -> None:
        """Initialize dataset.

        Args:
            df (pd.DataFrame): dataframe containing the data.
            data (np.ndarray): array containing the images.
            tf_base (A.Compose): base transformation.
            tf_augment (A.Compose): augmentation transformation.
            annotation_target (str): annotation target.
            aditional_args (Optional[dict]): aditional arguments. Used for nrays.

        Returns:
            None.

        Raises:
            None.
        """
        self.df = df
        self.data = data
        self.tf_base = tf_base
        self.tf_augment = tf_augment
        self.annotation_target = annotation_target
        self.aditional_args = aditional_args

    def __len__(self) -> int:
        """Get the length of the dataset.

        Args:
            None.

        Returns:
            len (int): length of the dataset.

        Raises:
            None.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        """Get item from the dataset.

        In the case of a classic stardist annotation target:
            - image
            - obj_probabilities map
            - distances map

        In the case of a stardist annotation target with classes:
            - image
            - obj_probabilities map
            - distances map
            - classes map

        Transform them if needed.

        Args:
            idx (int): index of the item.

        Returns:
            tuple: tuple containing:
                - image
                - obj_probabilities map
                - distances map
                - classes map (optional)

        Raises:
            ValueError: if the annotation target is not supported.
        """
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
        train_data_path: Union[str, None],
        valid_data_path: str,
        test_data_path: str,
        annotation_target: str = "stardist_class",
        batch_size: int = 4,
        num_workers: int = 4,
        input_size=540,
        seed: int = 303,
        aditional_args: Optional[dict] = None,
    ) -> None:
        """Create the datamodule and initialize the argument for the dataloaders.

        Args:
            train_data_path (str): path to the train data.
            valid_data_path (str): path to the valid data.
            test_data_path (str): path to the test data.
            annotation_target (str): annotation target.
            batch_size (int): batch size.
            num_workers (int): number of workers.
            input_size (int): input size.
            seed (int): seed.
            aditional_args (Optional[dict]): aditional arguments. Used for nrays.
        """
        super().__init__()

        if train_data_path is not None:
            train_data_p = Path(train_data_path)
            with train_data_p.open("rb") as f:
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

    def setup(self, stage: Optional[str] = None) -> None:
        """Data setup for training, define transformations and datasets.

        Args:
            stage (Optional[str]): stage.

        Returns:
            None.

        Raises:
            None.
        """

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
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader.

        Args:
            None.

        Returns:
            dataloader (DataLoader): the training dataloader.

        Raises:
            None.
        """
        if self.train_data is None:
            return None
        return DataLoader(self.train_ds, **self.dataloader_arguments)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader.

        Args:
            None.

        Returns:
            dataloader (DataLoader): the validation dataloader.

        Raises:
            None.
        """
        return DataLoader(self.valid_ds, **self.dataloader_arguments)

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader.

        Args:
            None.

        Returns:
            dataloader (DataLoader): the test dataloader.

        Raises:
            None.
        """
        return DataLoader(self.test_ds, **self.dataloader_arguments)
