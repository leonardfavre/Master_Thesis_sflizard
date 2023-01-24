import os.path as osp

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset, LightningDataset
from tqdm import tqdm

from sflizard.data_utils import get_graph


class LizardGraphDataset(Dataset):
    def __init__(
        self,
        transform=None,
        pre_transform=None,
        df: pd.DataFrame = None,
        data: np.ndarray = None,
        name: str = "",
        n_rays: int = 32,
        distance: int = 45,
        stardist_checkpoint: str = None,
        x_type: str = "ll",  # ll: last_layer, c: classification, p: position, a:area
        root="data/graph",
        consep_data=False,
        light=False,
    ):
        self.df = df
        self.data = data
        self.name = name
        self.n_rays = n_rays
        self.distance = distance
        self.stardist_checkpoint = stardist_checkpoint
        self.x_type = x_type
        self.consep_data = consep_data
        self.light=light
        root = f"{root}/{distance}/{x_type}/{'light' if light else 'w_img'}/{name}"
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"data_{idx}.pt" for idx in range(len(self.df))]

    def download(self):
        pass

    def process(self):
        for idx in tqdm(range(len(self.df)), desc=f"Processing {self.name} dataset"):
            image = (
                torch.tensor(self.data[self.df.iloc[idx].id]).permute(2, 0, 1)
                if self.data is not None
                else None
            )
            class_map = (
                self.df.iloc[idx].class_map if "class_map" in self.df.columns else None
            )
            inst_map = (
                self.df.iloc[idx].inst_map if "inst_map" in self.df.columns else None
            )
            predicted_classes = (
                self.df.iloc[idx].predicted_classes
                if "predicted_classes" in self.df.columns
                else None
            )
            points = self.df.iloc[idx].points if "points" in self.df.columns else None

            graph = get_graph(
                inst_map=inst_map,
                points=points,
                predicted_classes=predicted_classes,
                true_class_map=class_map,
                n_rays=self.n_rays,
                distance=self.distance,
                stardist_checkpoint=self.stardist_checkpoint,
                image=image,
                x_type=self.x_type,
                consep_data=self.consep_data,
            )
            if self.light:
                processed_data = Data(
                    x=graph["x"],
                    y=graph["y"],
                    edge_index=graph["edge_index"],
                    image_idx = idx,
                    #original_img=image,
                    #class_map=class_map,
                )
            else:
                processed_data = Data(
                    x=graph["x"],
                    y=graph["y"],
                    edge_index=graph["edge_index"],
                    image_idx = idx,
                    original_img=image,
                    class_map=class_map,
                )
            torch.save(processed_data, osp.join(self.processed_dir, f"data_{idx}.pt"))

    def len(self):
        return len(self.df)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        return data


def LizardGraphDataModule(
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_data: pd.DataFrame = None,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 303,
    stardist_checkpoint=None,
    x_type="ll",
    distance=45,
    root="data/graph",
    consep_data=False,
    light=False,
):
    """Data module to create dataloaders for graph training job.

    Two mode possible:
    - using pkl datapath: data must be in pkl format, with annotations and images.
    - using mat datapath: no test set.

    Args:
        train

    Returns:

    Raises:

    """

    # train_df, valid_df = train_test_split(
    #     train_data,
    #     test_size=0.2,
    #     random_state=seed,
    # )

    # train_df = train_data.reset_index(drop=True)
    train_df = train_data["annotations"] if type(train_data) == dict and "annotations" in train_data else train_data.reset_index(drop=True)
    images = train_data["images"] if type(train_data) == dict and "images" in train_data else None
    train_ds = LizardGraphDataset(
        df=train_df,
        data=images,
        name="train",
        stardist_checkpoint=stardist_checkpoint,
        distance=distance,
        x_type=x_type,
        root=root,
        consep_data=consep_data,
        light=light,
    )

    # valid_df = valid_data.reset_index(drop=True)
    valid_df = valid_data["annotations"] if type(valid_data) == dict and "annotations" in valid_data else valid_data.reset_index(drop=True)
    images = valid_data["images"] if type(valid_data) == dict and "images" in valid_data else None
    valid_ds = LizardGraphDataset(
        df=valid_df,
        data=images,
        name="valid",
        stardist_checkpoint=stardist_checkpoint,
        distance=distance,
        x_type=x_type,
        root=root,
        consep_data=consep_data,
        light=light,
    )

    if test_data is not None:
        # test_df = test_data.reset_index(drop=True)
        test_df = test_data["annotations"] if type(test_data) == dict and "annotations" in test_data else test_data.reset_index(drop=True)
        images = test_data["images"] if type(test_data) == dict and "images" in test_data else None
        test_ds = LizardGraphDataset(
            df=test_df,
            data=images,
            name="test",
            stardist_checkpoint=stardist_checkpoint,
            distance=distance,
            x_type=x_type,
            root=root,
            consep_data=consep_data,
            light=light,
        )
        return LightningDataset(
            train_ds, valid_ds, test_ds, batch_size=batch_size, num_workers=num_workers
        )
    else:
        return LightningDataset(
            train_ds, valid_ds, batch_size=batch_size, num_workers=num_workers
        )
