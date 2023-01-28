import os.path as osp

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset, LightningDataset
from tqdm import tqdm
from typing import Union
from sflizard.data_utils import get_graph


class LizardGraphDataset(Dataset):
    """Dataset object for the Graphs."""

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
        root: str="data/graph",
        consep_data: bool=False,
        light: bool=False,
    )-> None:
        """Initialize dataset.

        Args:
            transform (None): transform.
            pre_transform (None): pre_transform.
            df (pd.DataFrame): dataframe containing the data.
            data (np.ndarray): array containing the images.
            name (str): name of the dataset.
            n_rays (int): number of rays of stardist shape.
            distance (int): distance between 2 connected cells.
            stardist_checkpoint (str): path to the stardist checkpoint.
            x_type (str): type of the node feature vetor.
            root (str): root path.
            consep_data (bool): if the data is from consep.
            light (bool): if the data included in the graph needs to be minimum, speed up training.

        Returns:
            None.

        Raises:
            None.
        """
        self.df = df
        self.data = data
        self.name = name
        self.n_rays = n_rays
        self.distance = distance
        self.stardist_checkpoint = stardist_checkpoint
        self.x_type = x_type
        self.consep_data = consep_data
        self.light = light
        root = f"{root}/{distance}/{x_type}/{'light' if light else 'w_img'}/{name}"
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self)-> list:
        return []

    @property
    def processed_file_names(self)-> list:
        return [f"data_{idx}.pt" for idx in range(len(self.df))]

    def download(self)-> None:
        pass

    def process(self)-> None:
        """Process the dataset.

        Compute the graph from input data and save it for speed up use of the dataset.
        If the dataset is light, only the graph basic information is saved:
            - x: node features
            - edge_index: edges
            - y: labels
            - image_idx: image index
        If the dataset is not light, the graph full information is saved:
            - x: node features
            - edge_index: edges
            - y: labels
            - image_idx: image index
            - original_img: original image
            - inst_map: instance map
            - class_map: class map

        Args:
            None.
            
        Returns:
            None.
            
        Raises:
            None.
        """
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
                hovernet_metric=not self.light,
            )
            if graph is None:
                print(f"No cell detected for this image: {self.df.iloc[idx].id}")
            else:
                if self.light:
                    processed_data = Data(
                        x=graph["x"],
                        y=graph["y"],
                        edge_index=graph["edge_index"],
                        image_idx=self.df.iloc[idx].id,
                        # original_img=image,
                        # class_map=class_map,
                    )
                else:
                    processed_data = Data(
                        x=graph["x"],
                        y=graph["y"],
                        edge_index=graph["edge_index"],
                        pos=graph["pos"],
                        image_idx=self.df.iloc[idx].id,
                        original_img=image,
                        class_map=class_map,
                        inst_map=graph["inst_map"],
                    )
                torch.save(
                    processed_data, osp.join(self.processed_dir, f"data_{idx}.pt")
                )

    def len(self)-> int:
        """Return the length of the dataset.
        
        Args:
            None.

        Returns:
            int: length of the dataset.

        Raises:
            None.
        """
        return len(self.df)

    def get(self, idx) -> Data:
        """Return the data at index idx.

        Args:
            idx (int): index of the data to return.

        Returns:
            data (Data): data at index idx.

        Raises:
            None.
        """
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        return data


def LizardGraphDataModule(
    train_data: Union[dict, pd.DataFrame] = None,
    valid_data: Union[dict, pd.DataFrame] = None,
    test_data: Union[dict, pd.DataFrame] = None,
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
    - images and annotations dataframe contained in a dict.
    - directly the annotation dataframe.

    Args:
        train_data (dict or pd.DataFrame): train data.
        valid_data (dict or pd.DataFrame): valid data.
        test_data (dict or pd.DataFrame): test data.
        batch_size (int): batch size.
        num_workers (int): number of workers.
        seed (int): seed for random number generator.
        stardist_checkpoint (str): path to stardist checkpoint.
        x_type (str): type of node features.
        distance (int): distance for graph creation.
        root (str): root path for saving processed data.
        consep_data (bool): if True, use consep data.
        light (bool): if True, only save basic graph information.

    Returns:
        datamodule (LightningDataset): Datamodule containing the required datasets.

    Raises:
        ValueError: if no data is provided.
    """

    # train_df = train_data.reset_index(drop=True)
    if train_data is not None:
        train_df = (
            train_data["annotations"]
            if type(train_data) == dict and "annotations" in train_data
            else train_data.reset_index(drop=True)
        )
        images = (
            train_data["images"]
            if type(train_data) == dict and "images" in train_data
            else None
        )
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
    if valid_data is not None:
        valid_df = (
            valid_data["annotations"]
            if type(valid_data) == dict and "annotations" in valid_data
            else valid_data.reset_index(drop=True)
        )
        images = (
            valid_data["images"]
            if type(valid_data) == dict and "images" in valid_data
            else None
        )
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
        test_df = (
            test_data["annotations"]
            if type(test_data) == dict and "annotations" in test_data
            else test_data.reset_index(drop=True)
        )
        images = (
            test_data["images"]
            if type(test_data) == dict and "images" in test_data
            else None
        )
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

    if train_data is not None:
        if valid_data is not None:
            if test_data is not None:
                return LightningDataset(
                    train_ds,
                    valid_ds,
                    test_ds,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
            else:
                return LightningDataset(
                    train_ds, valid_ds, batch_size=batch_size, num_workers=num_workers
                )
        else:
            if test_data is not None:
                return LightningDataset(
                    train_ds, test_ds, batch_size=batch_size, num_workers=num_workers
                )
            else:
                return LightningDataset(
                    train_ds, batch_size=batch_size, num_workers=num_workers
                )
    else:
        if valid_data is not None:
            if test_data is not None:
                return LightningDataset(
                    valid_ds, test_ds, batch_size=batch_size, num_workers=num_workers
                )
            else:
                return LightningDataset(
                    valid_ds, batch_size=batch_size, num_workers=num_workers
                )
        else:
            if test_data is not None:
                return LightningDataset(
                    test_ds, batch_size=batch_size, num_workers=num_workers
                )
            else:
                raise ValueError("No dataset provided")
