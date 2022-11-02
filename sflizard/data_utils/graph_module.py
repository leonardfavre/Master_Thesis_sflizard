import os.path as osp
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Dataset, LightningDataset
from tqdm import tqdm

from sflizard.data_utils import get_graph_from_inst_map


class LizardGraphDataset(Dataset):
    def __init__(
        self,
        root="./data/graph",
        transform=None,
        pre_transform=None,
        df: pd.DataFrame = None,
        data: np.ndarray = None,
        name: str = "",
        n_rays: int = 32,
        distance: int = 45,
    ):
        self.df = df
        self.data = data
        self.name = name
        self.n_rays = n_rays
        self.distance = distance
        root = f"{root}/{name}"
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
            image = torch.tensor(self.data[self.df.iloc[idx].id]).permute(2, 0, 1)
            inst_map = self.df.iloc[idx].inst_map
            class_map = self.df.iloc[idx].class_map
            graph = get_graph_from_inst_map(
                inst_map,
                class_map,
                n_rays=self.n_rays,
                distance=self.distance,
            )
            processed_data = Data(
                x=graph["x"],
                y=graph["y"],
                pos=graph["pos"],
                edge_index=graph["edge_index"],
                edge_attr=["edge_attr"],
                original_img=image,
                class_map=class_map,
            )
            torch.save(processed_data, osp.join(self.processed_dir, f"data_{idx}.pt"))

    def len(self):
        return len(self.df)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        return data
        # return self.processed_data[idx]


def LizardGraphDataModule(
    data_path: str, batch_size: int = 32, num_workers: int = 4, seed: int = 303
):
    data_path = Path(data_path)
    with data_path.open("rb") as f:
        data = pickle.load(f)

    annotations = data["annotations"]

    train_df, test_df = train_test_split(
        annotations,
        test_size=0.2,
        random_state=seed,
    )

    train_df, valid_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=seed,
    )

    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_ds = LizardGraphDataset(df=train_df, data=data["images"], name="train")
    valid_ds = LizardGraphDataset(df=valid_df, data=data["images"], name="valid")
    test_ds = LizardGraphDataset(df=test_df, data=data["images"], name="test")

    return LightningDataset(
        train_ds, valid_ds, test_ds, batch_size=batch_size, num_workers=num_workers
    )
