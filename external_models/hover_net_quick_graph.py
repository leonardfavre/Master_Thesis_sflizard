import scipy.io as sio
from pathlib import Path
import torch

import os.path as osp
from pathlib import Path

import math

import numpy as np
import torch
from torch_geometric.data import Data, Dataset, LightningDataset
from tqdm import tqdm
import argparse
import os
from datetime import datetime
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from PIL import Image
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch_geometric.nn import GraphSAGE
import torch.nn as nn

###### GRAPH

CONSEP = False
TRUE_DATA_PATH = "../data/Lizard_dataset_test_split/Lizard_Labels_train/Labels/" # "../data/Lizard_dataset_test_split/Lizard_Labels_train/Labels/"
TRAIN_DATA_PATH = "hover_net/Lizard_train_out/mat/"
NUM_CLASSES = 7 # 5

def get_edge_list(vertex, distance):
    # edge distanche
    def distance_between_vertex(v_i, v_j):
        distance = ((v_i[0] - v_j[0]) ** 2 + (v_i[1] - v_j[1]) ** 2) ** (0.5)
        return distance

    edge_list = [[], [], []]
    for i in range(vertex.shape[0]):
        for j in range(i + 1, vertex.shape[0]):
            dist = distance_between_vertex(vertex[i], vertex[j])
            if dist < distance:
                edge_list[0].append(i)
                edge_list[1].append(j)
                edge_list[0].append(j)
                edge_list[1].append(i)
                edge_list[2].append(dist)
                edge_list[2].append(dist)
    return edge_list


def get_graph(
    points,
    predicted_class,
    true_class_map,
    distance,
    ):
    """Get the graph from the hovernet inference data.
    """
    graph = {}

    # get points with target
    y = []


    for i in range(points.shape[0]):
        yi1 = int(true_class_map[int(points[i, 1]), int(points[i, 0])])
        yi2 = int(true_class_map[math.ceil(points[i, 1]), int(points[i, 0])])
        yi3 = int(true_class_map[int(points[i, 1]), math.ceil(points[i, 0])])
        yi4 = int(true_class_map[math.ceil(points[i, 1]), math.ceil(points[i, 0])])
        if (yi1 == yi2) & (yi2 == yi3) & (yi3 == yi4):
            yi = yi1
        else:
            possible_y = [yi1, yi2, yi3, yi4]
            # remove 0
            possible_y = [x for x in possible_y if x != 0]
            yi = max(set(possible_y), key=possible_y.count)
        if CONSEP:
            if (yi == 3) or (yi == 4):
                yi = 3
            elif (yi == 5) | (yi == 6) | (yi == 7):
                yi = 4
        y.append(yi)

    # add class prediction info to graph
    graph["x"] = torch.Tensor(predicted_class)

    graph["pos"] = torch.Tensor(points)
    graph["y"] = torch.Tensor(y)

    # compute edge information
    edge_list = get_edge_list(points, distance)

    # add edge information to graph
    graph["edge_index"] = torch.tensor([edge_list[0], edge_list[1]], dtype=torch.long)
    graph["edge_attr"] = torch.tensor(edge_list[2], dtype=torch.float)

    return graph


class LizardGraphDataset(Dataset):
    def __init__(
        self,
        root="data/graph_lizard",
        transform=None,
        pre_transform=None,
        pred_data=None, 
        true_data=None,
        name: str = "",
        distance: int = 45,
    ):
        self.pred_data = pred_data
        self.true_data = true_data
        self.name = name
        self.distance = distance
        root = f"{root}/{name}"
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"data_{idx}.pt" for idx in range(len(self.pred_data))]

    def download(self):
        pass

    def process(self):
        for idx in tqdm(range(len(self.pred_data)), desc=f"Processing {self.name} dataset"):
            # image = torch.tensor(self.data[self.df.iloc[idx].id]).permute(2, 0, 1)
            if not CONSEP:
                mat_file = self.true_data[idx]
                ann_inst = mat_file["inst_map"]
                nuclei_id = np.squeeze(mat_file["id"]).tolist()
                patch_id = np.unique(ann_inst).tolist()[1:]
                ann_type = np.zeros(ann_inst.shape)
                for v in patch_id:
                    idn = nuclei_id.index(v)
                    ann_type[ann_inst == v] = mat_file["class"][idn]
                ann_type = ann_type.astype("int32")
                # ann = np.dstack([ann_inst, ann_type])
                # ann = ann.astype("int32")
            else:
                ann_type = self.true_data[idx]["type_map"]

            graph = get_graph(
                points=self.pred_data[idx]["inst_centroid"],
                predicted_class=self.pred_data[idx]["inst_type"],
                true_class_map=ann_type,
                distance=self.distance,
            )
            processed_data = Data(
                x=graph["x"],
                y=graph["y"],
                pos=graph["pos"],
                edge_index=graph["edge_index"],
                edge_attr=["edge_attr"],
            )
            torch.save(processed_data, osp.join(self.processed_dir, f"data_{idx}.pt"))

    def len(self):
        return len(self.pred_data)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f"data_{idx}.pt"))
        return data
        # return self.processed_data[idx]


def LizardGraphDataModule(
    pred_data: str, true_data:str, batch_size: int = 32, num_workers: int = 4, seed: int = 303
):
    ds = LizardGraphDataset(pred_data=pred_data, true_data=true_data, name="train")

    return LightningDataset(
        ds, batch_size=batch_size, num_workers=num_workers
    )

class Graph(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.01,
        num_features: int = 1,
        num_classes: int = NUM_CLASSES,
        seed: int = 303,
        max_epochs: int = 20,
        dim_h: int = 32,
        num_layers: int = 2,
    ):

        super().__init__()
        
        self.learning_rate = learning_rate
        self.num_features = num_features
        self.num_classes = num_classes
        
        self.model = GraphSAGE(
            in_channels=self.num_features,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=self.num_classes,
        )

        print(self.model)

        self.seed = seed
        self.max_epochs = max_epochs

        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        """Forward pass."""
        return self.model(x, edge_index)

    def _step(self, batch, name):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        label = batch.y
        label = label.long()

        outputs = self.model(x, edge_index)
        loss = self.loss(outputs, label)
        pred = outputs.argmax(-1)
        accuracy = (pred == label).sum() / pred.shape[0]

        if name == "train":
            self.log("train_loss", loss)
            self.log("train_acc", accuracy)
            return loss
        elif name == "val":
            return loss, accuracy
        elif name == "test":
            return loss, accuracy
        else:
            raise ValueError(f"Invalid step name given: {name}")

    def training_step(self, batch, batch_idx):
        """Training step."""
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Test step."""
        return self._step(batch, "test")

    def _epoch_end(self, outputs, name):
        """Epoch end."""

        loss = 0.0
        accuracy = 0
        batch_nbr = 0

        for lo, a in outputs:
            if lo == lo:
                loss += lo
            if a == a:
                accuracy += a
            batch_nbr += 1

        loss /= batch_nbr
        accuracy /= batch_nbr

        self.log(f"{name}_acc", accuracy)
        self.log(f"{name}_loss", loss)

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, "test")

    def configure_optimizers(self, scheduler="cosine"):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=5e-4,
        )
        schedulers = {
            "cosine": LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=int(self.max_epochs / 10),
                max_epochs=self.max_epochs,
            ),
            "step": torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[5, 10], gamma=0.1
            ),
            "lambda": torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 0.95**epoch
            ),
        }

        if scheduler not in schedulers.keys():
            raise ValueError(
                f"Invalid scheduler given: {scheduler}. You can implement a new one by modifying the Classifier.configure_optimizers method."
            )

        scheduler = schedulers[scheduler]
        return [optimizer], [scheduler]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument(
        "-dh",
        "--dimh",
        type=int,
        default=32,
        help="Dimension of the hidden layer in the grap model.",
    )
    parser.add_argument(
        "-l",
        "--layers",
        type=int,
        default=4,
        help="Number of layers in the grap model.",
    )

    args = parser.parse_args()

    data_path = TRUE_DATA_PATH
    file_list = list(Path(data_path).glob("*.mat"))

    true_data = []

    for file_path in file_list:
        base_name = file_path.stem
        mat = sio.loadmat(file_path)
        true_data.append(mat)

    data_path = TRAIN_DATA_PATH
    file_list = list(Path(data_path).glob("*.mat"))

    pred_data = []

    for file_path in file_list:
        base_name = file_path.stem
        mat = sio.loadmat(file_path)
        pred_data.append(mat)

    # create the datamodule
    dm = LizardGraphDataModule(
        pred_data= pred_data, 
        true_data= true_data,
    )
    dm.setup()

    for dim_h in [32, 64, 128]:
        for num_layers in [2, 4, 8, 16]:
            # create the model
            model = Graph(
                learning_rate = 0.01,
                max_epochs = 20,
                dim_h = dim_h,
                num_layers = num_layers,
            )

            device = torch.device(
                "cuda"
            )

            # create the trainer
            trainer = pl.Trainer.from_argparse_args(args)
            print(model.num_classes)
            # # train the model
            trainer.fit(model, dm)

            # # save the model
            now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            trainer.save_checkpoint(
                f"models/full_training_hover_net_lizard_graph_{args.max_epochs}epochs_{num_layers}layer_sage_{dim_h}h.ckpt"
            )


    # trainer.test(model, dm)