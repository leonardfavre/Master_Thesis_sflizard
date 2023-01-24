from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.nn import Linear
from torch_geometric.nn import (
    GCN,
    GAT,
    GIN,
    GraphSAGE,
    JumpingKnowledge,
    SAGEConv,
)
import torchmetrics


class GraphCustom(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, num_layers, layer_type):
        super().__init__()
        self.num_layers = num_layers
        self.model = torch.nn.ModuleList()
        self.model.append(Linear(dim_in, 1024))
        self.model.append(Linear(1024, 1024))
        self.model.append(Linear(1024, dim_h))
        for _ in range(self.num_layers):
            self.model.append(layer_type(dim_h, dim_h))
        # self.model.append(JumpingKnowledge("cat", dim_h))
        self.model.append(Linear(dim_h, dim_h))
        self.model.append(Linear(dim_h, dim_out))

    def forward(self, x, edge_index):
        x = self.model[0](x).sigmoid()
        x = self.model[1](x).relu()
        x = self.model[2](x).relu()
        for i in range(self.num_layers):
            x = self.model[3+i](x, edge_index).relu()
        x = self.model[-2](x).relu()
        x = self.model[-1](x).relu()
        return x


####################################################################################################


class Graph(pl.LightningModule):
    def __init__(
        self,
        model: str = "graph_gat",
        learning_rate: float = 0.01,
        num_features: int = 33,
        num_classes: int = 6,
        seed: int = 303,
        max_epochs: int = 20,
        dim_h: int = 32,
        num_layers: int = 0,
        heads: int = 1,
        class_weights: List[float] = [
            0,
            0.3713368309107073,
            0.008605586894052789,
            0.01929911238667816,
            0.06729488533622548,
            0.515399722585458,
            0.018063861886878453

            # 1,
            # 90.18723210806598,
            # 2.0900540911512655,
            # 4.68720951818412,
            # 16.344027681335106,
            # 125.17604110329907,
            # 4.38720204716698,
            # 1 / 0.8435234983048621,
            # 1 / 0.0015844697497448515,
            # 1 / 0.09702835179125052,
            # 1 / 0.018770678077839286,
            # 1 / 0.005716505874930195,
            # 1 / 0.0011799091886332306,
            # 1 / 0.03219658701273987,
        ],
    ):

        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.num_features = num_features
        self.num_classes = num_classes

        if "graph_gat" in model:
            self.model = GAT(
                in_channels=self.num_features,
                hidden_channels=dim_h,
                num_layers=num_layers,
                out_channels=self.num_classes,
            )
        elif "graph_gin" in model:
            self.model = GIN(
                in_channels=self.num_features,
                hidden_channels=dim_h,
                num_layers=num_layers,
                out_channels=self.num_classes,
            )
        elif "GCN" in model:
            self.model = GCN(
                in_channels=self.num_features,
                hidden_channels=dim_h,
                num_layers=num_layers,
                out_channels=self.num_classes,
            )
        elif model == "graph_sage":
            self.model = GraphSAGE(
                in_channels=self.num_features,
                hidden_channels=dim_h,
                num_layers=num_layers,
                out_channels=self.num_classes,
            )
        elif model == "graph_custom":
            self.model = GraphCustom(
                dim_in=num_features,
                dim_h=dim_h,
                dim_out=num_classes,
                num_layers=num_layers,
                layer_type=SAGEConv, 
            )
        self.seed = seed
        self.max_epochs = max_epochs

        self.val_acc = torchmetrics.Accuracy()
        self.val_acc_macro = torchmetrics.Accuracy(num_classes=self.num_classes, average="macro", mdmc_average="global")

        if class_weights is not None:
            class_weights = torch.tensor(class_weights).to("cuda")
            self.loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        """Forward pass."""
        return self.model(x, edge_index)

    def _step(self, batch, batch_idx, name):
        x, edge_index = batch.x, batch.edge_index
        label = batch.y 
        label = label.long()

        outputs = self.model(x, edge_index)
        loss = self.loss(outputs, label)
        pred = outputs.argmax(-1)

        if name == "train":
            self.log("train_loss", loss)
            return loss
        elif name == "val":
            self.val_acc(pred, label)
            self.val_acc_macro(pred, label)
            self.log(
                "val_loss",
                loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "val_acc",
                self.val_acc,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                "val_acc_macro",
                self.val_acc_macro,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )
            return loss #, accuracy
        elif name == "test":
            return loss #, accuracy
        else:
            raise ValueError(f"Invalid step name given: {name}")

    def training_step(self, batch, batch_idx):
        """Training step."""
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        """Test step."""
        # if batch_idx == 0:
        #     self.output_results(batch) # only work with batch size 1   aw     s
        return self._step(batch, batch_idx, "test")

    # def _epoch_end(self, outputs, name):
    #     """Epoch end."""

    #     # loss = 0.0
    #     # accuracy = 0
    #     # batch_nbr = 0

    #     # for lo, a in outputs:
    #     #     if lo == lo:
    #     #         loss += lo
    #     #     if a == a:
    #     #         accuracy += a
    #     #     batch_nbr += 1

    #     # loss /= batch_nbr
    #     # accuracy /= batch_nbr

    #     # self.log(f"{name}_acc", accuracy)
    #     # self.log(f"{name}_loss", loss)

    # def validation_epoch_end(self, outputs):
    #     self._epoch_end(outputs, "val")

    # def test_epoch_end(self, outputs):
    #     self._epoch_end(outputs, "test")

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

    # def output_results(self, batch):
    #     """Output results."""
    #     x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

    #     outputs = self.model(x, edge_index)  # , edge_attr)
    #     pred = outputs.argmax(-1)

    #     class_map = batch.class_map[0]
    #     pos = batch.pos.cpu().numpy()
    #     for idx, point in enumerate(pos):
    #         for b in range(-2, 3):
    #             for c in range(-2, 3):
    #                 class_map[(int(point[0]) + b) % 540][
    #                     (int(point[1]) + c) % 540
    #                 ] = pred[idx]

    #     class_map = class_map * 255 / 10
    #     class_map = class_map.astype(np.uint8)
    #     class_map_img = Image.fromarray(class_map)
    #     class_map_img.save("outputs/class_map.png")
