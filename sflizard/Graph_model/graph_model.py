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
    GAT,
    GIN,
    RECT_L,
    GCNConv,
    GraphSAGE,
    JumpingKnowledge,
    SAGEConv,
)


class CustomGCN(torch.nn.Module):
    def __init__(
        self,
        layer_type,
        dim_in,
        dim_h,
        dim_out,
        num_layers=0,
    ):
        super().__init__()

        self.conv1 = layer_type(dim_in, dim_h)
        self.convh = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convh.append(layer_type(dim_h, dim_h))
        self.conv2 = layer_type(dim_h, dim_out)

    def forward(self, x, edge_index, edge_attr):
        x = F.elu(self.conv1(x, edge_index))
        for i in range(len(self.convh)):
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.elu(self.convh[i](x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)


class CustomGATGraph(torch.nn.Module):
    def __init__(self, layer_type, dim_in, dim_h, dim_out, heads, num_layers):
        super().__init__()
        self.conv1 = layer_type(dim_in, dim_h, heads, dropout=0.6)
        self.convh = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convh.append(layer_type(dim_h * heads, dim_h, heads, dropout=0.6))
        self.conv2 = layer_type(
            dim_h * heads,
            dim_out,
            heads=1,
            dropout=0.6,
        )

    def forward(self, x, edge_index, edge_attr):
        x = F.elu(self.conv1(x, edge_index))
        for i in range(len(self.convh)):
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.elu(self.convh[i](x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGEModel(torch.nn.Module):
    """GraphSAGE"""

    def __init__(self, dim_in, dim_h, dim_out, num_layers):
        super().__init__()
        self.sage1 = SAGEConv(dim_in, dim_h)
        self.sageh = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.sageh.append(SAGEConv(dim_h, dim_h))
        self.sage2 = SAGEConv(dim_h, dim_out)

    def forward(self, x, edge_index, edge_attr):
        x = self.sage1(x, edge_index).relu()
        for i in range(len(self.sageh)):
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.elu(self.sageh[i](x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.sage2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GraphTest(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, layer_type):
        super().__init__()

        self.model = torch.nn.ModuleList()
        self.model.append(Linear(dim_in, 1024))
        self.model.append(Linear(1024, 1024))
        self.model.append(Linear(1024, dim_h))
        for i in range(3):
            self.model.append(layer_type(dim_h, dim_h))
            self.model.append(layer_type(dim_h, dim_h))
            self.model.append(layer_type(dim_h, dim_h))
            self.model.append(layer_type(dim_h, dim_h))
            self.model.append(JumpingKnowledge("cat", dim_h))
            self.model.append(Linear(4 * dim_h, dim_h))
        self.model.append(Linear(dim_h, dim_out))

    def forward(self, x, edge_index, edge_attr):
        x = self.model[0](x).relu()
        x = self.model[1](x).relu()
        x = self.model[2](x).relu()
        for i in range(3):
            xa = self.model[6 * i + 3](x, edge_index).relu()
            xb = self.model[6 * i + 4](xa, edge_index).relu()
            xc = self.model[6 * i + 5](xb, edge_index).relu()
            xd = self.model[6 * i + 6](xc, edge_index).relu()
            x = self.model[6 * i + 7]([xa, xb, xc, xd])
            x = self.model[6 * i + 8](x).relu()
        x = self.model[-1](x)
        return x


class GraphTestBest(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, layer_type):
        super().__init__()
        self.model = torch.nn.ModuleList()
        self.model.append(Linear(dim_in, 1024))
        self.model.append(Linear(1024, 1024))
        self.model.append(Linear(1024, dim_h))
        for i in range(3):
            self.model.append(layer_type(dim_h, dim_h))
            self.model.append(layer_type(dim_h, dim_h))
            self.model.append(layer_type(dim_h, dim_h))
            self.model.append(layer_type(dim_h, dim_h))
            self.model.append(JumpingKnowledge("cat", dim_h))
            self.model.append(Linear(4 * dim_h, dim_h))
        self.model.append(Linear(dim_h, dim_out))

    def forward(self, x, edge_index, edge_attr):
        x = self.model[0](x).relu()
        x = self.model[1](x).relu()
        x = self.model[2](x).relu()
        for i in range(3):
            xa = self.model[6 * i + 3](x, edge_index).relu()
            xb = self.model[6 * i + 4](xa, edge_index).relu()
            xc = self.model[6 * i + 5](xb, edge_index).relu()
            xd = self.model[6 * i + 6](xc, edge_index).relu()
            x = self.model[6 * i + 7]([xa, xb, xc, xd])
            x = self.model[6 * i + 8](x).relu()
        x = self.model[-1](x)
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
            1 / 0.8435234983048621,
            1 / 0.0015844697497448515,
            1 / 0.09702835179125052,
            1 / 0.018770678077839286,
            1 / 0.005716505874930195,
            1 / 0.0011799091886332306,
            1 / 0.03219658701273987,
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
                jk="cat",
            )
            # self.model = CustomGATGraph(
            #     layer_type=GATv2Conv if model == "graph_gatv2" else GATConv,
            #     dim_in=num_features,
            #     dim_h=dim_h,
            #     dim_out=num_classes,
            #     heads=heads,
            #     num_layers=num_layers,
            # )
        elif "graph_rect" in model:
            self.model = RECT_L(
                in_channels=self.num_features,
                hidden_channels=dim_h,
            )
        elif "graph_gin" in model:
            self.model = GIN(
                in_channels=self.num_features,
                hidden_channels=dim_h,
                num_layers=num_layers,
                out_channels=self.num_classes,
                jk="cat",
            )
        elif "GCN" in model:
            self.model = CustomGCN(
                layer_type=GCNConv,
                dim_in=self.num_features,
                dim_h=dim_h,
                dim_out=self.num_classes,
                num_layers=num_layers,
            )
        elif model == "graph_sage":
            self.model = GraphSAGE(
                in_channels=self.num_features,
                hidden_channels=dim_h,
                num_layers=num_layers,
                out_channels=self.num_classes,
            )
        elif model == "graph_test":
            self.model = GraphTestBest(
                dim_in=num_features,
                dim_h=dim_h,
                dim_out=num_classes,
                layer_type=SAGEConv,  # GraphConv,
            )
        self.seed = seed
        self.max_epochs = max_epochs

        if class_weights is not None:
            class_weights = torch.tensor(class_weights).to("cuda")
            self.loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        """Forward pass."""
        return self.model(x, edge_index)

    def _step(self, batch, name):
        x, edge_index = batch.x, batch.edge_index
        label = batch.y - 1
        # check if label contains background
        if torch.sum(label == -1) > 0:
            print("label contains background.")
            label = label + 1
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
        # if batch_idx == 0:
        #     self.output_results(batch) # only work with batch size 1   aw     s
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

    def output_results(self, batch):
        """Output results."""
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr

        outputs = self.model(x, edge_index)  # , edge_attr)
        pred = outputs.argmax(-1)

        class_map = batch.class_map[0]
        pos = batch.pos.cpu().numpy()
        for idx, point in enumerate(pos):
            for b in range(-2, 3):
                for c in range(-2, 3):
                    class_map[(int(point[0]) + b) % 540][
                        (int(point[1]) + c) % 540
                    ] = pred[idx]

        class_map = class_map * 255 / 10
        class_map = class_map.astype(np.uint8)
        class_map_img = Image.fromarray(class_map)
        class_map_img.save("outputs/class_map.png")
