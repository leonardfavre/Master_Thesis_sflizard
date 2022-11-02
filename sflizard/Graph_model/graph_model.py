import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.nn import Linear
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    GCNConv,
    GraphConv,
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
        self.model.append(layer_type(dim_in, dim_h))
        self.model.append(layer_type(dim_h, dim_h))
        self.model.append(layer_type(dim_h, dim_h))
        self.model.append(layer_type(dim_h, dim_h))
        self.jk = JumpingKnowledge("cat", dim_h)
        self.lin = Linear(4 * dim_h, dim_out)

    def forward(self, x, edge_index, edge_attr):
        x1 = F.elu(self.model[0](x, edge_index))
        x2 = F.dropout(x1, p=0.5, training=self.training)
        x2 = F.elu(self.model[1](x2, edge_index))
        x3 = F.dropout(x2, p=0.5, training=self.training)
        x3 = F.elu(self.model[2](x3, edge_index))
        x4 = F.dropout(x3, p=0.5, training=self.training)
        x4 = F.elu(self.model[2](x4, edge_index))
        x5 = self.jk([x1, x2, x3, x4])
        # x3 = TopKPooling(x3)
        x5 = self.lin(x5)
        return x5


####################################################################################################


class Graph(pl.LightningModule):
    def __init__(
        self,
        model: str = "graph_gat",
        learning_rate: float = 0.01,
        num_features: int = 32,
        num_classes: int = 7,
        seed: int = 303,
        max_epochs: int = 20,
        dim_h: int = 32,
        num_layers: int = 0,
        heads: int = 1,
    ):

        super().__init__()
        self.learning_rate = learning_rate
        self.num_features = num_features
        self.num_classes = num_classes
        # self.model = GraphSAGEModel(num_features, 256, num_classes)
        if "graph_gat" in model:
            self.model = CustomGATGraph(
                layer_type=GATv2Conv if model == "graph_gatv2" else GATConv,
                dim_in=num_features,
                dim_h=dim_h,
                dim_out=num_classes,
                heads=heads,
                num_layers=num_layers,
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
            self.model = GraphSAGEModel(
                dim_in=num_features,
                dim_h=dim_h,
                dim_out=num_classes,
                num_layers=num_layers,
            )
        elif model == "graph_test":
            self.model = GraphTest(
                dim_in=num_features,
                dim_h=dim_h,
                dim_out=num_classes,
                layer_type=GraphConv,
            )
        self.seed = seed
        self.max_epochs = max_epochs

        self.loss = F.cross_entropy

    def forward(self, x, edge_index, edge_attr):
        """Forward pass."""
        return self.model(x, edge_index, edge_attr)

    def _step(self, batch, name):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        label = batch.y
        label = label.long()

        outputs = self.model(x, edge_index, edge_attr)
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
            loss += lo
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
