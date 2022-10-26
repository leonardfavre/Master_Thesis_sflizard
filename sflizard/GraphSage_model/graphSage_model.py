import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import SAGEConv


class GraphSAGEModel(torch.nn.Module):
    """GraphSAGE"""

    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.sage1 = SAGEConv(dim_in, dim_h)
        self.sage2 = SAGEConv(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = self.sage1(x, edge_index).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.sage2(h, edge_index)
        return F.log_softmax(h, dim=1)


class GraphSAGE(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.01,
        num_features: int = 32,
        num_classes: int = 7,
        seed: int = 303,
    ):

        super().__init__()
        self.learning_rate = learning_rate
        self.num_features = num_features
        self.num_classes = num_classes
        self.model = GraphSAGEModel(num_features, 128, num_classes)
        self.seed = seed

        self.loss = F.cross_entropy

    def forward(self, x, edge_index):
        """Forward pass."""
        return self.model(x, edge_index)

    def _step(self, batch, name):
        x, edge_index = batch.x, batch.edge_index
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

        for loss, accuracy in outputs:
            loss += loss
            accuracy += accuracy
            batch_nbr += 1
        loss /= batch_nbr
        accuracy /= batch_nbr

        self.log(f"{name}_acc", accuracy)
        self.log(f"{name}_loss", loss)

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=5e-4,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            "min",
            factor=0.5,
            verbose=True,
            patience=6,
            eps=1e-8,
            threshold=1e-20,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
