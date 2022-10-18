import pytorch_lightning as pl
import torch
import torchmetrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from stardist.matching import matching_dataset

from sflizard.stardist_model import UNetStar as UNet
from sflizard.stardist_model import MyL1BCELoss


class Stardist(pl.LightningModule):
    """Stardist model class."""

    def __init__(
        self,
        learning_rate: float = 1e-4,
        input_size: int = 540,
        in_channels: int = 3,
        n_rays: int = 32,
        seed: int = 303,
    ):
        """Initialize the model."""
        super().__init__()
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.seed = seed

        self.model = UNet(in_channels, n_rays)

        self.loss = MyL1BCELoss()

        self.val_values = {
            "inputs": [],
            "dist": [],
            "prob": [],
            "targets": [],
            "distances": [],
        }

        self.test_values = {
            "inputs": [],
            "dist": [],
            "prob": [],
            "targets": [],
            "distances": [],
        }

    def forward(self, x):
        """Forward pass."""
        return self.model(x)

    def _step(self, batch, name):
        """General step."""
        inputs, targets, distances = batch

        outputs = self.model(inputs)

        distances = distances.squeeze(1)
        loss = self.loss(outputs, distances, targets)

        if name == "train":
            self.log("train_loss", loss)
        elif name == "val":
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
            )
            # dist, prob = outputs
            # self.val_values["inputs"].append(inputs.cpu())
            # self.val_values["dist"].append(dist.cpu())
            # self.val_values["prob"].append(prob.cpu())
            # self.val_values["targets"].append(targets.cpu())
            # self.val_values["distances"].append(distances.cpu())
            
        elif name == "test":
            dist, prob = outputs
            self.test_values["inputs"].append(inputs.cpu())
            self.test_values["dist"].append(dist.cpu())
            self.test_values["prob"].append(prob.cpu())
            self.test_values["targets"].append(targets.cpu())
            self.test_values["distances"].append(distances.cpu())
        else:
            raise ValueError(f"Invalid step name given: {name}")

        return loss

    def _log_metric(self, name, values):
        """Calculate metrics and log."""
        for val in values:
            values[val] = torch.cat(values[val])
        mask_true = self.model.compute_star_label(values["inputs"], values["distances"], values["targets"])
        mark_pred = self.model.compute_star_label(values["inputs"], values["dist"], values["prob"])
        metrics = matching_dataset(mask_true, mark_pred)
        self.log(f"{name}_acc", metrics.accuracy)

    def training_step(self, batch, batch_idx):
        """Training step."""
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self._step(batch, "val")

    # def validation_epoch_end(self, outputs):
    #     """Validation epoch end."""
    #     self._log_metric("val", self.val_values)
    #     self.val_values = {
    #         "inputs": [],
    #         "dist": [],
    #         "prob": [],
    #         "targets": [],
    #         "distances": [],
    #     }

    def test_step(self, batch, batch_idx):
        """Test step."""
        return self._step(batch, "test")

    def test_epoch_end(self, outputs):
        """Test epoch end."""
        self._log_metric("test", self.test_values)
        self.test_values = {
            "inputs": [],
            "dist": [],
            "prob": [],
            "targets": [],
            "distances": [],
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=5e-5,
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
