import pytorch_lightning as pl
import torch
import torchmetrics
from rich.console import Console
from rich.table import Table
from stardist.matching import matching_dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import wandb
from sflizard.stardist_model import ClassL1BCELoss, MyL1BCELoss
from sflizard.stardist_model import UNetStar as UNet


class Stardist(pl.LightningModule):
    """Stardist model class."""

    def __init__(
        self,
        learning_rate: float = 1e-4,
        input_size: int = 540,
        in_channels: int = 3,
        n_rays: int = 32,
        n_classes: int = 1,
        loss_power_scaler: float = 0.0,
        seed: int = 303,
        device: str = "cpu",
        wandb_log=False,
    ):
        """Initialize the model."""
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.input_size = input_size
        self.seed = seed

        self.classification = n_classes > 1

        if self.classification:
            self.model = UNet(in_channels, n_rays, n_classes)
            class_weights = [
                1 / 0.8421763419196278,
                1 / 0.0014213456163252062,
                1 / 0.09916321656931723,
                1 / 0.01824858239486998,
                1 / 0.005942925203180082,
                1 / 0.001227361184860687,
                1 / 0.03182022711181911,
            ]
            class_weights = torch.tensor(class_weights).to(device)
            loss_scale = [
                2**loss_power_scaler / 0.24240250366886978,
                3**loss_power_scaler / 0.16856008596522243,
                1**loss_power_scaler / 1.1985746181324908,
            ]
            self.loss = ClassL1BCELoss(class_weights, loss_scale)
            self.test_classes_acc = torchmetrics.Accuracy(
                num_classes=n_classes, mdmc_average="global"
            )
            self.test_classes_f1 = torchmetrics.F1Score(
                num_classes=n_classes, mdmc_average="global"
            )
            self.test_classes_acc_mac = torchmetrics.Accuracy(
                num_classes=n_classes, average="macro", mdmc_average="global"
            )
            self.test_classes_f1_mac = torchmetrics.F1Score(
                num_classes=n_classes, average="macro", mdmc_average="global"
            )
        else:
            self.model = UNet(in_channels, n_rays)
            self.loss = MyL1BCELoss()
        self.wandb_log = wandb_log
        if self.wandb_log:
            wandb.watch(self.model)

    def forward(self, x):
        """Forward pass."""
        return self.model(x)

    def _step(self, batch, name):
        """General step."""
        if self.classification:
            inputs, obj_probabilities, distances, classes = batch
            classes = classes.long()
        else:
            inputs, obj_probabilities, distances = batch

        outputs = self.model(inputs)

        distances = distances.squeeze(1)

        if self.classification:
            loss = self.loss(outputs, obj_probabilities, distances, classes)
        else:
            loss = self.loss(outputs, obj_probabilities, distances)

        if name == "train":
            self.log("train_loss", loss)
        elif name == "val":
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
            )

        elif name == "test":
            if self.classification:
                dist, prob, clas = outputs
            else:
                dist, prob = outputs

            mask_true = self.model.compute_star_label(
                inputs, distances, obj_probabilities
            )
            mark_pred = self.model.compute_star_label(inputs, dist, prob)
            metrics = matching_dataset(mask_true, mark_pred, show_progress=False)

            self.test_values["precision"].append(metrics.precision)
            self.test_values["recall"].append(metrics.recall)
            self.test_values["acc"].append(metrics.accuracy)
            self.test_values["f1"].append(metrics.f1)
            self.test_values["panoptic_quality"].append(metrics.panoptic_quality)

            if self.classification:
                best_clas = torch.argmax(clas, dim=1)
                self.test_classes_acc(best_clas, classes)
                self.test_classes_f1(best_clas, classes)
                self.test_classes_acc_mac(best_clas, classes)
                self.test_classes_f1_mac(best_clas, classes)
        else:
            raise ValueError(f"Invalid step name given: {name}")

        return loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        return self._step(batch, "train")

    def training_epoch_end(self, outputs):
        """Training epoch end."""
        outputs = [x["loss"] for x in outputs if x is not None]
        self._epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self._step(batch, "val")

    def validation_epoch_end(self, outputs):
        """Validation epoch end."""
        self._epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        """Test step."""
        if batch_idx == 0:
            self.test_values: dict[str, list] = {
                "precision": [],
                "recall": [],
                "acc": [],
                "f1": [],
                "panoptic_quality": [],
            }
        return self._step(batch, "test")

    def _epoch_end(self, outputs, name):
        """epoch end for train/val."""
        if name in ["train", "val"]:
            if self.wandb_log:
                wandb.log({f"{name}_loss": torch.stack(outputs).mean()})
        else:
            raise ValueError(f"Invalid step name given: {name}")

    def test_epoch_end(self, outputs):
        """Test epoch end."""
        for metric in self.test_values:
            self.test_values[metric] = (
                torch.tensor(self.test_values[metric]).float().mean()
            )
            self.log(f"test {metric}", self.test_values[metric])
        if self.classification:
            table = Table(title="Classification metrics")
            table.add_column("metric \\ avg", justify="center")
            table.add_column("micro", justify="center")
            table.add_column("macro", justify="center")
            table.add_row(
                "Accuracy",
                str(self.test_classes_acc.compute().item()),
                str(self.test_classes_acc_mac.compute().item()),
            )
            table.add_row(
                "F1",
                str(self.test_classes_f1.compute().item()),
                str(self.test_classes_f1_mac.compute().item()),
            )
            console = Console()
            console.print(table)

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
