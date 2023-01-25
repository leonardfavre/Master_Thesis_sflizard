import pytorch_lightning as pl
import torch
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

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
        max_epochs=200,
    ):
        """Initialize the model."""
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.input_size = input_size
        self.seed = seed

        # self.val_mse_dist = torchmetrics.MeanSquaredError()
        # self.val_acc_class = torchmetrics.Accuracy(num_classes=n_classes, average="micro", mdmc_average="global")
        # self.val_acc_class_macro = torchmetrics.Accuracy(num_classes=n_classes, average="macro", mdmc_average="global")

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

        else:
            self.model = UNet(in_channels, n_rays)
            self.loss = MyL1BCELoss()
        self.wandb_log = wandb_log
        if self.wandb_log:
            wandb.watch(self.model)

        self.max_epochs = max_epochs

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
            # self.val_acc_class(torch.index_select(torch.tensor(outputs), 1, 2), classes)
            # self.val_acc_class_macro(torch.index_select(torch.tensor(outputs), 1, torch.tensor([2])), classes)
            # self.val_mse_dist(torch.index_select(torch.tensor(outputs), 1, torch.tensor([0])), distances)
            # self.val_acc_class(outputs[2], classes)
            # self.val_acc_class_macro(outputs[2], classes)
            # self.val_mse_dist(outputs[0], distances)
            self.log(
                "val_loss",
                loss,
                on_step=False,
                on_epoch=True,
            )
            # self.log(
            #     "val_mse_dist",
            #     self.val_mse_dist,
            #     on_step=False,
            #     on_epoch=True,
            # )
            # self.log(
            #     "val_acc_class",
            #     self.val_acc_class,
            #     on_step=False,
            #     on_epoch=True,
            # )
            # self.log(
            #     "val_acc_class_macro",
            #     self.val_acc_class_macro,
            #     on_step=False,
            #     on_epoch=True,
            # )

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

    def _epoch_end(self, outputs, name):
        """epoch end for train/val."""
        if name in ["train", "val"]:
            if self.wandb_log:
                wandb.log({f"{name}_loss": torch.stack(outputs).mean()})
        else:
            raise ValueError(f"Invalid step name given: {name}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=5e-5,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=int(self.max_epochs / 10),
            max_epochs=self.max_epochs,
        )
        return [optimizer], [scheduler]
        # scheduler = ReduceLROnPlateau(
        #     optimizer,
        #     "min",
        #     factor=0.5,
        #     verbose=True,
        #     patience=6,
        #     eps=1e-8,
        #     threshold=1e-20,
        # )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": scheduler,
        #     "monitor": "val_loss",
        # }
