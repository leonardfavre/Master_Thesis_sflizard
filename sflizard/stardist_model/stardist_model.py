import pytorch_lightning as pl
import torch
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

import wandb
from sflizard.stardist_model import ClassL1BCELoss, MyL1BCELoss
from sflizard.stardist_model import UNetStar as UNet
from typing import List, Tuple

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
        wandb_log: bool=False,
        max_epochs: int=200,
    )-> None:
        """Initialize the model.
        
        Args:
            learning_rate (float): The learning rate.
            input_size (int): The input size.
            in_channels (int): The number of input channels.
            n_rays (int): The number of rays.
            n_classes (int): The number of classes.
            loss_power_scaler (float): The loss power scaler.
            seed (int): The seed.
            device (str): The device.
            wandb_log (bool): Whether to log to wandb.
            max_epochs (int): The maximum number of epochs.
            
        Returns:
            None.
            
        Raises:
            None.
        """
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

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """Forward pass.
        
        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            x (torch.Tensor): The output tensor.

        Raises:
            None.
        """
        return self.model(x)

    def _step(self, batch: torch.Tensor, name: str)-> torch.Tensor:
        """General step.
        
        Args:
            batch (torch.Tensor): The batch.
            name (str): The name of the step (train or val).
            
        Returns:
            loss (torch.Tensor): The loss.
            
        Raises:
            ValueError: If the name is not train or val.
        """
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

        else:
            raise ValueError(f"Invalid step name given: {name}")

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int)-> torch.Tensor:
        """Training step.
        
        Args:
            batch (torch.Tensor): The batch.
            batch_idx (int): The batch index.
            
        Returns:
            loss (torch.Tensor): The loss.
            
        Raises:
            None.
        """
        return self._step(batch, "train")

    def training_epoch_end(self, outputs: List[torch.Tensor])-> None:
        """Training epoch end.
        
        Args:
            outputs (List[torch.Tensor]): The outputs.
            
        Returns:
            None.
            
        Raises:
            None.
        """
        outputs = [x["loss"] for x in outputs if x is not None]
        self._epoch_end(outputs, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int)-> torch.Tensor:
        """Validation step.
        
        Args:
            batch (torch.Tensor): The batch.
            batch_idx (int): The batch index.
            
        Returns:
            loss (torch.Tensor): The loss.
            
        Raises:
            None.
        """
        return self._step(batch, "val")

    def validation_epoch_end(self, outputs: List[torch.Tensor])-> None:
        """Validation epoch end.
        
        Args:
            outputs (List[torch.Tensor]): The outputs.
            
        Returns:
            None.
            
        Raises:
            None.
        """
        self._epoch_end(outputs, "val")

    def _epoch_end(self, outputs: List[torch.Tensor], name: str)-> None:
        """epoch end for train/val.
        
        Args:
            outputs (List[torch.Tensor]): The outputs.
            name (str): The name of the step (train or val).
            
        Returns:
            None.
            
        Raises:
            ValueError: If the name is not train or val.
        """
        if name in ["train", "val"]:
            if self.wandb_log:
                wandb.log({f"{name}_loss": torch.stack(outputs).mean()})
        else:
            raise ValueError(f"Invalid step name given: {name}")

    def configure_optimizers(self)-> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]]:
        """Configure optimizers.

        Args:
            None.

        Returns:
            tuple: tuple containing:
                optimizers (List[torch.optim.Optimizer]): The optimizers.
                schedulers (List[torch.optim.lr_scheduler._LRScheduler]): The schedulers.

        Raises:
            None.
        """
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
