from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.nn import Linear
from torch_geometric.nn import GAT, GCN, GIN, GraphSAGE, SAGEConv

import wandb


class GraphCustom(torch.nn.Module):
    """Custom graph model adding linear layers before and after the graph layers."""

    def __init__(
        self, 
        dim_in: int, 
        dim_h: int, 
        dim_out: int, 
        num_layers: int, 
        layer_type: torch.nn.Module,
    )->None:
        """Initialize the model.
        
        Args:
            dim_in (int): The dimension of the input.
            dim_h (int): The dimension of the hidden layers.
            dim_out (int): The dimension of the output.
            num_layers (int): The number of graph layers.
            layer_type (torch.nn.Module): The type of graph layer to use.
            
        Returns:
            None.
            
        Raises:
            None.
        """
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor)->torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.
            edge_index (torch.Tensor): The edge index tensor.

        Returns:
            output (torch.Tensor): The output tensor.

        Raises:
            None.
        """
        x = self.model[0](x).sigmoid()
        x = self.model[1](x).relu()
        x = self.model[2](x).relu()
        for i in range(self.num_layers):
            x = self.model[3 + i](x, edge_index).relu()
        x = self.model[-2](x).relu()
        x = self.model[-1](x).relu()
        return x


####################################################################################################


class Graph(pl.LightningModule):
    """Graph model lightning module."""

    def __init__(
        self,
        model: str = "graph_gat",
        learning_rate: float = 0.01,
        num_features: int = 33,
        num_classes: int = 7,
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
            0.018063861886878453,
        ],
        wandb_log: bool=False,
    )->None:
        """Initialize the module.

        Args:
            model (str): The type of graph model to use.
            learning_rate (float): The learning rate.
            num_features (int): The number of features.
            num_classes (int): The number of classes.
            seed (int): The seed.
            max_epochs (int): The maximum number of epochs.
            dim_h (int): The dimension of the hidden layers.
            num_layers (int): The number of graph layers.
            heads (int): The number of heads for the graph attention layer.
            class_weights (List[float]): The class weights.
            wandb_log (bool): Whether to log to wandb.

        Returns:
            None.

        Raises:
            None.
        """

        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.num_features = num_features
        self.num_classes = num_classes
        self.wandb_log = wandb_log

        if "graph_gat" in model:
            self.model = GAT(
                in_channels=self.num_features,
                hidden_channels=dim_h,
                num_layers=num_layers,
                out_channels=self.num_classes,
                v2=True,
                heads=heads,
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
        if self.wandb_log:
            wandb.watch(self.model)
        self.seed = seed
        self.max_epochs = max_epochs

        self.val_acc = torchmetrics.Accuracy()
        self.val_acc_macro = torchmetrics.Accuracy(
            num_classes=self.num_classes, average="macro", mdmc_average="global"
        )

        if class_weights is not None:
            class_weights = torch.tensor(class_weights).to("cuda")
            self.loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor)-> torch.Tensor:
        """Forward pass.
        
        Args:
            x (torch.Tensor): The input tensor.
            edge_index (torch.Tensor): The edge index tensor.
            
        Returns:
            output (torch.Tensor): The output tensor.
        
        Raises:
            None.
        """
        return self.model(x, edge_index)

    def _step(
        self, 
        batch: torch.Tensor, 
        batch_idx: int, 
        name: str,
    )-> torch.Tensor:
        """Perform a step.

        Args:
            batch (torch.Tensor): The batch.
            batch_idx (int): The batch index.
            name (str): The name of the step.

        Returns:
            loss (torch.Tensor): The loss.

        Raises:
            ValueError: If the name is not train or val.
        """
        x, edge_index = batch.x, batch.edge_index
        label = batch.y
        label = label.long()
        logger_batch_size = len(batch.y)

        outputs = self.model(x, edge_index)
        loss = self.loss(outputs, label)
        pred = outputs.argmax(-1)

        if name == "train":
            self.log("train_loss", loss, batch_size=logger_batch_size)
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
                batch_size=logger_batch_size,
            )
            self.log(
                "val_acc",
                self.val_acc,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=logger_batch_size,
            )
            self.log(
                "val_acc_macro",
                self.val_acc_macro,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=logger_batch_size,
            )
        else:
            raise ValueError(f"Invalid step name given: {name}")

        return loss

    def training_step(
        self, 
        batch: torch.Tensor, 
        batch_idx: int,
    )-> torch.Tensor:
        """Training step.
        
        Args:
            batch (torch.Tensor): The batch.
            batch_idx (int): The batch index.

        Returns:
            loss (torch.Tensor): The loss.

        Raises:
            None.
        """
        return self._step(batch, batch_idx, "train")

    def validation_step(
        self, 
        batch: torch.Tensor, 
        batch_idx: int,
    )-> torch.Tensor:
        """Validation step.
        
        Args:
            batch (torch.Tensor): The batch.
            batch_idx (int): The batch index.
            
        Returns:
            loss (torch.Tensor): The loss.
            
        Raises:
            None.
        """
        return self._step(batch, batch_idx, "val")

    def _epoch_end(
        self, 
        outputs: List[torch.Tensor],
        name: str,
    )-> None:
        """Epoch end.
        
        Args:
            outputs (List[torch.Tensor]): The outputs.
            name (str): The name of the step.
            
        Returns:
            None.
            
        Raises:
            ValueError: If the name is not train or val.
        """
        if name in ["train", "val"]:
            if self.wandb_log:
                wandb.log({f"{name}_loss": torch.stack(outputs).mean()})
                if name == "val":
                    wandb.log({f"{name}_acc": self.val_acc})
                    wandb.log({f"{name}_acc_macro": self.val_acc_macro})
        else:
            raise ValueError(f"Invalid step name given: {name}")

    def training_epoch_end(self, outputs: List[torch.Tensor])-> None:
        """Training epoch end.
        
        Args:
            outputs (List[torch.Tensor]): The outputs.
            
        Returns:
            None.
            
        Raises:
            None.
        """
        self._epoch_end(outputs, "train")

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
            weight_decay=5e-4,
        )

        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=int(self.max_epochs / 10),
            max_epochs=self.max_epochs,
        )
        return [optimizer], [scheduler]
