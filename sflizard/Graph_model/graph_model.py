from typing import Any, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.nn import Linear
from torch_geometric.nn import GAT, GCN, GIN, GraphSAGE, SAGEConv

# import wandb


class GraphCustom(torch.nn.Module):
    """Custom graph model adding linear layers before and after the graph layers."""

    def __init__(
        self,
        dim_in: int,
        dim_h: int,
        dim_out: int,
        num_layers: int,
        layer_type: torch.nn.Module,
        custom_input_layer: int = 0,
        custom_input_hidden: int = 8,
        custom_output_layer: int = 0,
        custom_output_hidden: int = 8,
        custom_wide_connections: bool = False,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the model.

        Args:
            dim_in (int): The dimension of the input.
            dim_h (int): The dimension of the hidden layers.
            dim_out (int): The dimension of the output.
            num_layers (int): The number of graph layers.
            layer_type (torch.nn.Module): The type of graph layer to use.
            custom_input_layer (int): The number of linear input layers.
            custom_input_hidden (int): The dimension of the linear input hidden layers.
            custom_output_layer (int): The number of linear output layers.
            custom_output_hidden (int): The dimension of the linear output hidden layers.
            custom_wide_connections (bool): Whether to use wide connections.
            dropout (float): The dropout rate.

        Returns:
            None.

        Raises:
            None.
        """
        super().__init__()
        self.num_layers = num_layers
        self.custom_input_layer = custom_input_layer
        self.custom_output_layer = custom_output_layer

        # define the model
        self.model = torch.nn.ModuleList()
        # linear input layers
        for i in range(self.custom_input_layer):
            in_size = dim_in if i == 0 else custom_input_hidden
            if i == self.custom_input_layer - 1:
                if custom_wide_connections:
                    out_size = dim_h
                else:
                    out_size = dim_in
            else:
                out_size = custom_input_hidden
            self.model.append(Linear(in_size, out_size))
        # graph layers
        for i in range(self.num_layers):
            if custom_wide_connections:
                in_size = dim_in if self.custom_input_layer == 0 and i == 0 else dim_h
                out_size = (
                    dim_out
                    if self.custom_output_layer == 0 and i == self.num_layers - 1
                    else dim_h
                )
            else:
                in_size = dim_in if i == 0 else dim_h
                out_size = dim_out if i == self.num_layers - 1 else dim_h
            self.model.append(layer_type(in_size, out_size))
        # linear output layers
        for i in range(self.custom_output_layer):
            if i == 0:
                if custom_wide_connections:
                    in_size = dim_h
                else:
                    in_size = dim_out
            else:
                in_size = custom_output_hidden
            out_size = (
                dim_out if i == self.custom_output_layer - 1 else custom_output_hidden
            )
            if dropout > 0:
                self.model.append(
                    torch.nn.Sequential(nn.Dropout(dropout), Linear(in_size, out_size))
                )
            else:
                self.model.append(Linear(in_size, out_size))

        # log the model
        print(self.model.__repr__())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.
            edge_index (torch.Tensor): The edge index tensor.

        Returns:
            output (torch.Tensor): The output tensor.

        Raises:
            None.
        """
        # linear input layers
        for i in range(self.custom_input_layer):
            if i == 0:
                x = self.model[i](x).sigmoid()
            else:
                x = self.model[i](x).relu()
        # graph layers
        for i in range(
            self.custom_input_layer, self.custom_input_layer + self.num_layers
        ):
            x = self.model[i](x, edge_index).relu()
        # linear output layers
        for i in range(
            self.custom_input_layer + self.num_layers,
            self.custom_input_layer + self.num_layers + self.custom_output_layer,
        ):
            x = self.model[i](x).relu()
        return x


####################################################################################################


class Graph(pl.LightningModule):
    """Graph model lightning module."""

    def __init__(
        self,
        model: str = "graph_gat",
        learning_rate: float = 0.001,
        num_features: int = 33,
        num_classes: int = 7,
        seed: int = 303,
        max_epochs: int = 200,
        dim_h: int = 32,
        num_layers: int = 1,
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
        wandb_log: bool = False,
        custom_input_layer: int = 0,
        custom_input_hidden: int = 8,
        custom_output_layer: int = 0,
        custom_output_hidden: int = 8,
        custom_wide_connections: bool = False,
        dropout: float = 0.0,
    ) -> None:
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
            heads (int): The number of heads for the graph attention layer (only for graph_gat).
            class_weights (List[float]): The class weights.
            wandb_log (bool): Whether to log to wandb.
            custom_input_layer (int): The number of linear input layers (only for graph_custom).
            custom_input_hidden (int): The dimension of the linear input hidden layers (only for graph_custom).
            custom_output_layer (int): The number of linear output layers (only for graph_custom).
            custom_output_hidden (int): The dimension of the linear output hidden layers (only for graph_custom).
            custom_wide_connections (bool): Whether to use wide connections between linear and graph layers (only for graph_custom).
            dropout (float): The dropout rate.

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
                dropout=dropout,
            )
        elif "graph_gin" in model:
            self.model = GIN(
                in_channels=self.num_features,
                hidden_channels=dim_h,
                num_layers=num_layers,
                out_channels=self.num_classes,
                dropout=dropout,
            )
        elif "GCN" in model:
            self.model = GCN(
                in_channels=self.num_features,
                hidden_channels=dim_h,
                num_layers=num_layers,
                out_channels=self.num_classes,
                dropout=dropout,
            )
        elif model == "graph_sage":
            self.model = GraphSAGE(
                in_channels=self.num_features,
                hidden_channels=dim_h,
                num_layers=num_layers,
                out_channels=self.num_classes,
                dropout=dropout,
            )
        elif model == "graph_custom":
            self.model = GraphCustom(
                dim_in=num_features,
                dim_h=dim_h,
                dim_out=num_classes,
                num_layers=num_layers,
                layer_type=SAGEConv,
                custom_input_layer=custom_input_layer,
                custom_input_hidden=custom_input_hidden,
                custom_output_layer=custom_output_layer,
                custom_output_hidden=custom_output_hidden,
                custom_wide_connections=custom_wide_connections,
                dropout=dropout,
            )
        # if self.wandb_log:
        #     wandb.watch(self.model)
        self.seed = seed
        self.max_epochs = max_epochs

        self.val_acc = torchmetrics.Accuracy()
        self.val_acc_macro = torchmetrics.Accuracy(
            num_classes=self.num_classes, average="macro", mdmc_average="global"
        )

        if class_weights is not None:
            class_w = torch.tensor(class_weights).to("cuda")
            self.loss = nn.CrossEntropyLoss(weight=class_w)
        else:
            self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Any:  # type: ignore
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
    ) -> torch.Tensor:
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
        x, edge_index = batch.x, batch.edge_index  # type: ignore
        label = batch.y  # type: ignore
        label = label.long()
        logger_batch_size = len(batch.y)  # type: ignore

        outputs = self.model(x, edge_index)
        loss = self.loss(outputs, label)
        pred = outputs.argmax(-1)

        if name == "train":
            self.log("train_loss", loss, batch_size=logger_batch_size)
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

    def training_step(  # type: ignore
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
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

    def validation_step(  # type: ignore
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
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
    ) -> None:
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
                # if name == "val":
                #     wandb.log({f"{name}_acc": self.val_acc.compute()})
                #     wandb.log({f"{name}_acc_macro": self.val_acc_macro.compute()})
                pass
        else:
            raise ValueError(f"Invalid step name given: {name}")

    def training_epoch_end(self, outputs: List[torch.Tensor]) -> None:  # type: ignore
        """Training epoch end.

        Args:
            outputs (List[torch.Tensor]): The outputs.

        Returns:
            None.

        Raises:
            None.
        """
        self._epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs: List[torch.Tensor]) -> None:  # type: ignore
        """Validation epoch end.

        Args:
            outputs (List[torch.Tensor]): The outputs.

        Returns:
            None.

        Raises:
            None.
        """
        self._epoch_end(outputs, "val")

    def configure_optimizers(
        self,
    ) -> Tuple[
        List[torch.optim.Optimizer], List[torch.optim.lr_scheduler._LRScheduler]
    ]:
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
