# sflizard.Graph_model package

* [sflizard.Graph_model package](sflizard.Graph_model.md)


    * [Submodules](sflizard.Graph_model.md#submodules)


    * [sflizard.Graph_model.graph_model module](sflizard.Graph_model.md#module-sflizard.Graph_model.graph_model)


        * [`Graph`](sflizard.Graph_model.md#sflizard.Graph_model.graph_model.Graph)


            * [`Graph.configure_optimizers()`](sflizard.Graph_model.md#sflizard.Graph_model.graph_model.Graph.configure_optimizers)


            * [`Graph.forward()`](sflizard.Graph_model.md#sflizard.Graph_model.graph_model.Graph.forward)


            * [`Graph.training`](sflizard.Graph_model.md#sflizard.Graph_model.graph_model.Graph.training)


            * [`Graph.training_epoch_end()`](sflizard.Graph_model.md#sflizard.Graph_model.graph_model.Graph.training_epoch_end)


            * [`Graph.training_step()`](sflizard.Graph_model.md#sflizard.Graph_model.graph_model.Graph.training_step)


            * [`Graph.validation_epoch_end()`](sflizard.Graph_model.md#sflizard.Graph_model.graph_model.Graph.validation_epoch_end)


            * [`Graph.validation_step()`](sflizard.Graph_model.md#sflizard.Graph_model.graph_model.Graph.validation_step)


        * [`GraphCustom`](sflizard.Graph_model.md#sflizard.Graph_model.graph_model.GraphCustom)


            * [`GraphCustom.forward()`](sflizard.Graph_model.md#sflizard.Graph_model.graph_model.GraphCustom.forward)


            * [`GraphCustom.training`](sflizard.Graph_model.md#sflizard.Graph_model.graph_model.GraphCustom.training)


    * [Module contents](sflizard.Graph_model.md#module-sflizard.Graph_model)

## Submodules

## sflizard.Graph_model.graph_model module


### _class_ sflizard.Graph_model.graph_model.Graph(model: str = 'graph_gat', learning_rate: float = 0.01, num_features: int = 33, num_classes: int = 7, seed: int = 303, max_epochs: int = 20, dim_h: int = 32, num_layers: int = 0, heads: int = 1, class_weights: List[float] = [0, 0.3713368309107073, 0.008605586894052789, 0.01929911238667816, 0.06729488533622548, 0.515399722585458, 0.018063861886878453], wandb_log: bool = False, custom_input_layer: int = 0, custom_input_hidden: int = 8, custom_output_layer: int = 0, custom_output_hidden: int = 8, custom_wide_connections: bool = False, dropout: float = 0.0)
Bases: `LightningModule`

Graph model lightning module.


#### configure_optimizers()
Configure optimizers.

Args:

    None.

Returns:

    tuple: tuple containing:

        optimizers (List[torch.optim.Optimizer]): The optimizers.
        schedulers (List[torch.optim.lr_scheduler._LRScheduler]): The schedulers.

Raises:

    None.


#### forward(x: Tensor, edge_index: Tensor)
Forward pass.

Args:

    x (torch.Tensor): The input tensor.
    edge_index (torch.Tensor): The edge index tensor.

Returns:

    output (torch.Tensor): The output tensor.

Raises:

    None.




#### training_epoch_end(outputs: List[Tensor])
Training epoch end.

Args:

    outputs (List[torch.Tensor]): The outputs.

Returns:

    None.

Raises:

    None.


#### training_step(batch: Tensor, batch_idx: int)
Training step.

Args:

    batch (torch.Tensor): The batch.
    batch_idx (int): The batch index.

Returns:

    loss (torch.Tensor): The loss.

Raises:

    None.


#### validation_epoch_end(outputs: List[Tensor])
Validation epoch end.

Args:

    outputs (List[torch.Tensor]): The outputs.

Returns:

    None.

Raises:

    None.


#### validation_step(batch: Tensor, batch_idx: int)
Validation step.

Args:

    batch (torch.Tensor): The batch.
    batch_idx (int): The batch index.

Returns:

    loss (torch.Tensor): The loss.

Raises:

    None.


### _class_ sflizard.Graph_model.graph_model.GraphCustom(dim_in: int, dim_h: int, dim_out: int, num_layers: int, layer_type: Module, custom_input_layer: int = 0, custom_input_hidden: int = 8, custom_output_layer: int = 0, custom_output_hidden: int = 8, custom_wide_connections: bool = False, dropout: float = 0.0)
Bases: `Module`

Custom graph model adding linear layers before and after the graph layers.


#### forward(x: Tensor, edge_index: Tensor)
Forward pass of the model.

Args:

    x (torch.Tensor): The input tensor.
    edge_index (torch.Tensor): The edge index tensor.

Returns:

    output (torch.Tensor): The output tensor.

Raises:

    None.


