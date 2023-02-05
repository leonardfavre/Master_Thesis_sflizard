# sflizard.Graph_model package

## Modules

This package contains the following modules:

* [sflizard.Graph_model.graph_model](sflizard.Graph_model.md#sflizard.Graph_model.graph_model)

## sflizard.Graph_model.graph_model


### _class_ sflizard.Graph_model.graph_model.Graph
Bases: `LightningModule`

Graph neural network model lightning module.
This class define the lightning module containing the graph neural network model.

Args:
    model (str): The type of graph model to use. Default to "graph_gat".
    learning_rate (float): The learning rate. Default to 0.001.
    num_features (int): The number of features. Default to 33.
    num_classes (int): The number of classes. Default to 7.
    seed (int): The seed. Default to 303.
    max_epochs (int): The maximum number of epochs. Default to 200.
    dim_h (int): The dimension of the hidden layers. Default to 32.
    num_layers (int): The number of graph layers. Default to 1.
    heads (int): The number of heads for the graph attention layer (only for graph_gat). Default to 1.
    class_weights (List[float]): The class weights. Default to [0, 0.3713368309107073, 0.008605586894052789, 0.01929911238667816, 0.06729488533622548, 0.515399722585458, 0.018063861886878453].
    wandb_log (bool): Whether to log to wandb. Default to False.
    custom_input_layer (int): The number of linear input layers (only for graph_custom). Default to 0.
    custom_input_hidden (int): The dimension of the linear input hidden layers (only for graph_custom). Default to 8.
    custom_output_layer (int): The number of linear output layers (only for graph_custom). Default to 0.
    custom_output_hidden (int): The dimension of the linear output hidden layers (only for graph_custom). Default to 8.
    custom_wide_connections (bool): Whether to use wide connections between linear and graph layers (only for graph_custom). Default to False.
    dropout (float): The dropout rate. Default to 0.0.

Raises:
    None.


#### configure_optimizers
Configure optimizers and schedulers used for training. Currently implemented is Adam optimizer and LinearWarmupCosineAnnealingLR scheduler.

Args:

    None.

Returns:

    tuple: tuple containing:

        optimizers (List[torch.optim.Optimizer]): The optimizers.
        schedulers (List[torch.optim.lr_scheduler._LRScheduler]): The schedulers.

Raises:

    None.


#### forward
Perform a forward pass. Calls the model with the input tensor (node feature vector) and the edge index as input.

Args:

    x (torch.Tensor): The input tensor.
    edge_index (torch.Tensor): The edge index tensor.

Returns:

    output (torch.Tensor): The output tensor.

Raises:

    None.




#### training_epoch_end
Run at the end of a training epoch. Log loss to wandb if wandb_log was set to True.

Args:

    outputs (List[torch.Tensor]): The outputs.

Returns:

    None.

Raises:

    None.


#### training_step
Run at each step of training. Get the current batch data, run the model and compute loss.

Args:

    batch (torch.Tensor): The batch.
    batch_idx (int): The batch index.

Returns:

    loss (torch.Tensor): The loss.

Raises:

    None.


#### validation_epoch_end
Run at the end of a validation epoch. Log loss to wandb if wandb_log was set to True.

Args:

    outputs (List[torch.Tensor]): The outputs.

Returns:

    None.

Raises:

    None.


#### validation_step
Run at each step of validation. Get the current batch data, run the model and compute loss, accuracy and macro-accuracy.

Args:

    batch (torch.Tensor): The batch.
    batch_idx (int): The batch index.

Returns:

    loss (torch.Tensor): The loss.

Raises:

    None.


### _class_ sflizard.Graph_model.graph_model.GraphCustom
Bases: `Module`

Custom graph model adding linear layers before and after the graph layers.

Args:
    
    dim_in (int): The dimension of the input.
    dim_h (int): The dimension of the hidden layers.
    dim_out (int): The dimension of the output.
    num_layers (int): The number of graph layers.
    layer_type (torch.nn.Module): The type of graph layer to use.
    custom_input_layer (int): The number of linear input layers. Default to 0.
    custom_input_hidden (int): The dimension of the linear input hidden layers. Default to 8.
    custom_output_layer (int): The number of linear output layers. Default to 0.
    custom_output_hidden (int): The dimension of the linear output hidden layers. Default to 8.
    custom_wide_connections (bool): Whether to use wide connections. Default to False.
    dropout (float): The dropout rate. Default to 0.0.

Raises:

    None.

#### forward
Perform a forward pass. Calls the model with the input tensor (node feature vector) and the edge index as input.

Args:

    x (torch.Tensor): The input tensor.
    edge_index (torch.Tensor): The edge index tensor.

Returns:

    output (torch.Tensor): The output tensor.

Raises:

    None.


