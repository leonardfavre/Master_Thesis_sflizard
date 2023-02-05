# sflizard.stardist_model package

## Modules

This package contains the following modules:

* [sflizard.stardist_model.stardist_model](sflizard.stardist_model.md#sflizardstardist_modelstardist_model)

* [sflizard.stardist_model.models_utils](sflizard.stardist_model.md#sflizardstardist_modelmodels_utils)


## sflizard.stardist_model.stardist_model


### _class_ sflizard.stardist_model.stardist_model.Stardist
Bases: `LightningModule`

Stardist model lightning module class.
This class define the lightning module containing the Stardist model. 

Args:

    - learning_rate (float): The learning rate. Default to 1e-4.
    - input_size (int): The input size. Default to 540.
    - in_channels (int): The number of input channels. Default to 3.
    - n_rays (int): The number of rays. Default to 32.
    - n_classes (int): The number of classes. Default to 1.
    - loss_power_scaler (float): The loss power scaler. Default to 0.0.
    - seed (int): The seed. Default to 303.
    - device (str): The device. Default to cpu.
    - wandb_log (bool): Whether to log to wandb. Default to False.
    - max_epochs (int): The maximum number of epochs. Default to 200.

Raises:

    None.




#### configure_optimizers
Configure optimizers and schedulers used for training. Currently implemented is Adam optimizer and LinearWarmupCosineAnnealingLR scheduler.

Args:

    None.

Returns:

    - tuple: tuple containing:
        * optimizers (List[torch.optim.Optimizer]): The optimizers.
        * schedulers (List[torch.optim.lr_scheduler._LRScheduler]): The schedulers.

Raises:

    None.


#### forward
Perform a forward pass. Calls the model with the input tensor as input.

Args:

    - x (torch.Tensor): The input tensor.

Returns:

    - x (torch.Tensor): The output tensor.

Raises:

    None.




#### training_epoch_end
Run at the end of a training epoch. Log loss to wandb if wandb_log was set to True.

Args:

    - outputs (List[torch.Tensor]): The outputs.

Returns:

    None.

Raises:

    None.


#### training_step
Run at each step of training. Get the current batch data, run the model and compute loss.

Args:

    - batch (torch.Tensor): The batch.
    - batch_idx (int): The batch index.

Returns:

    - loss (torch.Tensor): The loss.

Raises:

    None.


#### validation_epoch_end
Run at the end of a validation epoch. Log loss to wandb if wandb_log was set to True.

Args:

    - outputs (List[torch.Tensor]): The outputs.

Returns:

    None.

Raises:

    None.


#### validation_step
Run at each step of validation. Get the current batch data, run the model and compute loss.

Args:

    - batch (torch.Tensor): The batch.
    - batch_idx (int): The batch index.

Returns:

    - loss (torch.Tensor): The loss.

Raises:

    None.


## sflizard.stardist_model.models_utils


### _class_ sflizard.stardist_model.models_utils.ClassL1BCELoss
Bases: `Module`

Custom loss function for StarDist. Improvement of MyL1BCELoss by adding class loss.

Args:

    - class_weights (torch.Tensor): Weights for each class.
    - scale (list, optional): Scale for each loss. Defaults to [1, 1, 1].

Raises:

    None.


#### forward
Loss forward pass.

Args:

    - prediction (torch.Tensor): The prediction.
    - obj_probabilities (torch.Tensor): The probabilities map true values.
    - target_dists (torch.Tensor): The distances map true values.
    - classes (torch.Tensor): The classes map true values.

Returns:

    - torch.Tensor: The loss.

Raises:

    None.




### _class_ sflizard.stardist_model.models_utils.MyL1BCELoss
Bases: `Module`

Custom loss function for StarDist. source: [https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/distance_loss.py](https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/distance_loss.py)


Args:

    - scale (list, optional): Scale for each loss. Defaults to [1, 1].

Raises:

    None.


#### forward
Loss forward pass.

Args:

    - prediction (torch.Tensor): The prediction.
    - obj_probabilities (torch.Tensor): The probabilities map true values.
    - target_dists (torch.Tensor): The distances map true values.

Returns:

    - torch.Tensor: The loss.

Raises:

    None.




### _class_ sflizard.stardist_model.models_utils.UNetStar
Bases: `Module`

UNetStar model. source: [https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_model.py](https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_model.py)
Modified to add class prediction and to add last global layer as an output.


Args:

    - n_channels (int): Number of channels.
    - n_rays (int): Number of rays.
    - n_classes (int, optional): Number of classes. Defaults to None.
    - last_layer_out (bool, optional): If True, the last layer is returned. Defaults to False.

Raises:

    None.

#### compute_star_label
Compute the star label of images according dist and prob.

Args:

    - image (np.array): The image.
    - dist (torch.Tensor): The distances map.
    - prob (torch.Tensor): The probabilities map.
    - get_points (bool, optional): If True, the points are returned. Defaults to False.

Returns:

    - star_labels (np.array): The star label.

Raises:

    None.


#### forward
Forward the input in the network.

Args:

    - x (torch.Tensor): The input.

Returns:

    - List[torch.Tensor]: The output:
        * [0]: The distances map (torch.Tensor).
        * [1]: The probabilities map (torch.Tensor).
        * [2]: The classes map (torch.Tensor).

Raises:

    None.




### _class_ sflizard.stardist_model.models_utils.double_conv
Bases: `Module`

source: [https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_parts_gn.py](https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_parts_gn.py)

Args:

    - in_ch (int): input size.
    - out_ch (int): output size.

Raises:

    None.

#### forward
Forward the input in the network.

Args:

    - x (torch.Tensor): The input.

Returns:

    - x (torch.Tensor): The output.

Raises:

    None.


### _class_ sflizard.stardist_model.models_utils.down
Bases: `Module`

source: [https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_parts_gn.py](https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_parts_gn.py)

Args:

    - in_ch (int): input size.
    - out_ch (int): output size.

Raises:

    None.

#### forward
Forward the input in the network.

Args:

    - x (torch.Tensor): The input.

Returns:

    - x (torch.Tensor): The output.

Raises:

    None.


### _class_ sflizard.stardist_model.models_utils.inconv
Bases: `Module`

source: [https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_parts_gn.py](https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_parts_gn.py)

Args:

    - in_ch (int): input size.
    - out_ch (int): output size.

Raises:

    None.

#### forward
Forward the input in the network.

Args:

    - x (torch.Tensor): The input.

Returns:

    - x (torch.Tensor): The output.

Raises:

    None.


### _class_ sflizard.stardist_model.models_utils.outconv
Bases: `Module`

source: [https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_parts_gn.py](https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_parts_gn.py)

Args:

    - in_ch (int): input size.
    - out_ch (int): output size.

Raises:

    None.

#### forward
Forward the input in the network.

Args:

    - x (torch.Tensor): The input.

Returns:

    - x (torch.Tensor): The output.

Raises:

    None.


### _class_ sflizard.stardist_model.models_utils.up
Bases: `Module`

source: [https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_parts_gn.py](https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_parts_gn.py)

Args:

    - in_ch (int): input size.
    - out_ch (int): output size.

Raises:

    None.

#### forward
Forward the input in the network.

Args:

    - x1 (torch.Tensor): The input.
    - x2 (torch.Tensor): The input.

Returns:

    - x (torch.Tensor): The output.

Raises:

    None.

