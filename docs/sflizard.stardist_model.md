# sflizard.stardist_model package

    * [Submodules](sflizard.stardist_model.md#submodules)


    * [sflizard.stardist_model.models_utils module](sflizard.stardist_model.md#module-sflizard.stardist_model.models_utils)


        * [`ClassL1BCELoss`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.ClassL1BCELoss)


            * [`ClassL1BCELoss.forward()`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.ClassL1BCELoss.forward)


            * [`ClassL1BCELoss.training`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.ClassL1BCELoss.training)


        * [`MyL1BCELoss`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.MyL1BCELoss)


            * [`MyL1BCELoss.forward()`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.MyL1BCELoss.forward)


            * [`MyL1BCELoss.training`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.MyL1BCELoss.training)


        * [`UNetStar`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.UNetStar)


            * [`UNetStar.compute_star_label()`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.UNetStar.compute_star_label)


            * [`UNetStar.forward()`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.UNetStar.forward)


            * [`UNetStar.training`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.UNetStar.training)


        * [`double_conv`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.double_conv)


            * [`double_conv.forward()`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.double_conv.forward)


            * [`double_conv.training`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.double_conv.training)


        * [`down`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.down)


            * [`down.forward()`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.down.forward)


            * [`down.training`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.down.training)


        * [`inconv`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.inconv)


            * [`inconv.forward()`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.inconv.forward)


            * [`inconv.training`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.inconv.training)


        * [`outconv`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.outconv)


            * [`outconv.forward()`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.outconv.forward)


            * [`outconv.training`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.outconv.training)


        * [`up`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.up)


            * [`up.forward()`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.up.forward)


            * [`up.training`](sflizard.stardist_model.md#sflizard.stardist_model.models_utils.up.training)


    * [sflizard.stardist_model.stardist_model module](sflizard.stardist_model.md#module-sflizard.stardist_model.stardist_model)


        * [`Stardist`](sflizard.stardist_model.md#sflizard.stardist_model.stardist_model.Stardist)


            * [`Stardist.configure_optimizers()`](sflizard.stardist_model.md#sflizard.stardist_model.stardist_model.Stardist.configure_optimizers)


            * [`Stardist.forward()`](sflizard.stardist_model.md#sflizard.stardist_model.stardist_model.Stardist.forward)


            * [`Stardist.training`](sflizard.stardist_model.md#sflizard.stardist_model.stardist_model.Stardist.training)


            * [`Stardist.training_epoch_end()`](sflizard.stardist_model.md#sflizard.stardist_model.stardist_model.Stardist.training_epoch_end)


            * [`Stardist.training_step()`](sflizard.stardist_model.md#sflizard.stardist_model.stardist_model.Stardist.training_step)


            * [`Stardist.validation_epoch_end()`](sflizard.stardist_model.md#sflizard.stardist_model.stardist_model.Stardist.validation_epoch_end)


            * [`Stardist.validation_step()`](sflizard.stardist_model.md#sflizard.stardist_model.stardist_model.Stardist.validation_step)


    * [Module contents](sflizard.stardist_model.md#module-sflizard.stardist_model)

## Modules

This package contains the following modules:

* [sflizard.run_hovernet_metric_tool](#module-sflizard.run_hovernet_metric_tool)

* [sflizard.run_test_pipeline](#module-sflizard.run_test_pipeline)

* [sflizard.training](#module-sflizard.training)

## sflizard.stardist_model.models_utils


### _class_ sflizard.stardist_model.models_utils.ClassL1BCELoss(class_weights: Tensor, scale: list = [1, 1, 1])
Bases: `Module`

Custom loss function for StarDist. Improvement of MyL1BCELoss by adding class loss


#### forward(prediction: Tensor, obj_probabilities: Tensor, target_dists: Tensor, classes: Tensor)
Forward pass.

Args:

    prediction (torch.Tensor): The prediction.
    obj_probabilities (torch.Tensor): The probabilities map true values.
    target_dists (torch.Tensor): The distances map true values.
    classes (torch.Tensor): The classes map true values.

Returns:

    torch.Tensor: The loss.

Raises:

    None.




### _class_ sflizard.stardist_model.models_utils.MyL1BCELoss(scale=[1, 1])
Bases: `Module`

Custom loss function for StarDist. source: [https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/distance_loss.py](https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/distance_loss.py)


#### forward(prediction, obj_probabilities, target_dists)
Defines the computation performed at every call.

Should be overridden by all subclasses.

**NOTE**: Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.




### _class_ sflizard.stardist_model.models_utils.UNetStar(n_channels: int, n_rays: int, n_classes: int | None = None, last_layer_out: bool = False)
Bases: `Module`

UNetStar model. source: [https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_model.py](https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_model.py)


#### compute_star_label(image: array, dist: Tensor, prob: Tensor, get_points: bool = False)
Compute the star label of images according dist and prob.

Args:

    image (np.array): The image.
    dist (torch.Tensor): The distances map.
    prob (torch.Tensor): The probabilities map.
    get_points (bool, optional): If True, the points are returned. Defaults to False.

Returns:

    star_labels (np.array): The star label.

Raises:

    None.


#### forward(x: Tensor)
Forward the input in the network.

Args:

    x (torch.Tensor): The input.

Returns:

    List[torch.Tensor]: The output:


        * [0]: The distances map (torch.Tensor).


        * [1]: The probabilities map (torch.Tensor).


        * [2]: The classes map (torch.Tensor).

Raises:

    None.




### _class_ sflizard.stardist_model.models_utils.double_conv(in_ch, out_ch)
Bases: `Module`

source: [https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_parts_gn.py](https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_parts_gn.py)


#### forward(x)
Forward the input in the network.




### _class_ sflizard.stardist_model.models_utils.down(in_ch, out_ch)
Bases: `Module`

source: [https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_parts_gn.py](https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_parts_gn.py)


#### forward(x)
Forward the input in the network.




### _class_ sflizard.stardist_model.models_utils.inconv(in_ch, out_ch)
Bases: `Module`

source: [https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_parts_gn.py](https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_parts_gn.py)


#### forward(x)
Forward the input in the network.




### _class_ sflizard.stardist_model.models_utils.outconv(in_ch, out_ch)
Bases: `Module`

source: [https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_parts_gn.py](https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_parts_gn.py)


#### forward(x)
Forward the input in the network.




### _class_ sflizard.stardist_model.models_utils.up(in_ch, out_ch)
Bases: `Module`

source: [https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_parts_gn.py](https://github.com/ASHISRAVINDRAN/stardist_pytorch/blob/master/unet/unet_parts_gn.py)


#### forward(x1, x2)
Forward the input in the network.



## sflizard.stardist_model.stardist_model module


### _class_ sflizard.stardist_model.stardist_model.Stardist(learning_rate: float = 0.0001, input_size: int = 540, in_channels: int = 3, n_rays: int = 32, n_classes: int = 1, loss_power_scaler: float = 0.0, seed: int = 303, device: str = 'cpu', wandb_log: bool = False, max_epochs: int = 200)
Bases: `LightningModule`

Stardist model class.


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


#### forward(x: Tensor)
Forward pass.

Args:

    x (torch.Tensor): The input tensor.

Returns:

    x (torch.Tensor): The output tensor.

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