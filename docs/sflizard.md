# sflizard package

## Subpackages


* [sflizard.Graph_model package](sflizard.Graph_model.md)


* [sflizard.data_utils package](sflizard.data_utils.md)


* [sflizard.pipeline package](sflizard.pipeline.md)


* [sflizard.stardist_model package](sflizard.stardist_model.md)


## Modules

This package contains the following modules:

* [sflizard.run_hovernet_metric_tool](#module-sflizard.run_hovernet_metric_tool)

* [sflizard.run_test_pipeline](#module-sflizard.run_test_pipeline)

* [sflizard.training](#module-sflizard.training)

## sflizard.run_hovernet_metric_tool

TODO : description and usage

## sflizard.run_test_pipeline

TODO : description and usage

## sflizard.training

TODO : description and usage.

The following functions are present in the module:

* sflizard.training.full_training
* sflizard.training.init_graph_training
* sflizard.training.init_stardist_training

### sflizard.training.full_training

Train the model on the whole dataset.

Args:

    args (argparse.Namespace): the arguments from the command line.

Returns:

    None.

Raises:

    ValueError: if the model is not implemented.


### sflizard.training.init_graph_training
Init the training for the graphSage model.

Args:

    args (argparse.Namespace): the arguments from the command line.

Returns:

    tuple: tuple containing:

        dm (LizardGraphDataModule): the datamodule.
        model (Graph): the model.
        callbacks (List[pl.callbacks.Callback]): the callbacks.

Raises:

    None.


### sflizard.training.init_stardist_training
Init the training for the stardist model.

Args:

    args (argparse.Namespace): the arguments from the command line.
    device (Union[str, torch.device]): the device to use.
    debug (bool): if True, print debug messages.

Returns:

    tuple: tuple containing:

        dm (LizardDataModule): the datamodule.
        model (Stardist): the model.
        callbacks (List[pl.callbacks.Callback]): the callbacks.

Raises:

    None.

