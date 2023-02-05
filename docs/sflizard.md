# sflizard package

## Subpackages


* [sflizard.Graph_model package](sflizard.Graph_model.md)


* [sflizard.data_utils package](sflizard.data_utils.md)


* [sflizard.pipeline package](sflizard.pipeline.md)


* [sflizard.stardist_model package](sflizard.stardist_model.md)


## Modules

This package contains the following modules:

* [sflizard.run_hovernet_metric_tool](#module-sflizardrun_hovernet_metric_tool)

* [sflizard.run_test_pipeline](#module-sflizardrun_test_pipeline)

* [sflizard.training](#module-sflizardtraining)

## sflizard.run_hovernet_metric_tool

This script can be called from command line to launch the execution of the HoverNet metric tool.

The following constant are defined in the file:

    TEST_MODE: if set to True, `WEIGHTS_PATH` will be used. Otherwise, `WEIGHTS_SELECTOR` will be.
    WEIGHTS_SELECTOR: dict of selector for checkpoint selection.
    WEIGHTS_PATH: dict of checkpoint paths to test.
    MODE: dataset to test. Can be "valid" or "test".
    DISTANCE: distance to use for graph creation.
    X_TYPE: type of node feature vector.

**Weights selectors:**

The following entry can be used to select the checkpoints to test:

    model: list of models to test. Example: ["graph_sage", "graph_gin"].
    dimh: list of hidden dimension to test. Example: [512, 1024].
    num_layers: list of num_layers to test. Example: [2, 4].
    heads: list of heads dimension to test. Example: [1, 2].
    custom_combinations: list of custom combinations to test. These combination are part of the name of the checkpoints and represent other hyperparameters not directly implemented. This is usefull for example on custom graph using linear layers, or models using dropout. Example: ["0.1-0-0-3-16-wide"]

**Weights path:**

Weights paths need to be provided in the following form: `{name: path}`. Example:

```
{
"mod1-b-0.0-t": "weights/graph_custom-1024-4-4ll-45-0-0-3-16-wide-0.0005-acc-epoch=103-val_acc=0.7817.ckpt",
}
```

For a more in-depth description of the hovernet metric tool, see [sflizard.pipeline.hovernet_metric_tool](sflizard.pipeline.md#sflizard.pipeline.hovernet_metric_tool).

## sflizard.run_test_pipeline

This script can be called from command line to launch the test pipeline that test the Stardist model and the graph neural network together.

The available arguments are the following:

    -vdp / --valid_data_path: Path to the .pkl file containing the data. Default to "data/Lizard_dataset_extraction/data_final_split_valid.pkl".
    -tdp / --test_data_path: Path to the .pkl file containing the test data. Default to "data/Lizard_dataset_extraction/data_final_split_test.pkl".
    -swp / --stardist_weights_path: Path to the file containing the stardist model weights. Default to "weights/final3_stardist_crop-cosine_200epochs_1.0losspower_0.0005lr.ckpt".
    -gwp / --graph_weights_path: Path to the file containing the graph model weights (multiple). Default to "weights/graph_custom-1024-4-4ll-45-0-0-3-16-wide-0.0005-acc-epoch=103-val_acc=0.7817.ckpt".
    -nr / --n_rays: Number of rays to use in the stardist model. Default to 32.
    -nc / --n_classes: Number of classes to use in the stardist model (1 = no classification). Default to 7.
    -bs / --batch_size: Batch size to use during training.Default to 1.
    -s / --seed: Seed to use for the data split. Default to 303.
    -od / --output_dir: Path to the directory where the results will be saved. Default to "./output/stardist_pipeline/final_result/".
    -itd / --imgs_to_display: Number of images to display in the report. Default to 30.
    -d / --distance: Distance to use for the graph. Default to 45.
    -m / --mode: Mode to use for the test ('valid' or 'test').Default to "test".

For a more in-depth description of the test pipeline, see [sflizard.pipeline.test_pipeline](sflizard.pipeline.md#sflizardpipelinetest_pipeline).

## sflizard.training

This script can be called from command line to launch a training for the Stardist model or the graph network model.

The available arguments are:

    - tdp / --train_data_path: Path to the .pkl file containing the train data. Default to "data/Lizard_dataset_extraction/data_test_split_train.pkl".
    - vdp / --valid_data_path: Path to the .pkl file containing the validation data. Default to "data/Lizard_dataset_extraction/data_test_split_valid.pkl".
    - tp / --test_data_path: Path to the .pkl file containing the test data. Default to "data/Lizard_dataset_extraction/data_test_split_test.pkl".
    - m / --model: Model to train. Can be 'stardist', 'graph_gat', 'graph_custom', 'graph_gcn', 'graph_sage' or 'graph_gin'. Default to "graph_sage".
    - bs / --batch_size: Batch size to use for the dataloaders. Default to 64.
    - nw / --num_workers: Number of workers to use for the dataloaders. Default to 8.
    - is / --input_size: Input size to use for the dataloaders. Default to 540.
    - lr / --learning_rate: Learning rate to use for the optimizer. Default to 5e-4.
    - s / --seed: Seed to use for randomization. Default to 303.
    - nc / --num_classes: Number of classes to use for the classification problem. Default to 7.
    - lps / --loss_power_scaler: Loss scaler to use for the stardist model. Default to 1.0.
    - dh / --dimh: Dimension of the hidden layer in the grap model. Default to 1024.
    - nl / --num_layers: Number of layers in the grap model. Default to 4.
    - he / --heads: Number of heads in the graph gat model. Default to 8.
    - cil / --custom_input_layer: Custom linear input layer number in the custom graph model. Default to 0.
    - cih / --custom_input_hidden: Custom linear input hidden layer size in the custom graph model. Default to 8.
    - col / --custom_output_layer: Custom linear output layer number in the custom graph model. Default to 0.
    - coh / --custom_output_hidden: Custom linear output hidden layer size in the custom graph model. Default to 7.
    - cwc / --custom_wide_connections: Custom wide connections in the custom graph model. Default to False.
    - do / --dropout: Dropout to use for the graph model. Default to 0.0.
    - xt / --x_type: Type of the input in the grap model. Default to "4ll".
    - d / --distance: Distance to use for the graph model. Default to 45.
    - sn / --save_name: Name to add to the saved model. Default to "".

Additionaly, the following constant are defined in the file:

    IN_CHANNELS: Number of channels of input images. Default to 3.
    N_RAYS: Number of rays for the Stardist model. Default to 32.
    STARDIST_CHECKPOINT: Stardist checkpoint to use when training graph model. Default to "weights/final3_stardist_crop-cosine_200epochs_1.0losspower_0.0005lr.ckpt".

### sflizard.training.full_training

Train the model on the whole dataset.
Initiate the model, dataloader, callbacks. Creates the pytorch-lightning Trainer, fit the model on data and save final checkpoint.

Args:

    - args (argparse.Namespace): the arguments from the command line.

Returns:

    None.

Raises:

    ValueError: if the model is not implemented.


### sflizard.training.init_graph_training
Init the training for the graphSage model.
Creates a LizardGraphDataModule with provided data path and other provided arguments (see [sflizard.data_utils.graph_module](sflizard.data_utils.md#sflizarddata_utilsgraph_module)).
Creates a Graph Lightning module containing the model (see [sflizard.Graph_model.graph_model](sflizard.Graph_model.md#sflizardGraph_modelgraph_model)).
Creates 3 callbacks: loss, accuracy and macro-accuracy.

Args:

    - args (argparse.Namespace): the arguments from the command line.

Returns:

    - tuple: tuple containing:
        * dm (LizardGraphDataModule): the datamodule.
        * model (Graph): the model.
        * callbacks (List[pl.callbacks.Callback]): the callbacks.

Raises:

    None.


### sflizard.training.init_stardist_training
Init the training for the stardist model.
Creates a LizardDataModule with provided data path and other provided arguments (see [sflizard.data_utils.data_module](sflizard.data_utils.md#sflizarddata_utilsdata_module)).
Creates a Stardist Lightning module containing the model (see [sflizard.stardist_model.stardist_model](sflizard.stardist_model.md#sflizardstardist_modelstardist_model)).
Creates 1 callbacks for loss.

Args:

    - args (argparse.Namespace): the arguments from the command line.
    - device (Union[str, torch.device]): the device to use.
    - debug (bool): if True, print debug messages. Default to False.

Returns:

    - tuple: tuple containing:
        * dm (LizardDataModule): the datamodule.
        * model (Stardist): the model.
        * callbacks (List[pl.callbacks.Callback]): the callbacks.

Raises:

    None.

