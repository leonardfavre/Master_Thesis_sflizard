# sflizard.pipeline package

## Modules

This package contains the following modules:

* [sflizard.pipeline.test_pipeline](sflizard.pipeline.md#sflizard.pipeline.test_pipeline)

* [sflizard.pipeline.pipeline_utils](sflizard.pipeline.md#sflizard.pipeline.pipeline_utils)

* [sflizard.pipeline.segmentation_metric_tool](sflizard.pipeline.md#sflizard.pipeline.segmentation_metric_tool)

* [sflizard.pipeline.report](sflizard.pipeline.md#sflizard.pipeline.report)

* [sflizard.pipeline.hovernet_metric_tool](sflizard.pipeline.md#sflizard.pipeline.hovernet_metric_tool)


## sflizard.pipeline.test_pipeline


### _class_ sflizard.pipeline.test_pipeline.TestPipeline
Bases: `object`

A pipeline to test the model. Uses the Stardist checkpoint given to compute segmentation on data, if graph network checkpoint is given, improve resulting classification using the provided network(s).

Provides logging and report options.

Args:

    - valid_data_path (str): The path to the valid data.
    - test_data_path (str): The path to the test data.
    - stardist_weights_path (str): The path to the stardist weights.
    - graph_weights_path (List[str]): The path to the graph weights.
    - graph_distance (int): The distance to use for the graph.
    - n_rays (int): The number of rays to use for stardist.
    - n_classes (int): The number of classes.
    - batch_size (int): The batch size to use.
    - seed (int): The seed to use for constant randomization.
    - mode (str): The mode to use (test or valid).

Raises:

    None.

#### test
Run the pipeline and test the model.

Args:

    - output_dir (str): The path to the output directory for report and images. Default to None.
    - imgs_to_display (int): The number of images to display in the report. Default to 0.

Returns:

    None.

Raises:

    None.


## sflizard.pipeline.pipeline_utils


### sflizard.pipeline.pipeline_utils.get_class_map_from_graph
Get the class map from the graph prediction. Uses the provided instance map as a base and assign class predicted by the graph to the segmented cells using position stored in graph.

If the graph prediction is empty, will print "problem with graph prediction" to report the issue, but will use provided class map as replacement to avoid crashing pipeline.

If the point in the graph doesn't correspond to a cell in the instance map, will print "problem between graph and stardist" to report the issue, and will ignore the prediction.

Args:

    - graph (list): The graph.
    - inst_map (list): The instance map.
    - graph_pred (list): The graph prediction.
    - class_pred (list): The class prediction.

Returns:

    - class_maps (np.array): The class map.

Raises:

    None.


### sflizard.pipeline.pipeline_utils.improve_class_map
Improve the class map by assigning the same class to each segmented object.
The provided class map is used with the points to assign each entity in the segmentation map a class.

Args:

    - class_map (np.array): The class map.
    - predicted_masks (np.array): The predicted masks.
    - points (np.array): The points of the cells detected in the masks.

Returns:

    - improved_class_map (np.array): The improved class map.

Raises:

    None.


## sflizard.pipeline.segmentation_metric_tool


### _class_ sflizard.pipeline.segmentation_metric_tool.SegmentationMetricTool
Bases: `object`

A tool to compute metrics for segmentation.
This tool uses the `matching_dataset` function provided by Stardist to compute metrics about segmentation quality.

If n_classes > 1, will also compute per-class metrics.

This tool is able to log the results nicely, using the `rich` library. Additionaly, if the `PRINT_LATEX_STRING` constant is set to True, the results tables are printed in LateX format for easy integration into a LateX document. 

Args:

    - n_classes (int): The number of classes.
    - device (str): The device to use.
    - console (Console): The rich console.

Raises:

    None.

#### add_batch
Add a batch of instance map data to the metric tool.

Args:

    - batch_idx (int): The batch index.
    - true_masks (np.array): The true masks.
    - pred_masks (np.array): The predicted masks.

Returns:

    None.

Raises:

    None.


#### add_batch_class
Add a batch of class map data to the metric tool. The class maps will be divided in n_classes uni-class class map (one class map for each class, with only one class visible and the other set to 0).

Args:

    - batch_idx (int): The batch index.
    - true_class_map (np.array): The true class map.
    - pred_class_map (np.array): The predicted class map.

Returns:

    None.

Raises:

    None.


#### compute_metrics
Compute the metrics.
Uses the given instance maps and class maps to compute the `matching_dataset` function. To compute metrics on the class maps, the instance maps are mandatory to allow the `matching_dataset` function to differentiate between the different cells.

Args:

    None.

Returns:

    None.

Raises:

    None.


#### log_results(console: Console)
Log the results in rich tables. if the `PRINT_LATEX_STRING` constant is set to True, will also print results in LateX format.

Args:

    - console (Console): The rich console.

Returns:

    None.

Raises:

    None.

## sflizard.pipeline.report


### _class_ sflizard.pipeline.report.ReportGenerator
Bases: `object`

MD report generator.
This tool creates a MD report containing metrics and shows <imgs_to_display> example images of the process.

Args:
    
    - output_dir (str): The output directory.
    - imgs_to_display (int): The number of images to display.
    - n_classes (int): The number of classes.
    - console (Console): The rich console.

Raises:
    
    None.

#### add_batch
Add a batch of images to the report. All images are in np.array format, except for original images that are in torch.Tensor.

Args:

    - images (list): The images.
    - true_masks (list): The true masks.
    - pred_masks (list): The predicted masks.
    - true_class_map (list): The true class map.
    - pred_class_map (list): The predicted class map.
    - pred_class_map_improved (list): The improved predicted class map.
    - graphs (list): The graphs.
    - graphs_class_map (list): The graphs class map.

Returns:

    None.

Raises:

    None.


#### add_final_metrics
Add final metrics to the report.
These are the metrics computed by [sflizard.pipeline.segmentation_metric_tool](sflizard.pipeline.md#sflizard.pipeline.segmentation_metric_tool).

Args:

    - segmentation_metric (dict): The segmentation metric.
    - segmentation_classification_metric (dict): The segmentation classification metric.
    - graph_segmentation_classification_metric (dict): The graph segmentation classification metric.

Returns:

    None.

Raises:

    None.


#### generate_md
Generate a markdown file with the report. Will create tables and save images, then generate a MD file.

Args:

    None.

Returns:

    None.

Raises:

    None.


## sflizard.pipeline.hovernet_metric_tool

### _class_ sflizard.pipeline.hovernet_metric_tool.HoverNetMetricTool
Bases: `object`

Tool to evaluate the performance of Graph model on the Lizard dataset using HoverNet `compute_stats` function. To be able to use this function, the tool will save the graph classification result in a `.mat` file in a format compatible with the function.

2 modes are available:
* provide dict of selectors `weights_selector`, consisting of list of hyperparameter. The tool will look in folders defined in `CHECKPOINT_PATH` and test the models corresponding to the selectors. The models find can be distinguish between the following checkpoints type: accuracy, macro-accuracy, loss and final.
* provide directly the path to the graph checkpoints in `paths`

If `paths` is not empty, the tool will run in quick mode and only output to the terminal.

Otherwise, the tool will create a log.txt file containing results and output of the test runs. It will also create a pkl file containing metric results stored in a table according to the `weights_selector` options.

`weights_selector` options are the following:

    model: base model of the graph network.
    dimh: dimension of hidden layers.
    num_layers: number of layers.
    heads: number of heads (only for GAT model).
    custom_combinations: additionnal combination of parameters.



The results will be saved in tables with the following hierarchy:

    results[models][checkpoint_type][combination][heads][dim_h][num_layers]

The following constant are defined:

    TRAIN_DATA_PATH: Path to the train dataset pkl file. Default to "data/Lizard_dataset_extraction/data_final_split_train.pkl".
    VALID_DATA_PATH: Path to the valid dataset pkl file. Default to "data/Lizard_dataset_extraction/data_final_split_valid.pkl".
    TEST_DATA_PATH: Path to the test dataset pkl file. Default to "data/Lizard_dataset_extraction/data_final_split_test.pkl".
    SEED: Seed for randomization. Default to 303.
    STARDIST_CHECKPOINT: Path to the stardist checkpoint file. Default to "models/final3_stardist_crop-cosine_200epochs_1.0losspower_0.0005lr.ckpt".
    CHECKPOINT_PATH: Path to folders containing graph checkpoints. Default to ["models/", "models/cp_acc_graph/", "models/loss_cb_graph/"].
    TRUE_DATA_PATH_START: Path to the true data folder containing label separate files. Default to "data/Lizard_dataset_split/patches/Lizard_Labels_".
    TEST_DROPOUT: Boolean to force use of dropout (usefull to test models using dropout). Default to True.

Args:
    
    - mode (str): "valid" or "test" depending on the dataset to use. Default to "valid".
    - weights_selector (dict): dict of list of model, dimh, num_layers and heads to test. Default to {'dimh': [], 'heads': [], 'model': [], 'num_layers': []}.
    - distance (int): distance used in creation of graph. Default to 45.
    - x_type (str): type of node feature vector. Default to "4ll"
    - paths (dict): dict of paths to the model checkpoints to test. Default to {}.

Raises:
    None.