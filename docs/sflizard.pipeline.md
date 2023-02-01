# sflizard.pipeline package

    * [Submodules](sflizard.pipeline.md#submodules)


    * [sflizard.pipeline.hovernet_metric_tool module](sflizard.pipeline.md#module-sflizard.pipeline.hovernet_metric_tool)


        * [`HoverNetMetricTool`](sflizard.pipeline.md#sflizard.pipeline.hovernet_metric_tool.HoverNetMetricTool)


            * [`HoverNetMetricTool.clean_folder()`](sflizard.pipeline.md#sflizard.pipeline.hovernet_metric_tool.HoverNetMetricTool.clean_folder)


            * [`HoverNetMetricTool.get_weights_path()`](sflizard.pipeline.md#sflizard.pipeline.hovernet_metric_tool.HoverNetMetricTool.get_weights_path)


            * [`HoverNetMetricTool.init_graph_inference()`](sflizard.pipeline.md#sflizard.pipeline.hovernet_metric_tool.HoverNetMetricTool.init_graph_inference)


            * [`HoverNetMetricTool.init_result_table()`](sflizard.pipeline.md#sflizard.pipeline.hovernet_metric_tool.HoverNetMetricTool.init_result_table)


            * [`HoverNetMetricTool.run_hovernet_metric_tool()`](sflizard.pipeline.md#sflizard.pipeline.hovernet_metric_tool.HoverNetMetricTool.run_hovernet_metric_tool)


            * [`HoverNetMetricTool.save_mat()`](sflizard.pipeline.md#sflizard.pipeline.hovernet_metric_tool.HoverNetMetricTool.save_mat)


            * [`HoverNetMetricTool.save_result_in_table()`](sflizard.pipeline.md#sflizard.pipeline.hovernet_metric_tool.HoverNetMetricTool.save_result_in_table)


            * [`HoverNetMetricTool.save_result_to_file()`](sflizard.pipeline.md#sflizard.pipeline.hovernet_metric_tool.HoverNetMetricTool.save_result_to_file)


    * [sflizard.pipeline.pipeline_utils module](sflizard.pipeline.md#module-sflizard.pipeline.pipeline_utils)


        * [`get_class_map_from_graph()`](sflizard.pipeline.md#sflizard.pipeline.pipeline_utils.get_class_map_from_graph)


        * [`improve_class_map()`](sflizard.pipeline.md#sflizard.pipeline.pipeline_utils.improve_class_map)


        * [`merge_stardist_class_together()`](sflizard.pipeline.md#sflizard.pipeline.pipeline_utils.merge_stardist_class_together)


        * [`rotate_and_pred()`](sflizard.pipeline.md#sflizard.pipeline.pipeline_utils.rotate_and_pred)


    * [sflizard.pipeline.report module](sflizard.pipeline.md#module-sflizard.pipeline.report)


        * [`ReportGenerator`](sflizard.pipeline.md#sflizard.pipeline.report.ReportGenerator)


            * [`ReportGenerator.add_batch()`](sflizard.pipeline.md#sflizard.pipeline.report.ReportGenerator.add_batch)


            * [`ReportGenerator.add_final_metrics()`](sflizard.pipeline.md#sflizard.pipeline.report.ReportGenerator.add_final_metrics)


            * [`ReportGenerator.generate_md()`](sflizard.pipeline.md#sflizard.pipeline.report.ReportGenerator.generate_md)


    * [sflizard.pipeline.segmentation_metric_tool module](sflizard.pipeline.md#module-sflizard.pipeline.segmentation_metric_tool)


        * [`SegmentationMetricTool`](sflizard.pipeline.md#sflizard.pipeline.segmentation_metric_tool.SegmentationMetricTool)


            * [`SegmentationMetricTool.add_batch()`](sflizard.pipeline.md#sflizard.pipeline.segmentation_metric_tool.SegmentationMetricTool.add_batch)


            * [`SegmentationMetricTool.add_batch_class()`](sflizard.pipeline.md#sflizard.pipeline.segmentation_metric_tool.SegmentationMetricTool.add_batch_class)


            * [`SegmentationMetricTool.compute_metrics()`](sflizard.pipeline.md#sflizard.pipeline.segmentation_metric_tool.SegmentationMetricTool.compute_metrics)


            * [`SegmentationMetricTool.log_results()`](sflizard.pipeline.md#sflizard.pipeline.segmentation_metric_tool.SegmentationMetricTool.log_results)


    * [sflizard.pipeline.test_pipeline module](sflizard.pipeline.md#module-sflizard.pipeline.test_pipeline)


        * [`TestPipeline`](sflizard.pipeline.md#sflizard.pipeline.test_pipeline.TestPipeline)


            * [`TestPipeline.test()`](sflizard.pipeline.md#sflizard.pipeline.test_pipeline.TestPipeline.test)


    * [Module contents](sflizard.pipeline.md#module-sflizard.pipeline)


## Submodules

## sflizard.pipeline.hovernet_metric_tool module


### _class_ sflizard.pipeline.hovernet_metric_tool.HoverNetMetricTool(mode: str = 'valid', weights_selector: dict = {'dimh': [], 'heads': [], 'model': [], 'num_layers': []}, distance: int = 45, x_type: str = 'll+c', paths: dict = {})
Bases: `object`

Tool to evaluate the performance of Graph model on the Lizard dataset using hovernet compute_metric tool.


#### clean_folder(save_folder: str)
Clean the folder with the results to save disk space.

Args:

    save_folder (str): folder to save the results.

Returns:

    None.

Raises:

    None.


#### get_weights_path(weights_selector: dict)
Looks for the weights path for each model available and return the one that matches the selection.

Args:

    weights_selector (dict): dictionary with the model parameters to select.

Returns:

    weights_path (dict): dictionary with the weights path to test.

Raises:

    None.


#### init_graph_inference(weights_path: str)
Initialize the graph model for inference.

Args:

    weights_path (str): path to the checkpoint.

Returns:

    None.

Raises:

    None.


#### init_result_table(weights_selector: dict)
Initialize the result table to save the results.

Args:

    weights_selector (dict): dictionary with the weights to use.

Returns:

    None.

Raises:

    None.


#### run_hovernet_metric_tool(save_folder: str)
Run the hovernet metric tool to compute the metrics.

Args:

    save_folder (str): folder to save the results.

Returns:

    result (str): string with the results metrics.

Raises:

    None.


#### save_mat(graph_model: Module, save_folder: str)
Run the inference on the test data and save the results in a .mat file.

Args:

    graph_model (torch.nn.Module): graph model to use for inference.
    save_folder (str): folder to save the results.

Returns:

    None.

Raises:

    None.


#### save_result_in_table(save_folder: str, result: str)
Save the result of a model in the result table.

Args:

    save_folder (str): folder to save the results.
    result (str): string with the results metrics.

Returns:

    None.

Raises:

    None.


#### save_result_to_file()
Save the result table to a file in a good format to use later.

Args:

    None.

Returns:

    None.

Raises:

    None.

## sflizard.pipeline.pipeline_utils module


### sflizard.pipeline.pipeline_utils.get_class_map_from_graph(graph: list, inst_maps: list, graph_pred: list, class_pred: list)
Get the class map from the graph prediction.

Args:

    graph (list): The graph.
    inst_map (list): The instance map.
    graph_pred (list): The graph prediction.
    class_pred (list): The class prediction.

Returns:

    class_maps (np.array): The class map.

Raises:

    None.


### sflizard.pipeline.pipeline_utils.improve_class_map(class_map: array, predicted_masks: array, points: array)
Improve the class map by assigning the same class to each segmented object.

Args:

    class_map (np.array): The class map.
    predicted_masks (np.array): The predicted masks.
    points (np.array): The points of the cells detected in the masks.

Returns:

    improved_class_map (np.array): The improved class map.

Raises:

    None.


### sflizard.pipeline.pipeline_utils.merge_stardist_class_together(p0: array, p1: array, p2: array, p3: array)
Merge the 4 stardist class prediction together.

Args:

    p0 (np.array): The first class prediction.
    p1 (np.array): The second class prediction.
    p2 (np.array): The third class prediction.
    p3 (np.array): The fourth class prediction.

Returns:

    class_map (np.array): The merged class prediction.

Raises:

    None.


### sflizard.pipeline.pipeline_utils.rotate_and_pred(stardist: Module, inputs: Tensor, angle: int)
Rotate the input image and predict the mask with stardist.

Args:

    stardist (torch.nn.Module): The stardist model.
    inputs (torch.Tensor): The input image.
    angle (int): The angle to rotate the image.

Returns:

    tuple: tuple containing:

        pred_mask_rotated (np.array): The predicted mask.
        c (torch.Tensor): The predicted classes.

Raises:

    None.

## sflizard.pipeline.report module


### _class_ sflizard.pipeline.report.ReportGenerator(output_dir: str, imgs_to_display: int, n_classes: int)
Bases: `object`

MD report generator.


#### add_batch(images: list, true_masks: list, pred_masks: list, true_class_map: list | None = None, pred_class_map: list | None = None, pred_class_map_improved: list | None = None, graphs: list | None = None, graphs_class_map: list | None = None)
Add a batch to the report.

Args:

    images (list): The images.
    true_masks (list): The true masks.
    pred_masks (list): The predicted masks.
    true_class_map (list): The true class map.
    pred_class_map (list): The predicted class map.
    pred_class_map_improved (list): The improved predicted class map.
    graphs (list): The graphs.
    graphs_class_map (list): The graphs class map.

Returns:

    None.

Raises:

    None.


#### add_final_metrics(segmentation_metric: Dict[Any, Any] | None, segmentation_classification_metric: Dict[Any, Any] | None, graph_segmentation_classification_metric: Dict[Any, Any] | None)
Add final metrics to the report.

Args:

    segmentation_metric (dict): The segmentation metric.
    classification_metric (dict): The classification metric.
    graph_classification_metric (dict): The graph classification metric.
    segmentation_classification_metric (dict): The segmentation classification metric.
    graph_segmentation_classification_metric (dict): The graph segmentation classification metric.

Returns:

    None.

Raises:

    None.


#### generate_md()
Generate a markdown file with the report.

Args:

    None.

Returns:

    None.

Raises:

    None.

## sflizard.pipeline.segmentation_metric_tool module


### _class_ sflizard.pipeline.segmentation_metric_tool.SegmentationMetricTool(n_classes: int, device: str)
Bases: `object`

A tool to compute metrics for segmentation.


#### add_batch(batch_idx: int, true_masks: array, pred_masks: array)
Add a batch to the metric tool.

Args:

    batch_idx (int): The batch index.
    true_masks (np.array): The true masks.
    pred_masks (np.array): The predicted masks.

Returns:

    None.

Raises:

    None.


#### add_batch_class(batch_idx: int, true_class_map: array, pred_class_map: array)
Add a batch to the metric tool.

Args:

    batch_idx (int): The batch index.
    true_class_map (np.array): The true class map.
    pred_class_map (np.array): The predicted class map.

Returns:

    None.

Raises:

    None.


#### compute_metrics()
Compute the metrics.

Args:

    None.

Returns:

    None.

Raises:

    None.


#### log_results(console: Console)
Log the results in rich tables.

Args:

    console (Console): The rich console.

Returns:

    None.

Raises:

    None.

## sflizard.pipeline.test_pipeline module


### _class_ sflizard.pipeline.test_pipeline.TestPipeline(valid_data_path: str, test_data_path: str, stardist_weights_path: str, graph_weights_path: List[str], graph_distance: int, n_rays: int, n_classes: int, batch_size: int, seed: int, mode: str)
Bases: `object`

A pipeline to test the model.


#### test(output_dir=None, imgs_to_display=0)
Run the pipeline and test the model.

Args:

    output_dir (str): The path to the output directory for report and images.
    imgs_to_display (int): The number of images to display in the report.

Returns:

    None.

Raises:

    None.
