# sflizard.data_utils package

    * [Submodules](sflizard.data_utils.md#submodules)


    * [sflizard.data_utils.classes_utils module](sflizard.data_utils.md#module-sflizard.data_utils.classes_utils)


        * [`get_class_color()`](sflizard.data_utils.md#sflizard.data_utils.classes_utils.get_class_color)


        * [`get_class_name()`](sflizard.data_utils.md#sflizard.data_utils.classes_utils.get_class_name)


    * [sflizard.data_utils.data_extraction module](sflizard.data_utils.md#module-sflizard.data_utils.data_extraction)


        * [`extract_annotation_patches()`](sflizard.data_utils.md#sflizard.data_utils.data_extraction.extract_annotation_patches)


        * [`extract_data()`](sflizard.data_utils.md#sflizard.data_utils.data_extraction.extract_data)


        * [`extract_image_patches()`](sflizard.data_utils.md#sflizard.data_utils.data_extraction.extract_image_patches)


        * [`extract_patches()`](sflizard.data_utils.md#sflizard.data_utils.data_extraction.extract_patches)


        * [`remove_missing_data()`](sflizard.data_utils.md#sflizard.data_utils.data_extraction.remove_missing_data)


    * [sflizard.data_utils.data_module module](sflizard.data_utils.md#module-sflizard.data_utils.data_module)


        * [`LizardDataModule`](sflizard.data_utils.md#sflizard.data_utils.data_module.LizardDataModule)


            * [`LizardDataModule.setup()`](sflizard.data_utils.md#sflizard.data_utils.data_module.LizardDataModule.setup)


            * [`LizardDataModule.test_dataloader()`](sflizard.data_utils.md#sflizard.data_utils.data_module.LizardDataModule.test_dataloader)


            * [`LizardDataModule.train_dataloader()`](sflizard.data_utils.md#sflizard.data_utils.data_module.LizardDataModule.train_dataloader)


            * [`LizardDataModule.val_dataloader()`](sflizard.data_utils.md#sflizard.data_utils.data_module.LizardDataModule.val_dataloader)


        * [`LizardDataset`](sflizard.data_utils.md#sflizard.data_utils.data_module.LizardDataset)


    * [sflizard.data_utils.data_module_utils module](sflizard.data_utils.md#module-sflizard.data_utils.data_module_utils)


        * [`compute_stardist()`](sflizard.data_utils.md#sflizard.data_utils.data_module_utils.compute_stardist)


        * [`get_edge_list()`](sflizard.data_utils.md#sflizard.data_utils.data_module_utils.get_edge_list)


        * [`get_graph()`](sflizard.data_utils.md#sflizard.data_utils.data_module_utils.get_graph)


        * [`get_graph_for_inference()`](sflizard.data_utils.md#sflizard.data_utils.data_module_utils.get_graph_for_inference)


        * [`get_stardist_data()`](sflizard.data_utils.md#sflizard.data_utils.data_module_utils.get_stardist_data)


        * [`get_stardist_distances()`](sflizard.data_utils.md#sflizard.data_utils.data_module_utils.get_stardist_distances)


        * [`get_stardist_obj_probabilities()`](sflizard.data_utils.md#sflizard.data_utils.data_module_utils.get_stardist_obj_probabilities)


        * [`get_stardist_point_for_graph()`](sflizard.data_utils.md#sflizard.data_utils.data_module_utils.get_stardist_point_for_graph)


    * [sflizard.data_utils.graph_module module](sflizard.data_utils.md#module-sflizard.data_utils.graph_module)


        * [`LizardGraphDataModule()`](sflizard.data_utils.md#sflizard.data_utils.graph_module.LizardGraphDataModule)


        * [`LizardGraphDataset`](sflizard.data_utils.md#sflizard.data_utils.graph_module.LizardGraphDataset)


            * [`LizardGraphDataset.download()`](sflizard.data_utils.md#sflizard.data_utils.graph_module.LizardGraphDataset.download)


            * [`LizardGraphDataset.get()`](sflizard.data_utils.md#sflizard.data_utils.graph_module.LizardGraphDataset.get)


            * [`LizardGraphDataset.len()`](sflizard.data_utils.md#sflizard.data_utils.graph_module.LizardGraphDataset.len)


            * [`LizardGraphDataset.process()`](sflizard.data_utils.md#sflizard.data_utils.graph_module.LizardGraphDataset.process)


            * [`LizardGraphDataset.processed_file_names`](sflizard.data_utils.md#sflizard.data_utils.graph_module.LizardGraphDataset.processed_file_names)


            * [`LizardGraphDataset.raw_file_names`](sflizard.data_utils.md#sflizard.data_utils.graph_module.LizardGraphDataset.raw_file_names)


    * [sflizard.data_utils.remove_dead_data module](sflizard.data_utils.md#module-sflizard.data_utils.remove_dead_data)


        * [`remove_dead_data()`](sflizard.data_utils.md#sflizard.data_utils.remove_dead_data.remove_dead_data)


    * [Module contents](sflizard.data_utils.md#module-sflizard.data_utils)

## Submodules

## sflizard.data_utils.classes_utils module


### sflizard.data_utils.classes_utils.get_class_color()
Get the color to use for plotting for the background and the cells from the yaml file.

Args:

    None.

Returns:

    class_color (list): List of the color used for each cells and background.

Raises:

    None.


### sflizard.data_utils.classes_utils.get_class_name()
Get the class name of the cells from the yaml file.

Args:

    None.

Returns:

    class_name (dict): Dict of the class name of the cells.

Raises:

    None.

## sflizard.data_utils.data_extraction module


### sflizard.data_utils.data_extraction.extract_annotation_patches(annotation_file: str, annotations: DataFrame, patch_size: int, patch_step: int)
Extract patches from annotations.

Args:

    annotation_file (str): path to the annotation file.
    annotations (pd.Dataframe): dataframe of annotations.
    patch_size (int): size of the patches.
    patch_step (int): step between patches.

Returns:

    annotations (pd.Dataframe): dataframe of annotations with the new patches.

Raises:

    None.


### sflizard.data_utils.data_extraction.extract_data(args: Namespace)
Extract data from the original dataset folder.

Args:

    args (argparse.Namespace): arguments from the command line (see list bellow).

Returns:

    None.

Raises:

    None.


### sflizard.data_utils.data_extraction.extract_image_patches(image_file: str, patch_size: int, patch_step: int)
Extract patches from an image.

Args:

    image_file (str): path to the image file.
    patch_size (int): size of the patches.
    patch_step (int): step between patches.

Returns:

    images (dict): dictionary of images with the new patches.

Raises:

    None.


### sflizard.data_utils.data_extraction.extract_patches(array: array, array_name: str, patch_size: int, patch_step: int)
Extract patches from an image or an instance map.

Args:

    array (np.array): array to be patched.
    array_name (str): name of the array.
    patch_size (int): size of the patches.
    patch_step (int): step between patches.

Returns:

    array_dict (dict): dictionary of array patches.

Raises:

    None.


### sflizard.data_utils.data_extraction.remove_missing_data(images: dict, annotations: DataFrame, set_name: str)
Clean data by removing image without annotations and annotations without images.

Args:

    images (dict): dictionary of images.
    annotations (pd.DataFrame): dataframe of annotations.
    set_name (str): name of the set.

Returns:

    tuple: tuple containing:


        * images (dict): dictionary of images.


        * annotations (pd.DataFrame): dataframe of annotations.

Raises:

    None.

## sflizard.data_utils.data_module module


### _class_ sflizard.data_utils.data_module.LizardDataModule(train_data_path: str | None, valid_data_path: str, test_data_path: str, annotation_target: str = 'inst', batch_size: int = 4, num_workers: int = 4, input_size=540, seed: int = 303, aditional_args: dict | None = None)
Bases: `LightningDataModule`

DataModule that returns the correct dataloaders for the Lizard dataset.


#### setup(stage: str | None = None)
Data setup for training, define transformations and datasets.

Args:

    stage (Optional[str]): stage.

Returns:

    None.

Raises:

    None.


#### test_dataloader()
Return the test dataloader.

Args:

    None.

Returns:

    dataloader (DataLoader): the test dataloader.

Raises:

    None.


#### train_dataloader()
Return the training dataloader.

Args:

    None.

Returns:

    dataloader (DataLoader): the training dataloader.

Raises:

    None.


#### val_dataloader()
Return the validation dataloader.

Args:

    None.

Returns:

    dataloader (DataLoader): the validation dataloader.

Raises:

    None.


### _class_ sflizard.data_utils.data_module.LizardDataset(df: DataFrame, data: ndarray, tf_base: Compose, tf_augment: Compose, annotation_target: str, aditional_args: dict | None = None)
Bases: `Dataset`

Dataset object for the Lizard.

## sflizard.data_utils.data_module_utils module


### sflizard.data_utils.data_module_utils.compute_stardist(dist: Tensor, prob: Tensor)
Compute the star valid label of image according dist and prob.

Args:

    dist (torch.Tensor): distance map.
    prob (torch.Tensor): probability map.

Returns:

    tuple: tuple containing:

        points (np.ndarray): detected cells centroid.
        probs (np.ndarray): probability of each pixel to be a cell.
        dists (np.ndarray): distances corresponding to the cells shape.

Raises:

    None.


### sflizard.data_utils.data_module_utils.get_edge_list(vertex: array, distance: int)
Get the edge list from the vertex for each vertex closer than distance.

Args:

    vertex (np.array): vertex.
    distance (int): distance.

Returns:

    edge_list (list): edge list.

Raises:

    None.


### sflizard.data_utils.data_module_utils.get_graph(inst_map: array | None = None, points: array | None = None, predicted_classes: array | None = None, true_class_map: array | None = None, n_rays: int | None = None, distance: int = 45, stardist_checkpoint: str | None = None, image: array | None = None, x_type: str = 'll', consep_data: bool = False, hovernet_metric: bool = False)
Get the graph from the instance map.

The graph can be computed from a Stardist checkpoint, in this case the following data are required:

    image: used as input for Stardist.

If the hovernet_metric is True, the instance map computed by stardist is included in the returned dict.
The x_type available for this technique is “c”, “x”, “ll”, “c+x”, “ll+x”, “ll+c”, “ll+x+c”, “c”, “4x”, “4ll”, “4c+x”, “4ll+x”, “4ll+c”, “4ll+x+c”.

The graph can be computed from the points and predicted classes, in this case the following data are required:

    points: list of detected cells centroid.
    predicted_classes: list of predicted classes corresponding to the cells in points array.

The x_type available for this technique is “c”.
This technique is used for the hovernet model.

The graph can be computed from an instance map, in this case the following data are required:

    inst_map: instance map.
    n_rays: number of rays of stardist objects.

It will call the function compute_stardist to compute the points, probs and dists.
The x_type available for this technique is “dist”.
This technique is deprecated.

If true_class_map is not None, the graph includes the labels of the cells in the “y” entry.
If the consep_data is True, the labels will be corrected to apply the HoverNet simplification of the classes.

Args:

    inst_map (np.ndarray): instance map.
    points (np.ndarray): list of detected cells centroid.
    predicted_classes (np.ndarray): list of predicted classes corresponding to the cells in points array.
    true_class_map (np.ndarray): true class map.
    n_rays (int): number of rays of stardist objects.
    distance (int): distance between two vertex to have an edge.
    stardist_checkpoint (str): path to stardist checkpoint.
    image (np.ndarray): image.
    x_type (str): type of x : ll or ll+c or ll+x or ll+c+x or 4ll or 4ll+c.
    consep_data (bool): if True, the data is from consep datset.

Returns:

    graph (dict): Computed graph.

Raises:

    ValueError: if input is insuficient to compute graph.


### sflizard.data_utils.data_module_utils.get_graph_for_inference(batch: Tensor, distance: int, stardist_checkpoint: str, x_type: str = 'll')
Get the graph for inference.

Args:

    batch (torch.Tensor): batch containing images.
    distance (int): distance between two vertex to create an edge.
    stardist_checkpoint (str): path to stardist checkpoint.
    x_type (str): type of node feature vector.

Returns:

    graphs (list): list of computed graphs.

Raises:

    None.


### sflizard.data_utils.data_module_utils.get_stardist_data(inst_map: array, aditional_args: dict | None, class_map: array | None = None)
Get the data for stardist.

Args:

    inst_map (np.array): instance map.
    aditional_args (dict): additional arguments, must contain n_rays.
    class_map (np.array): class map.

Returns:

    tuple: tuple containing:

        dist (torch.Tensor): distance map.
        prob (torch.Tensor): probability map.
        classes (torch.Tensor): list of classes, if arg class_map is not None.

Raises:

    ValueError: if n_rays is not in aditional_args.


### sflizard.data_utils.data_module_utils.get_stardist_distances(inst_map: array, n_rays: int)
Get the distances (rays) of stardist from an instance map.

Args:

    inst_map (np.array): annotation dictionary.
    n_rays (int): number of rays.

Returns:

    distances (torch.Tensor): distance map.


### sflizard.data_utils.data_module_utils.get_stardist_obj_probabilities(inst_map: array)
Get the object probabilities of stardist from an instance map.

Args:

    inst_map (np.array): instance map.

Returns:

    obj_probabilities (torch.Tensor): object probabilities.


### sflizard.data_utils.data_module_utils.get_stardist_point_for_graph(image: array, model_ll: Module, model_c: Module, points: array, x_type: str = 'll', rotate: int = 0)
Get the node feature vector from stardist for graph creation.

Args:

    image (np.array): image
    model_ll (torch.nn.Module): model for last layer
    model_c (torch.nn.Module): model for complete network
    points (np.array): points
    x_type (str): type of node feature vector. Defaults to “ll”.
    rotate (int): rotation of image. Defaults to 0.

Returns:

    torch.Tensor: node feature vector.

Raises:

    None.

## sflizard.data_utils.graph_module module


### sflizard.data_utils.graph_module.LizardGraphDataModule(train_data: dict | DataFrame | None = None, valid_data: dict | DataFrame | None = None, test_data: dict | DataFrame | None = None, batch_size: int = 32, num_workers: int = 4, seed: int = 303, stardist_checkpoint=None, x_type='ll', distance=45, root='data/graph', consep_data=False, light=False)
Data module to create dataloaders for graph training job.

Two mode possible:

    
    * images and annotations dataframe contained in a dict.


    * directly the annotation dataframe.

In the case of direct annotation dataframe, the images are not saved.

Args:

    train_data (dict or pd.DataFrame): train data.
    valid_data (dict or pd.DataFrame): valid data.
    test_data (dict or pd.DataFrame): test data.
    batch_size (int): batch size.
    num_workers (int): number of workers.
    seed (int): seed for random number generator.
    stardist_checkpoint (str): path to stardist checkpoint.
    x_type (str): type of node features.
    distance (int): distance for graph creation.
    root (str): root path for saving processed data.
    consep_data (bool): if True, use consep data.
    light (bool): if True, only save basic graph information.

Returns:

    datamodule (LightningDataset): Datamodule containing the required datasets.

Raises:

    ValueError: if no data is provided.


### _class_ sflizard.data_utils.graph_module.LizardGraphDataset(transform=None, pre_transform=None, df: ~pandas.core.frame.DataFrame = Empty DataFrame Columns: [] Index: [], data: ~numpy.ndarray = array([], dtype=float64), name: str = '', n_rays: int = 32, distance: int = 45, stardist_checkpoint: str | None = None, x_type: str = 'll', root: str = 'data/graph', consep_data: bool = False, light: bool = False)
Bases: `Dataset`

Dataset object for the Graphs.


#### download()
Downloads the dataset to the `self.raw_dir` folder.


#### get(idx)
Return the data at index idx.

Args:

    idx (int): index of the data to return.

Returns:

    data (Data): data at index idx.

Raises:

    None.


#### len()
Return the length of the dataset.

Args:

    None.

Returns:

    int: length of the dataset.

Raises:

    None.


#### process()
Process the dataset.

Compute the graph from input data.
If the dataset is light, only the graph basic information is saved:

> 
> * x: node features.


> * edge_index: edges.


> * y: labels.


> * image_idx: image index.

If the dataset is not light, the graph full information is saved:

    
    * x: node features.


    * edge_index: edges.


    * y: labels.


    * image_idx: image index.


    * original_img: original image.


    * inst_map: instance map.


    * class_map: class map.

The graph is in both cases saved to speed up use of the dataset.

Args:

    None.

Returns:

    None.

Raises:

    None.


#### _property_ processed_file_names(_: lis_ )
The name of the files in the `self.processed_dir` folder that
must be present in order to skip processing.


#### _property_ raw_file_names(_: lis_ )
The name of the files in the `self.raw_dir` folder that must
be present in order to skip downloading.

## sflizard.data_utils.remove_dead_data module


### sflizard.data_utils.remove_dead_data.remove_dead_data()
Remove dead data from the dataset.
