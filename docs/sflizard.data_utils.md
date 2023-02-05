# sflizard.data_utils package

## Modules

This package contains the following modules:

* [sflizard.data_utils.classes_utils](sflizard.data_utils.md#sflizarddata_utilsclasses_utils)

* [sflizard.data_utils.data_extraction](sflizard.data_utils.md#sflizarddata_utilsdata_extraction)

* [sflizard.data_utils.data_module](sflizard.data_utils.md#sflizarddata_utilsdata_module)

* [sflizard.data_utils.data_module_utils](sflizard.data_utils.md#sflizarddata_utilsdata_module_utils)

* [sflizard.data_utils.graph_module](sflizard.data_utils.md#sflizarddata_utilsgraph_module)

## sflizard.data_utils.classes_utils


### sflizard.data_utils.classes_utils.get_class_color
Get the color to use for plotting for the background and the cells from the yaml file.

Args:

    None.

Returns:

    - class_color (list): List of the color used for each cells and background.

Raises:

    None.


### sflizard.data_utils.classes_utils.get_class_name
Get the class name of the cells from the yaml file.

Args:

    None.

Returns:

    - class_name (dict): Dict of the class name of the cells.

Raises:

    None.

## sflizard.data_utils.data_extraction

This module is a tool to extract data from the Lizard and CoNSeP dataset and save the extracted data in specific format for training and testing.

Only pkl extraction is mandatory for final test pipeline and training.
Files extraction is needed for using the hovernet_metric tool.

This tool can be called directly from command line.

The available arguments are the following:

    -ip / --images_path: Path to the folder containing images (multiple).
    -ie / --images_extension: Extension of the images. Default to "png".
    -mf / --matlab_file: Path to the .mat file containing the annotations (multiple).
    -of / --output_base_name: Name for the output file. Default to "data_final_split".
    -dsr / --train_test_split: Ratio of the dataset to be used for training. Default to 0.9.
    -s / --seed: Seed used for the random. Default to 303.
    -psi / --patch_size: Size of the window to extract patches from the images. Default to 540.
    -spt / --patch_step: Step size to extract patches from the images. Default to 200.

Additionaly, the following constant are defined:

    FILE_SAVE: If set to True, will save the separate files. Default to True.
    PKL_SAVE:  If set to True, will save the data in a pkl file. Default to True.
    DATASET: Name of the dataset. Default to "Lizard".


To run extraction of Lizard dataset : 

```
python sflizard/data_utils/data_extraction.py --images_path data/Lizard_dataset/Lizard_Images1/ data/Lizard_dataset/Lizard_Images2/ -mf data/Lizard_dataset/Lizard_Labels/Labels/
```


### sflizard.data_utils.data_extraction.extract_annotation_patches
Extract patches from annotations.
Loads the mat file, extract patches from instance map, select classes, nuclei_id, centroid to correspond to the extracted patches, compute the class map and save the result in a pd.Dataframe.

Args:

    - annotation_file (str): path to the annotation file.
    - annotations (pd.Dataframe): dataframe of annotations.
    - patch_size (int): size of the patches.
    - patch_step (int): step between patches.

Returns:

    - annotations (pd.Dataframe): dataframe of annotations with the new patches.

Raises:

    None.


### sflizard.data_utils.data_extraction.extract_data
Extract data from the original dataset folder.
First extract all images, then extract all annotations, then clean data and finaly save data in PKL, FILE or both.

Args:

    - args (argparse.Namespace): arguments from the command line (see list bellow).

Returns:

    None.

Raises:

    None.


### sflizard.data_utils.data_extraction.extract_image_patches
Extract patches from an image.
Reads the image, extract patches from image and save the result in a dict.

Args:

    - image_file (str): path to the image file.
    - patch_size (int): size of the patches.
    - patch_step (int): step between patches.

Returns:

    - images (dict): dictionary of images with the new patches.

Raises:

    None.


### sflizard.data_utils.data_extraction.extract_patches
Extract patches from an image or an instance map. 
Iterate over the image in each 2 first dimension to extract all possible patches. Additionaly, extract the patches from the border by adapting the step to include all picture.

Args:

    - array (np.array): array to be patched.
    - array_name (str): name of the array.
    - patch_size (int): size of the patches.
    - patch_step (int): step between patches.

Returns:

    - array_dict (dict): dictionary of array patches.

Raises:

    None.


### sflizard.data_utils.data_extraction.remove_missing_data
Clean data extracted data.
Removes image without annotations and annotations without images.
Also removes images and annotations with no cells (instance map is all 0).

Args:

    - images (dict): dictionary of images.
    - annotations (pd.DataFrame): dataframe of annotations.
    - set_name (str): name of the set.

Returns:

    - tuple: tuple containing:
        * images (dict): dictionary of images.
        * annotations (pd.DataFrame): dataframe of annotations.

Raises:

    None.

## sflizard.data_utils.data_module


### _class_ sflizard.data_utils.data_module.LizardDataModule
Bases: `LightningDataModule`

DataModule that returns the correct dataloaders for the Lizard dataset.

Args:

    - train_data_path (str): path to the train data.
    - valid_data_path (str): path to the valid data.
    - test_data_path (str): path to the test data.
    - annotation_target (str): annotation target. Default to "stardist_class".
    - batch_size (int): batch size. Default to 4.
    - num_workers (int): number of workers. Default to 4.
    - input_size (int): input size. Default to 540.
    - seed (int): seed. Default to 303.
    - aditional_args (Optional[dict]): aditional arguments. Used for nrays. Default to None.

Raises: None.


#### setup
Data setup for training, define transformations and datasets.
Defines a dataset for each path: train, valid and test.
Define a base transformation composition for each dataset (resize, normalize and toTensor). 
Defines a augmentation transformation composition for train set (horizontalFlip, verticalFlip, RandomRotate90, RandomSizedCrop).

Args:

    - stage (Optional[str]): stage.

Returns:

    None.

Raises:

    None.


#### test_dataloader
Return the test dataloader.

Args:

    None.

Returns:

    - dataloader (DataLoader): the test dataloader.

Raises:

    None.


#### train_dataloader
Return the training dataloader.

Args:

    None.

Returns:

    - dataloader (DataLoader): the training dataloader.

Raises:

    None.


#### val_dataloader
Return the validation dataloader.

Args:

    None.

Returns:

    - dataloader (DataLoader): the validation dataloader.

Raises:

    None.


### _class_ sflizard.data_utils.data_module.LizardDataset
Bases: `Dataset`

Dataset object for the Lizard.
Used to get items from the dataset.

In the case of a classic stardist annotation target, items will be:
    - image
    - obj_probabilities map
    - distances map

In the case of a stardist annotation target with classes, items will be:
    - image
    - obj_probabilities map
    - distances map
    - classes map

They are compute with the `sflizard.data_utils.data_module_utils.get_stardist_data` function and are transform if needed.


Args:
    
    - df (pd.DataFrame): dataframe containing the data.
    - data (np.ndarray): array containing the images.
    - tf_base (A.Compose): base transformation.
    - tf_augment (A.Compose): augmentation transformation.
    - annotation_target (str): annotation target.
    - aditional_args (Optional[dict]): aditional arguments. Used for nrays. Default to None.

Raises:

    None.

## sflizard.data_utils.data_module_utils


### sflizard.data_utils.data_module_utils.compute_stardist
Compute the star valid label of image according dist and prob.

Args:

    - dist (torch.Tensor): distance map.
    - prob (torch.Tensor): probability map.

Returns:

    - tuple: tuple containing:
        * points (np.ndarray): detected cells centroid.
        * probs (np.ndarray): probability of each pixel to be a cell.
        * dists (np.ndarray): distances corresponding to the cells shape.

Raises:

    None.


### sflizard.data_utils.data_module_utils.get_edge_list
Get the edge list from the vertex for each vertex closer than distance. The distance computed is the Euclidian distance.

Args:

    - vertex (np.array): vertex.
    - distance (int): distance.

Returns:

    - edge_list (list): edge list.

Raises:

    None.


### sflizard.data_utils.data_module_utils.get_graph
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

    - inst_map (np.ndarray): instance map. Default to None.
    - points (np.ndarray): list of detected cells centroid. Default to None.
    - predicted_classes (np.ndarray): list of predicted classes corresponding to the cells in points array. Default to None.
    - true_class_map (np.ndarray): true class map. Default to None.
    - n_rays (int): number of rays of stardist objects. Default to None.
    - distance (int): distance between two vertex to have an edge. Default to 45.
    - stardist_checkpoint (str): path to stardist checkpoint.
    - image (np.ndarray): image. Default to None.
    - x_type (str): type of x : ll or ll+c or ll+x or ll+c+x or 4ll or 4ll+c. Default to "4ll".
    - consep_data (bool): if True, the data is from consep datset. Default to False.

Returns:

    - graph (dict): Computed graph.

Raises:

    ValueError: if input is insuficient to compute graph.


### sflizard.data_utils.data_module_utils.get_graph_for_inference
Get the graph for inference. 
Uses the `sflizard.data_utils.data_module_utils.get_graph` to compute graph for a full batch.

Args:

    - batch (torch.Tensor): batch containing images.
    - distance (int): distance between two vertex to create an edge.
    - stardist_checkpoint (str): path to stardist checkpoint.
    - x_type (str): type of node feature vector. Default to "4ll".

Returns:

    - graphs (list): list of computed graphs.

Raises:

    None.


### sflizard.data_utils.data_module_utils.get_stardist_data
Get the data for stardist.
Uses the instance map to get distances and obj_probability. The class map is then added if given in argument to get stardist data for classes as well.

Args:

    - inst_map (np.array): instance map.
    - aditional_args (dict): additional arguments, must contain n_rays.
    - class_map (np.array): class map. Default to None.

Returns:

    - tuple: tuple containing:
        * dist (torch.Tensor): distance map.
        * prob (torch.Tensor): probability map.
        * classes (torch.Tensor): list of classes, if arg class_map is not None.

Raises:

    ValueError: if n_rays is not in aditional_args.


### sflizard.data_utils.data_module_utils.get_stardist_distances
Get the distances (rays) of stardist from an instance map.
The distances are retrieved using the `star_dist` function from Stardist library.

Args:

    - inst_map (np.array): annotation dictionary.
    - n_rays (int): number of rays.

Returns:

    - distances (torch.Tensor): distance map.


### sflizard.data_utils.data_module_utils.get_stardist_obj_probabilities
Get the object probabilities of stardist from an instance map.
The object probabilities are retrieved using the `edt_prob` function from Stardist library.

Args:

    - inst_map (np.array): instance map.

Returns:

    - obj_probabilities (torch.Tensor): object probabilities.
 

### sflizard.data_utils.data_module_utils.get_stardist_point_for_graph
Get the node feature vector from Stardist for graph creation.
Rotate the image if needed, compute the result of the Stardist model(s) on the rotated image, rotate the result in the inverse sens, create the node feature vector from the resulting data.
Node features that can be extracted are "x", "c" and "ll". Possible node features are all combinations of those 3 base features.

Args:

    - image (np.array): image
    - model_ll (torch.nn.Module): model for last layer
    - model_c (torch.nn.Module): model for complete network
    - points (np.array): points
    - x_type (str): type of node feature vector. Defaults to “ll”.
    - rotate (int): rotation of image. Defaults to 0.

Returns:

    - torch.Tensor: node feature vector.

Raises:

    None.

## sflizard.data_utils.graph_module


### sflizard.data_utils.graph_module.LizardGraphDataModule
Data module to create dataloaders for graph training job.

Two mode possible:
* images and annotations dataframe contained in a dict.
* directly the annotation dataframe.

In the case of direct annotation dataframe, the images are not saved.

Args:

    - train_data (dict or pd.DataFrame): train data.
    - valid_data (dict or pd.DataFrame): valid data.
    - test_data (dict or pd.DataFrame): test data.
    - batch_size (int): batch size.
    - num_workers (int): number of workers.
    - seed (int): seed for random number generator.
    - stardist_checkpoint (str): path to stardist checkpoint.
    - x_type (str): type of node features.
    - distance (int): distance for graph creation.
    - root (str): root path for saving processed data.
    - consep_data (bool): if True, use consep data.
    - light (bool): if True, only save basic graph information.

Returns:

    - datamodule (LightningDataset): Datamodule containing the required datasets.

Raises:

    ValueError: if no data is provided.


### _class_ sflizard.data_utils.graph_module.LizardGraphDataset
Bases: `Dataset`

Dataset object for the Graphs.

Args:

    - transform (None): transform. Default to None.
    - pre_transform (None): pre_transform. Default to None.
    - df (pd.DataFrame): dataframe containing the data. Default to pd.DataFrame().
    - data (np.ndarray): array containing the images. Default to np.array([]).
    - name (str): name of the dataset. Default to "".
    - n_rays (int): number of rays of stardist shape. Default to 32.
    - distance (int): distance between 2 connected cells. Default to 45.
    - stardist_checkpoint (str): path to the stardist checkpoint. Default to None.
    - x_type (str): type of the node feature vetor. Default to "4ll".
    - root (str): root path. Default to "data/graph".
    - consep_data (bool): if the data is from consep. Default to False.
    - light (bool): if the data included in the graph needs to be minimum, speed up training. Default to False.

Raises:

    None.


#### download
Downloads the dataset to the `self.raw_dir` folder.


#### get
Return the data at index idx.

Args:

    - idx (int): index of the data to return.

Returns:

    - data (Data): data at index idx.

Raises:

    None.


#### len
Return the length of the dataset.

Args:

    None.

Returns:

    - int: length of the dataset.

Raises:

    None.


#### process
Process the dataset.

Compute the graph from input data.
If the dataset is light, only the graph basic information is saved:

    x: node features.
    edge_index: edges.
    y: labels.
    image_idx: image index.

If the dataset is not light, the graph full information is saved:
    
    x: node features.
    edge_index: edges.
    y: labels.
    image_idx: image index.
    original_img: original image.
    inst_map: instance map.
    class_map: class map.

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