<!-- Master_Thesis_LeonardFavre documentation master file, created by
sphinx-quickstart on Sat Jan 28 01:14:28 2023.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive. -->
# Welcome to SF Lizards documentation!

This documentation describes and explains the different packages and modules developed during the SF Lizard project.

This documentation describes and explains the different packages and modules developed during the SF Lizard project.

The sflizard package contains 3 modules: 

* **run_hovernet_metric_tool** is a test run utility using the `compute_stats` metric function of the hover_net package.
* **run_test_pipeline** is a utility to run the image segmentation and classification pipeline with Stardist and the Graph Model. 
* **training** is a utility to launch training of Stardist model or neural graph network

The sflizard package also contains 4 sub packages:

* **data_utils**: set of tools concerning the preparation and the management of the data necessary to the various tasks.
* **stardist_model**: definition of the stardist model and utilities related to the model.
* **graph_model**: definition of graph neural networks models and utilities related to the models.
* **pipeline**: definition of the test pipeline and related utilities.

## Links to specific documentation :

* [sflizard package](sflizard.md)

    * [sflizard.run_hovernet_metric_tool](sflizard.md#module-sflizard.run_hovernet_metric_tool)


    * [sflizard.run_test_pipeline](sflizard.md#module-sflizard.run_test_pipeline)


    * [sflizard.training](sflizard.md#module-sflizard.training)

    
    * [sflizard.data_utils package](sflizard.data_utils.md)


        * [sflizard.data_utils.classes_utils](sflizard.data_utils.md#module-sflizard.data_utils.classes_utils)


        * [sflizard.data_utils.data_extraction](sflizard.data_utils.md#module-sflizard.data_utils.data_extraction)


        * [sflizard.data_utils.data_module](sflizard.data_utils.md#module-sflizard.data_utils.data_module)


        * [sflizard.data_utils.data_module_utils](sflizard.data_utils.md#module-sflizard.data_utils.data_module_utils)


        * [sflizard.data_utils.graph_module](sflizard.data_utils.md#module-sflizard.data_utils.graph_module)


        * [sflizard.data_utils.remove_dead_data](sflizard.data_utils.md#module-sflizard.data_utils.remove_dead_data)


    * [sflizard.stardist_model package](sflizard.stardist_model.md)


        * [sflizard.stardist_model.models_utils](sflizard.stardist_model.md#module-sflizard.stardist_model.models_utils)


        * [sflizard.stardist_model.stardist_model](sflizard.stardist_model.md#module-sflizard.stardist_model.stardist_model)

    
    * [sflizard.Graph_model package](sflizard.Graph_model.md)


        * [sflizard.Graph_model.graph_model](sflizard.Graph_model.md#module-sflizard.Graph_model.graph_model)


    * [sflizard.pipeline package](sflizard.pipeline.md)


        * [sflizard.pipeline.hovernet_metric_tool](sflizard.pipeline.md#module-sflizard.pipeline.hovernet_metric_tool)


        * [sflizard.pipeline.pipeline_utils](sflizard.pipeline.md#module-sflizard.pipeline.pipeline_utils)


        * [sflizard.pipeline.report](sflizard.pipeline.md#module-sflizard.pipeline.report)


        * [sflizard.pipeline.segmentation_metric_tool](sflizard.pipeline.md#module-sflizard.pipeline.segmentation_metric_tool)


        * [sflizard.pipeline.test_pipeline](sflizard.pipeline.md#module-sflizard.pipeline.test_pipeline)