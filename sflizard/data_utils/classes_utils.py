from pathlib import Path
import yaml  # type: ignore
import numpy as np

def get_class_name():
    """ Get the class name of the cells from the yaml file.

    Args:
        None.

    Returns:
        class_name (dict): Dict of the class name of the cells.

    Raises:
        None.
    """
    # load the class name from file
    yaml_file = Path(__file__).parents[0] / "classes.yaml"
    with open(yaml_file, "r") as stream:
        classes_file: dict = yaml.full_load(stream)
        class_name = classes_file["classes_name"]
    return class_name

def get_class_color():
    """ Get the color to use for plotting for the background and the cells from the yaml file.

    Args:
        None.

    Returns:
        class_color (list): List of the color used for each cells and background.

    Raises:
        None.
    """
    # load the class color from file
    yaml_file = Path(__file__).parents[0] / "classes.yaml"
    with open(yaml_file, "r") as stream:
        classes_file: dict = yaml.full_load(stream)
        class_color = np.array(classes_file["classes_color"]) / 255
    return class_color