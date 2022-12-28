from pathlib import Path
import yaml  # type: ignore
import numpy as np

def get_class_name():
    # load the class name from file
    yaml_file = Path(__file__).parents[0] / "classes.yaml"
    with open(yaml_file, "r") as stream:
        classes_file: dict = yaml.full_load(stream)
        class_name = classes_file["classes_name"]
    return class_name

def get_class_color():
    # load the class color from file
    yaml_file = Path(__file__).parents[0] / "classes.yaml"
    with open(yaml_file, "r") as stream:
        classes_file: dict = yaml.full_load(stream)
        class_color = np.array(classes_file["classes_color"]) / 255
    return class_color