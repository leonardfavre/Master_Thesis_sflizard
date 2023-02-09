import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from sflizard.data_utils import get_class_name


def improve_class_map(
    class_map: np.ndarray, predicted_masks: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Improve the class map by assigning the same class to each segmented object.

    Args:
        class_map (np.ndarray): The class map.
        predicted_masks (np.ndarray): The predicted masks.
        points (np.ndarray): The points of the cells detected in the masks.

    Returns:
        improved_class_map (np.ndarray): The improved class map.

    Raises:
        None.
    """
    # add one dim if only 2 dims
    improved_class_map = np.zeros_like(class_map)
    for p in points:
        p_cell = predicted_masks[p[0]][p[1]]
        if class_map[p[0]][p[1]] != 0:
            improved_class_map[predicted_masks == p_cell] = class_map[p[0]][p[1]]
        else:
            improved_class_map[predicted_masks == p_cell] = np.argmax(
                np.bincount(class_map[predicted_masks == p_cell])
            )
    return improved_class_map


def get_class_map_from_graph(
    graph: list,
    inst_maps: list,
    graph_pred: list,
    class_pred: list,
) -> np.ndarray:
    """Get the class map from the graph prediction.

    Args:
        graph (list): The graph.
        inst_map (list): The instance map.
        graph_pred (list): The graph prediction.
        class_pred (list): The class prediction.

    Returns:
        class_maps (np.ndarray): The class map.

    Raises:
        None.
    """
    class_maps_list = []
    for idx, inst_map in enumerate(inst_maps):
        if graph_pred[idx] is None:
            class_maps_list.append(class_pred[idx])
            print("problem with graph prediction")
        else:
            class_map = np.zeros_like(inst_map)
            graph_points = graph[idx]["pos"].int().cpu().numpy()
            for idp, p in enumerate(graph_points):
                id_cell = inst_map[p[0], p[1]]
                if id_cell != 0:
                    if idp < len(graph_pred[idx]):
                        class_map[inst_map == id_cell] = (
                            graph_pred[idx][idp].cpu().numpy()
                        )
                else:
                    print("problem between graph and stardist")
            class_maps_list.append(class_map)
    class_maps = np.array(class_maps_list).astype("int32")
    return class_maps


def log_confmat(confmat: torch.Tensor, title: str, console: Console) -> None:
    """Log the confusion matrix.

    Args:
        confmat (torch.Tensor): The confusion matrix.
        title (str): The title of the confusion matrix.
        console (Console): The console to log the confusion matrix.

    Returns:
        None.

    Raises:
        None.
    """
    cm = confmat.cpu().numpy()
    table = Table(
        show_header=True,
        header_style="bold magenta",
        title=title,
    )
    table.add_column(" ", style="bold magenta", justify="center")
    for i in range(cm.shape[0]):
        table.add_column(get_class_name()[i], justify="center")
    for i in range(cm.shape[1]):
        str_cm_line = [str(c) for c in cm[i]]
        table.add_row(get_class_name()[i], *str_cm_line)
    console.print(table)
