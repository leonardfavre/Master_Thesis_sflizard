import numpy as np
import torch


def rotate_and_pred(
    stardist: torch.nn.Module,
    inputs: torch.Tensor,
    angle: int,
) -> tuple[np.ndarray, torch.Tensor]:
    """Rotate the input image and predict the mask with stardist.

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
    """
    inputs_rotated = torch.rot90(inputs, angle, [2, 3])
    with torch.no_grad():
        dist, prob, c = stardist(inputs_rotated)
    pred_mask_rotated = stardist.compute_star_label(inputs_rotated, dist, prob)
    # return rotated np array
    return np.rot90(pred_mask_rotated, -angle, [1, 2]), torch.rot90(
        c.cpu(), -angle, [2, 3]
    )


def merge_stardist_class_together(
    p0: np.array,
    p1: np.array,
    p2: np.array,
    p3: np.array,
) -> np.array:
    """Merge the 4 stardist class prediction together.

    Args:
        p0 (np.array): The first class prediction.
        p1 (np.array): The second class prediction.
        p2 (np.array): The third class prediction.
        p3 (np.array): The fourth class prediction.

    Returns:
        class_map (np.array): The merged class prediction.

    Raises:
        None.
    """
    class_map = np.stack((p0, p1, p2, p3), dim=3)
    class_map = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), 3, class_map)
    # class_map = torch.from_numpy(class_map)

    return class_map


def improve_class_map(
    class_map: np.array, predicted_masks: np.array, points: np.array
) -> np.array:
    """Improve the class map by assigning the same class to each segmented object.

    Args:
        class_map (np.array): The class map.
        predicted_masks (np.array): The predicted masks.
        points (np.array): The points of the cells detected in the masks.

    Returns:
        improved_class_map (np.array): The improved class map.

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
) -> np.array:
    """Get the class map from the graph prediction.

    Args:
        graph (list): The graph.
        inst_map (list): The instance map.
        graph_pred (list): The graph prediction.
        class_pred (list): The class prediction.

    Returns:
        class_maps (np.array): The class map.

    Raises:
        None.
    """
    class_maps = []
    for idx, inst_map in enumerate(inst_maps):
        if graph_pred[idx] is None:
            class_maps.append(class_pred[idx])
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
            class_maps.append(class_map)
    class_maps = np.array(class_maps).astype("int32")
    return class_maps
