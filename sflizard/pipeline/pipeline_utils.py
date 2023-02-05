import numpy as np


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
