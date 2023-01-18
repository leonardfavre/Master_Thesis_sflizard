import torch
import numpy as np

def rotate_and_pred(stardist, inputs, angle) -> tuple[np.ndarray, torch.Tensor]:
    """Rotate the input image and predict the mask with stardist.
    
    Args:
        stardist (StarDist2D): The stardist model.
        inputs (torch.Tensor): The input image.
        angle (int): The angle to rotate the image.
        
    Returns:
        tuple[np.ndarray, torch.Tensor]: The predicted mask and the predicted classes.
            
    Raises:
        None.
    """
    inputs_rotated = torch.rot90(inputs, angle, [2, 3])
    with torch.no_grad():
        dist, prob, c = stardist(inputs_rotated)
    pred_mask_rotated = stardist.compute_star_label(
        inputs_rotated, dist, prob
    )
    # return rotated np array
    return np.rot90(pred_mask_rotated, -angle, [1, 2]), torch.rot90(
        c.cpu(), -angle, [2, 3]
    )

def merge_stardist_class_together(p0, p1, p2, p3):
    """Merge the 4 stardist class prediction together.

    Args:
        p0 (torch.Tensor): The first class prediction.
        p1 (torch.Tensor): The second class prediction.
        p2 (torch.Tensor): The third class prediction.
        p3 (torch.Tensor): The fourth class prediction.

    Returns:
        torch.Tensor: The merged class prediction.

    Raises:
        None.
    """
    class_map = torch.stack((p0, p1, p2, p3), dim=3)
    class_map = class_map.numpy().astype(int)
    class_map = np.apply_along_axis(
        lambda x: np.argmax(np.bincount(x)), 3, class_map
    )
    class_map = torch.from_numpy(class_map)

    return class_map

def improve_class_map(class_map, predicted_masks):
    """Improve the class map by assigning the same class to each segmented object.

    Args:
        class_map (np.ndarray): The class map.
        predicted_masks (np.ndarray): The predicted masks.

    Returns:
        np.ndarray: The improved class map.

    Raises:
        None.
    """
    # add one dim if only 2 dims
    improved_class_map = np.zeros_like(class_map)
    for i in range(1, np.unique(predicted_masks).shape[0]):
        present_class = np.unique(class_map[predicted_masks == i])
        best_class = 0
        possible_class = [x for x in present_class if x != 0]
        if len(possible_class) > 0:
            best_class = max(set(possible_class), key=possible_class.count)
        
        # for j in range(len(present_class)):
        #     best_class = present_class[j]
        #     if best_class != 0:
        #         break
        improved_class_map[predicted_masks == i] = best_class
    return improved_class_map

def get_class_map_from_graph(graph, predicted_masks, graph_pred, class_pred):
    """Get the class map from the graph prediction.

    Args:
        graph (list): The graph.
        predicted_masks (list): The predicted masks.
        graph_pred (list): The graph prediction.
        class_pred (list): The class prediction.

    Returns:
        list: The class map.

    Raises:
        None.
    """
    class_maps = []
    for idx, pm in enumerate(predicted_masks):
        if graph_pred[idx] is None:
            class_maps.append(class_pred[idx].int().cpu().numpy())
        else:
            class_map = np.zeros_like(pm)
            graph_points = graph[idx]["pos"].int().cpu().numpy()
            for i in range(1, np.unique(pm).shape[0]):
                # get all points in mask with value i
                points = np.argwhere(pm == i)
                # get the point in the graph included in the points
                for idp, p in enumerate(graph_points):
                    if (p == points).all(axis=1).any():
                        class_map[pm == i] = (
                            graph_pred[idx][idp].int().cpu().numpy()
                        )
                        break
            class_maps.append(class_map)
    return class_maps