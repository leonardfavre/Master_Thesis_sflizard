import math

import numpy as np
import torch
from stardist import edt_prob, non_maximum_suppression, star_dist

from sflizard import Stardist


def get_stardist_distances(inst_map, n_rays):
    """Get the distances for stardist.

    Args:
        inst_map (): annotation dictionary.
        n_rays (int): number of rays.

    Returns:
        distances (torch.Tensor): distance map.
    """
    distances = star_dist(inst_map, n_rays)
    distances = torch.from_numpy(np.transpose(distances, (2, 0, 1)))
    return distances


def get_stardist_obj_probabilities(inst_map):
    """Get the object probabilities for stardist.

    Args:
        inst_map (): instance map.

    Returns:
        obj_probabilities (torch.Tensor): object probabilities.
    """
    obj_probabilities = edt_prob(inst_map)
    obj_probabilities = torch.from_numpy(np.expand_dims(obj_probabilities, 0))
    return obj_probabilities


def get_stardist_data(inst_map, aditional_args, class_map=None):
    """Get the data for stardist.

    Args:
        inst_map (): instance map.
        aditional_args (dict): additional arguments, must contain n_rays.
        classes (list): list of classes.

    Returns:
        tuple: tuple containing:
            - dist (np.ndarray): distance map.
            - prob (np.ndarray): probability map.
            - classes (list): list of classes.

    Raises:
        ValueError: if n_rays is not in aditional_args.
    """
    if "n_rays" not in aditional_args.keys():
        raise ValueError("n_rays not in aditional_args. Mandatory for stardist model.")
    distances = get_stardist_distances(inst_map, aditional_args["n_rays"])
    obj_probabilities = get_stardist_obj_probabilities(inst_map)
    if class_map is not None:
        return obj_probabilities, distances, torch.from_numpy(class_map)
    else:
        return obj_probabilities, distances


# GRAPH UTILS


def compute_stardist(
    dist: torch.Tensor,
    prob: torch.Tensor,
):
    """Compute the stare label of image according dist and prob."""
    dist_numpy = dist.detach().cpu().numpy().squeeze()
    prob_numpy = prob.detach().cpu().numpy().squeeze()
    dist_numpy = np.transpose(dist_numpy, (1, 2, 0))
    points, probs, dists = non_maximum_suppression(
        dist_numpy, prob_numpy, nms_thresh=0.5, prob_thresh=0.5
    )
    return points, probs, dists


def get_edge_list(vertex, distance):
    # edge distanche
    def distance_between_vertex(v_i, v_j):
        distance = ((v_i[0] - v_j[0]) ** 2 + (v_i[1] - v_j[1]) ** 2) ** (0.5)
        return distance

    edge_list = [[], [], []]
    for i in range(vertex.shape[0]):
        for j in range(i + 1, vertex.shape[0]):
            dist = distance_between_vertex(vertex[i], vertex[j])
            if dist < distance:
                edge_list[0].append(i)
                edge_list[1].append(j)
                edge_list[0].append(j)
                edge_list[1].append(i)
                edge_list[2].append(dist)
                edge_list[2].append(dist)
    return edge_list


def get_graph(
    inst_map=None,
    points=None,
    predicted_classes=None,
    true_class_map=None,
    n_rays=None,
    distance=None,
    stardist_checkpoint=None,
    image=None,
    x_type="ll",
    consep_data=False,
    hovernet_metric=False,
):
    """Get the graph from the instance map.

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
        vertex (torch.Tensor): vertex.

    Raises:
        NotImplementedError: if x_type is not known.
    """
    graph = {}
    if stardist_checkpoint is None:
        if x_type == "dist":
            if inst_map is None or n_rays is None:
                raise ValueError(
                    "inst_map and n_rays must be provided for xtype dist and no stardist checkpoint."
                )
            # get stardist result from instance map
            prob, dist = get_stardist_data(inst_map, {"n_rays": n_rays})
            points, _, dists = compute_stardist(dist, prob)
            # add vertex info to graph
            graph["x"] = torch.Tensor(dists)
        if x_type == "c":
            if predicted_classes is None or points is None:
                raise ValueError(
                    "predicted_classes and points must be provided for xtype c and no stardist checkpoint."
                )
            graph["x"] = torch.Tensor(predicted_classes)
        else:
            raise ValueError(
                "x_type not implemented for graph without stardist checkpoint."
            )

    else:
        model_c = Stardist.load_from_checkpoint(
            stardist_checkpoint,
            n_classes=7,
            wandb_log=False,
        )
        model_c = model_c.model.to("cuda")

        with torch.no_grad():
            input = torch.Tensor(image).unsqueeze(0).float().to("cuda")
            dist, prob, _ = model_c(input)
            points, _, _ = compute_stardist(dist, prob)
            # get instance map
            if hovernet_metric:
                graph["inst_map"] = model_c.compute_star_label(input, dist, prob)

        if points.shape[0] == 0:
            graph["x"] = torch.Tensor([])
        else:
            # init models
            if "ll" in x_type:
                model_ll = Stardist.load_from_checkpoint(
                    stardist_checkpoint,
                    n_classes=7,
                    wandb_log=False,
                )
                model_ll = model_ll.model.to("cuda")
                model_ll.output_last_layer = True
            else:
                model_ll = None

            # get stardist points
            if "4" in x_type:
                sp0 = get_stardist_point_for_graph(
                    image, model_ll, model_c, points, x_type=x_type
                )
                sp1 = get_stardist_point_for_graph(
                    image, model_ll, model_c, points, x_type=x_type, rotate=90
                )
                sp2 = get_stardist_point_for_graph(
                    image, model_ll, model_c, points, x_type=x_type, rotate=180
                )
                sp3 = get_stardist_point_for_graph(
                    image, model_ll, model_c, points, x_type=x_type, rotate=270
                )
                stardist_points = torch.cat((sp0, sp1, sp2, sp3), dim=1)
            else:
                stardist_points = get_stardist_point_for_graph(
                    image, model_ll, model_c, points, x_type=x_type
                )
            # set x for graph
            graph["x"] = stardist_points.detach().cpu()

    graph["pos"] = torch.Tensor(points)

    # compute edge information
    edge_list = get_edge_list(points, distance)

    # add edge information to graph
    graph["edge_index"] = torch.tensor([edge_list[0], edge_list[1]], dtype=torch.long)
    graph["edge_attr"] = torch.tensor(edge_list[2], dtype=torch.float)

    # add target to graph
    if true_class_map is not None:
        #
        # get points with target
        y = []
        for i in range(points.shape[0]):
            if type(points[i, 0]) == np.int64:
                yi = true_class_map[points[i, 0], points[i, 1]]
            else:
                # get the 4 nearest points in the class map
                # dataset input centroid: 1, 0
                yi1 = int(true_class_map[int(points[i, 1]), int(points[i, 0])])
                yi2 = int(true_class_map[math.ceil(points[i, 1]), int(points[i, 0])])
                yi3 = int(true_class_map[int(points[i, 1]), math.ceil(points[i, 0])])
                yi4 = int(
                    true_class_map[math.ceil(points[i, 1]), math.ceil(points[i, 0])]
                )
                # if all 4 points have the same class
                if (yi1 == yi2) & (yi2 == yi3) & (yi3 == yi4):
                    yi = yi1
                # if not, take the most common class
                else:
                    possible_y = [yi1, yi2, yi3, yi4]
                    # remove 0
                    possible_y = [x for x in possible_y if x != 0]
                    yi = max(set(possible_y), key=possible_y.count)
            if consep_data:
                if (yi == 3) or (yi == 4):
                    yi = 3
                elif (yi == 5) | (yi == 6) | (yi == 7):
                    yi = 4
            y.append(yi)
        graph["y"] = torch.Tensor(y)

    return graph


def get_stardist_point_for_graph(
    image, model_ll, model_c, points, x_type="ll", rotate=0
):
    """
    Get the stardist point for graph.
    """
    input = torch.Tensor(image).unsqueeze(0).float().to("cuda")
    if rotate != 0:
        input = torch.rot90(input, rotate // 90, [2, 3])
    with torch.no_grad():
        if "ll" in x_type:
            ll = model_ll(input)[0]
        if "c" in x_type:
            c = model_c(input)[2]
    if rotate != 0:
        if "ll" in x_type:
            ll = torch.rot90(ll, -rotate // 90, [2, 3])
        if "c" in x_type:
            c = torch.rot90(c, -rotate // 90, [2, 3])

    stardist_points = []
    for i in range(points.shape[0]):
        if "ll" in x_type:
            lli = torch.select(
                torch.select(ll[0], dim=1, index=points[i, 0]),
                dim=1,
                index=points[i, 1],
            )
        if "c" in x_type:
            ci = torch.select(
                torch.select(c[0], dim=1, index=points[i, 0]),
                dim=1,
                index=points[i, 1],
            )
        if "x" in x_type:
            xi = torch.Tensor([points[i, 0], points[i, 1]])

        # concat everything together
        sp = torch.Tensor([]).to("cuda")
        if "ll" in x_type:
            sp = torch.cat((sp, lli))
        if "c" in x_type:
            sp = torch.cat((sp, ci))
        if "x" in x_type:
            sp = torch.cat((sp, xi))

        stardist_points.append(sp)
    stardist_points = torch.stack(stardist_points)
    return stardist_points


def get_graph_for_inference(batch, distance, stardist_checkpoint, x_type="ll"):
    """Get the graph for inference.

    Args:
        batch (dict): batch.
        distance (int): distance between two vertex to have an edge.
        stardist_checkpoint (str): path to stardist checkpoint.
        x_type (str): type of x.

    Returns:
        graph (dict): graph.
    """
    graphs = []
    for i in range(batch.shape[0]):
        graph = get_graph(
            distance=distance,
            stardist_checkpoint=stardist_checkpoint,
            image=batch[i],
            x_type=x_type,
        )
        graphs.append(graph)
    return graphs
