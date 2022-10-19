import numpy as np
import torch
from stardist import star_dist, edt_prob, non_maximum_suppression


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


def get_stardist_data(inst_map, aditional_args):
    """Get the data for stardist.

    Args:
        inst_map (): instance map.
        aditional_args (dict): additional arguments, must contain n_rays.

    Returns:
        tuple: tuple containing:
            - dist (np.ndarray): distance map.
            - prob (np.ndarray): probability map.

    Raises:
        ValueError: if n_rays is not in aditional_args.
    """
    if "n_rays" not in aditional_args.keys():
        raise ValueError("n_rays not in aditional_args. Mandatory for stardist model.")
    distances = get_stardist_distances(inst_map, aditional_args["n_rays"])
    obj_probabilities = get_stardist_obj_probabilities(inst_map)
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


def get_graph_from_inst_map(inst_map, n_rays, distance):
    """Get the graph from the instance map.

    Args:
        inst_map (): instance map.
        n_rays (int): number of rays of stardist objects.
        distance (int): distance between two vertex to have an edge.

    Returns:
        vertex (torch.Tensor): vertex.
    """
    graph = {}
    # get stardist result from instance map
    prob, dist = get_stardist_data(inst_map, {"n_rays": n_rays})
    points, probs, dists = compute_stardist(dist, prob)

    # add vertex info to graph
    graph["x"] = torch.Tensor(dists)
    graph["pos"] = torch.Tensor(points)

    # compute edge information
    edge_list = get_edge_list(points, distance)

    # add edge information to graph
    graph["edge_index"] = torch.tensor([edge_list[0], edge_list[1]], dtype=torch.long)
    graph["edge_attr"] = torch.tensor(edge_list[2], dtype=torch.float)

    return graph
