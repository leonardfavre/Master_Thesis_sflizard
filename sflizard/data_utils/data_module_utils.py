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


def get_graph_from_inst_map(
    inst_map, 
    class_map=None,
    n_rays=None, 
    distance=None, 
    stardist_checkpoint=None, 
    image=None,
    x_type="ll"
    ):
    """Get the graph from the instance map.

    Args:
        inst_map (): instance map.
        class_map (): class map.
        n_rays (int): number of rays of stardist objects.
        distance (int): distance between two vertex to have an edge.
        stardist_checkpoint (str): path to stardist checkpoint.

    Returns:
        vertex (torch.Tensor): vertex.
    """
    graph = {}
    # get stardist result from instance map
    prob, dist = get_stardist_data(inst_map, {"n_rays": n_rays})
    points, probs, dists = compute_stardist(dist, prob)

    # get points with target
    y = []

    for i in range(points.shape[0]):
        y.append(class_map[points[i, 0], points[i, 1]])

    # is stardist, add stardist last layer to x
    if stardist_checkpoint is not None:
        model = Stardist.load_from_checkpoint(
            stardist_checkpoint,
            n_classes=7,
        )
        model = model.model.to('cuda')
        model.output_last_layer = True
        input = torch.Tensor(image).unsqueeze(0).float().to('cuda')
        with torch.no_grad():
            x = model(input)[0]
        # stardist_map = torch.argmax(stardist_map[0], dim=0)
        stardist_points = []
        for i in range(points.shape[0]):
            stardist_points.append(
                torch.select(
                    torch.select(x[0], dim=1, index=points[i, 0]),
                    dim=1,
                    index=points[i, 1],
                )
            )
        stardist_points = torch.stack(stardist_points)
        # stardist_points = torch.Tensor(stardist_points).unsqueeze(1)
        graph["x"] = stardist_points.detach().cpu()

    else:
        # add vertex info to graph
        graph["x"] = torch.Tensor(dists)

    graph["pos"] = torch.Tensor(points)
    graph["y"] = torch.Tensor(y)

    # compute edge information
    edge_list = get_edge_list(points, distance)

    # add edge information to graph
    graph["edge_index"] = torch.tensor([edge_list[0], edge_list[1]], dtype=torch.long)
    graph["edge_attr"] = torch.tensor(edge_list[2], dtype=torch.float)

    return graph

def get_graph_for_inference(dist, prob, class_map, distance, last_layer=None):
    """Get the graph for inference.

    Args:
        dist (torch.Tensor): distance map.
        prob (torch.Tensor): probability map.
        class_map (): class map.
        distance (int): distance between two vertex to have an edge.
        last_layer (torch.Tensor): last layer of stardist.

    Returns:
        graph (dict): graph.
    """
    graphs = []
    # get stardist result from instance map
    # prob, dist = get_stardist_data(inst_map, {"n_rays": n_rays})
    for i in range(dist.shape[0]):
        graph = {}
        points, probs, dists = compute_stardist(dist[i], prob[i])
        if len(points > 0):
            if last_layer is not None:
                # stardist_map = torch.argmax(stardist_map[0], dim=0)
                ll = last_layer[i]
                stardist_points = []
                for j in range(points.shape[0]):
                    stardist_points.append(
                        torch.select(
                            torch.select(ll, dim=1, index=points[j, 0]),
                            dim=1,
                            index=points[j, 1],
                        )
                    )
                
                stardist_points = torch.stack(stardist_points)
                # stardist_points = torch.Tensor(stardist_points).unsqueeze(1)
                graph["x"] = stardist_points.detach().cpu()

            else:
                # add vertex info to graph
                graph["x"] = torch.Tensor(dists)
                # add class info to x
                stardist_points = []
                for j in range(points.shape[0]):
                    stardist_points.append(class_map[i, points[j, 0], points[j, 1]])
                stardist_points = torch.Tensor(stardist_points).unsqueeze(1)
                graph["x"] = torch.cat((graph["x"], stardist_points), dim=1)
            
            graph["pos"] = torch.Tensor(points)

            # compute edge information
            edge_list = get_edge_list(points, distance)
            # add edge information to graph
            graph["edge_index"] = torch.tensor([edge_list[0], edge_list[1]], dtype=torch.long)
            graph["edge_attr"] = torch.tensor(edge_list[2], dtype=torch.float)
            graphs.append(graph)
        else:
            graph["x"] = torch.Tensor([])
            graph["pos"] = torch.Tensor([])
            graph["edge_index"] = torch.tensor([[], []], dtype=torch.long)
            graph["edge_attr"] = torch.tensor([], dtype=torch.float)
            graphs.append(graph)
    return graphs

    