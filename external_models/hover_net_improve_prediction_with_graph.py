from pathlib import Path

import numpy as np
import scipy.io as sio
import torch


from sflizard import Graph, get_graph


def init_graph_inference(weights_path: str) -> None:
    print("Loading graph model...")
    model = Graph.load_from_checkpoint(
        weights_path,
    )
    graph = model.model
    print("Graph model loaded.")
    return graph


if __name__ == "__main__":
    device = "cuda"

    model_list = [
        # ["models/full_training_hover_net_graph_500epochs_1layer_sage_16h.ckpt", "1-16"],
        # ["models/full_training_hover_net_graph_500epochs_2layer_sage_16h.ckpt", "2-16"],
        # ["models/full_training_hover_net_graph_500epochs_3layer_sage_16h.ckpt", "3-16"],
        # ["models/full_training_hover_net_graph_500epochs_4layer_sage_16h.ckpt", "4-16"],
        # ["models/full_training_hover_net_graph_500epochs_5layer_sage_16h.ckpt", "5-16"],
        # ["models/full_training_hover_net_graph_500epochs_1layer_sage_32h.ckpt", "1-32"],
        # ["models/full_training_hover_net_graph_500epochs_2layer_sage_32h.ckpt", "2-32"],
        # ["models/full_training_hover_net_graph_500epochs_3layer_sage_32h.ckpt", "3-32"],
        # ["models/full_training_hover_net_graph_500epochs_4layer_sage_32h.ckpt", "4-32"],
        # ["models/full_training_hover_net_graph_500epochs_5layer_sage_32h.ckpt", "5-32"],
        # ["models/full_training_hover_net_graph_500epochs_1layer_sage_64h.ckpt", "1-64"],
        # ["models/full_training_hover_net_graph_500epochs_2layer_sage_64h.ckpt", "2-64"],
        # ["models/full_training_hover_net_graph_500epochs_3layer_sage_64h.ckpt", "3-64"],
        # ["models/full_training_hover_net_graph_500epochs_4layer_sage_64h.ckpt", "4-64"],
        # ["models/full_training_hover_net_graph_500epochs_5layer_sage_64h.ckpt", "5-64"],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_2layer_sage_4h.ckpt",
            "2-4",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_4layer_sage_4h.ckpt",
            "4-4",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_8layer_sage_4h.ckpt",
            "8-4",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_16layer_sage_4h.ckpt",
            "16-4",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_2layer_sage_8h.ckpt",
            "2-8",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_4layer_sage_8h.ckpt",
            "4-8",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_8layer_sage_8h.ckpt",
            "8-8",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_16layer_sage_8h.ckpt",
            "16-8",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_2layer_sage_16h.ckpt",
            "2-16",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_4layer_sage_16h.ckpt",
            "4-16",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_8layer_sage_16h.ckpt",
            "8-16",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_16layer_sage_16h.ckpt",
            "16-16",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_2layer_sage_32h.ckpt",
            "2-32",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_4layer_sage_32h.ckpt",
            "4-32",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_8layer_sage_32h.ckpt",
            "8-32",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_16layer_sage_32h.ckpt",
            "16-32",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_2layer_sage_64h.ckpt",
            "2-64",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_4layer_sage_64h.ckpt",
            "4-64",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_8layer_sage_64h.ckpt",
            "8-64",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_16layer_sage_64h.ckpt",
            "16-64",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_2layer_sage_128h.ckpt",
            "2-128",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_4layer_sage_128h.ckpt",
            "4-128",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_8layer_sage_128h.ckpt",
            "8-128",
        ],
        [
            "models/full_training_hover_net_lizard_graph_500epochs_16layer_sage_128h.ckpt",
            "16-128",
        ],
    ]

    data_path = "hover_net/Lizard_test_out/mat/"
    file_list = list(Path(data_path).glob("*.mat"))

    for model_path, save_folder in model_list:

        dim_h = int(save_folder.split("-")[1])
        num_layers = int(save_folder.split("-")[0])

        model = init_graph_inference(model_path)
        model.to(device)

        save_path = f"hover_net/Lizard_test_out/graph/{save_folder}/"
        Path(save_path).mkdir(parents=True, exist_ok=True)

        for file_path in file_list:
            base_name = file_path.stem
            pred = sio.loadmat(file_path)

            points = pred["inst_centroid"]
            predicted_class = pred["inst_type"]
            # graph predicted mask
            graph = get_graph(points=points, predicted_class=predicted_class, distance=45, x_type="c")
            with torch.no_grad():
                out = model(
                    graph["x"].to(device),
                    graph["edge_index"].to(device),
                )
                graph_pred = out.argmax(-1)
            pred["inst_type"] = graph_pred.cpu().numpy()
            pred["inst_type"] = np.reshape(
                pred["inst_type"], (pred["inst_type"].shape[0], 1)
            )

            sio.savemat(f"{save_path}{base_name}.mat", pred)
