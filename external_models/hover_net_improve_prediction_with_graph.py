from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
import pandas as pd

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
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_2layer_sage_4h.ckpt",
        #     "2-4",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_4layer_sage_4h.ckpt",
        #     "4-4",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_8layer_sage_4h.ckpt",
        #     "8-4",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_16layer_sage_4h.ckpt",
        #     "16-4",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_2layer_sage_8h.ckpt",
        #     "2-8",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_4layer_sage_8h.ckpt",
        #     "4-8",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_8layer_sage_8h.ckpt",
        #     "8-8",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_16layer_sage_8h.ckpt",
        #     "16-8",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_2layer_sage_16h.ckpt",
        #     "2-16",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_4layer_sage_16h.ckpt",
        #     "4-16",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_500epochs_8layer_sage_16h.ckpt",
        #     "8-16",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_16layer_sage_16h.ckpt",
        #     "16-16",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_2layer_sage_32h.ckpt",
        #     "2-32",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_4layer_sage_32h.ckpt",
        #     "4-32",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_8layer_sage_32h.ckpt",
        #     "8-32",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_16layer_sage_32h.ckpt",
        #     "16-32",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_500epochs_2layer_sage_64h.ckpt",
        #     "2-64",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_4layer_sage_64h.ckpt",
        #     "4-64",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_8layer_sage_64h.ckpt",
        #     "8-64",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_16layer_sage_64h.ckpt",
        #     "16-64",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_2layer_sage_128h.ckpt",
        #     "2-128",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_4layer_sage_128h.ckpt",
        #     "4-128",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_8layer_sage_128h.ckpt",
        #     "8-128",
        # ],
        # [
        #     "checkpoints/full_training_hover_net_lizard_graph_200epochs_16layer_sage_128h.ckpt",
        #     "16-128",
        # ],
        # [
        #     "checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_2layer_sage_64h-accmacro-epoch=130-val_acc_macro=0.6160.ckpt",
        #     "2-64-acc-macro",
        # ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_2layer_sage_16h-acc-epoch=105-val_acc=0.8032.ckpt',
            "2-16-acc-macro-1",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_2layer_sage_16h-accmacro-epoch=97-val_acc_macro=0.6155.ckpt',
            "2-16-acc-macro-0",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_2layer_sage_32h-acc-epoch=126-val_acc=0.8032.ckpt',
            "2-32-acc-macro-1",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_2layer_sage_32h-accmacro-epoch=234-val_acc_macro=0.6143.ckpt',
            "2-32-acc-macro-0",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_2layer_sage_64h-acc-epoch=187-val_acc=0.8041.ckpt',
            "2-64-acc-macro-1",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_2layer_sage_64h-acc-epoch=30-val_acc=0.8042.ckpt',
            "2-64-acc-macro-2",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_2layer_sage_64h-accmacro-epoch=130-val_acc_macro=0.6160.ckpt',
            "2-64-acc-macro-3",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_2layer_sage_64h-accmacro-epoch=157-val_acc_macro=0.6149.ckpt',
            "2-64-acc-macro-0",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_2layer_sage_128h-acc-epoch=105-val_acc=0.8038.ckpt',
            "2-128-acc-macro-1",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_2layer_sage_128h-accmacro-epoch=143-val_acc_macro=0.6151.ckpt',
            "2-128-acc-macro-0",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_4layer_sage_16h-acc-epoch=8-val_acc=0.8021.ckpt',
            "4-16-acc-macro-1",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_4layer_sage_16h-accmacro-epoch=100-val_acc_macro=0.6127.ckpt',
            "4-16-acc-macro-0",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_4layer_sage_32h-acc-epoch=35-val_acc=0.8029.ckpt',
            "4-32-acc-macro-1",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_4layer_sage_32h-accmacro-epoch=42-val_acc_macro=0.6085.ckpt',
            "4-32-acc-macro-0",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_4layer_sage_64h-acc-epoch=36-val_acc=0.8021.ckpt',
            "4-64-acc-macro-1",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_4layer_sage_64h-accmacro-epoch=27-val_acc_macro=0.6152.ckpt',
            "4-64-acc-macro-0",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_4layer_sage_128h-acc-epoch=24-val_acc=0.8019.ckpt',
            "4-128-acc-macro-1",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_4layer_sage_128h-accmacro-epoch=41-val_acc_macro=0.6140.ckpt',
            "4-128-acc-macro-0",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_8layer_sage_16h-acc-epoch=17-val_acc=0.8021.ckpt',
            "8-16-acc-macro-1",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_8layer_sage_16h-accmacro-epoch=17-val_acc_macro=0.6108.ckpt',
            "8-16-acc-macro-0",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_8layer_sage_32h-acc-epoch=34-val_acc=0.8016.ckpt',
            "8-32-acc-macro-1",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_8layer_sage_32h-accmacro-epoch=33-val_acc_macro=0.6104.ckpt',
            "8-32-acc-macro-0",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_8layer_sage_64h-acc-epoch=65-val_acc=0.8007.ckpt',
            "8-64-acc-macro-1",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_8layer_sage_64h-accmacro-epoch=15-val_acc_macro=0.6126.ckpt',
            "8-64-acc-macro-0",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_8layer_sage_128h-acc-epoch=81-val_acc=0.8008.ckpt',
            "8-128-acc-macro-1",
        ],
        [
            'checkpoints/cp_acc/fin_training_hover_net_lizard_graph_500epochs_8layer_sage_128h-accmacro-epoch=30-val_acc_macro=0.6100.ckpt',
            "8-128-acc-macro-0",
        ],
    ]

    SEED = 303
    scope = "test"
    dataset = "Lizard"
    if scope == "test":
        data_path = f"output/{dataset}_test_out/mat/"
        file_list = list(Path(data_path).glob("*.mat"))
    else:
        if dataset == "CoNSeP":
            data_path = f"output/{dataset}_train_out/mat/"
            file_list = list(Path(data_path).glob("*.mat"))
            df = pd.DataFrame(
                {
                    "files": file_list,
                }
            )
            valid_df = df.sample(frac = 0.2, random_state=SEED)
            train_df = df.drop(valid_df.index)
            file_list = valid_df["files"].tolist()
        else:
            data_path = f"output/{dataset}_valid_out/mat/"
            file_list = list(Path(data_path).glob("*.mat"))

    for model_path, save_folder in model_list:

        dim_h = int(save_folder.split("-")[1])
        num_layers = int(save_folder.split("-")[0])

        model = init_graph_inference(model_path)
        model.to(device)

        if scope == "valid" and dataset == "CoNSeP":
            save_path = f"output/{dataset}_train_out/graph/{save_folder}/"
        elif scope == "valid":
            save_path = f"output/{dataset}_valid_out/graph/{save_folder}/"
        else:
            save_path = f"output/{dataset}_test_out/graph/{save_folder}/"
        Path(save_path).mkdir(parents=True, exist_ok=True)

        for file_path in file_list:
            print(file_path)
            base_name = file_path.stem
            pred = sio.loadmat(file_path)

            points = pred["inst_centroid"]
            predicted_class = pred["inst_type"]

            if len(points) == 0:
                continue
            # graph predicted mask
            graph = get_graph(
                points=points, 
                predicted_classes=predicted_class, 
                distance=45, 
                x_type="c", 
            )
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
