import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.io as sio
import torch

from sflizard import Graph, LizardGraphDataModule

###### GRAPH

CONSEP = False
TRUE_DATA_PATH = "../data/Lizard_dataset_test_split/Lizard_Labels_train/Labels/"  # "../data/Lizard_dataset_test_split/Lizard_Labels_train/Labels/"
TRAIN_DATA_PATH = "hover_net/Lizard_train_out/mat/"
NUM_CLASSES = 7  # 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument(
        "-dh",
        "--dimh",
        type=int,
        default=32,
        help="Dimension of the hidden layer in the grap model.",
    )
    parser.add_argument(
        "-l",
        "--layers",
        type=int,
        default=4,
        help="Number of layers in the grap model.",
    )

    args = parser.parse_args()

    # load the true data
    data_path = TRUE_DATA_PATH
    file_list = list(Path(data_path).glob("*.mat"))

    class_map = []

    for file_path in file_list:
        base_name = file_path.stem
        mat = sio.loadmat(file_path)
        if not CONSEP:
            ann_inst = mat["inst_map"]
            nuclei_id = np.squeeze(mat["id"]).tolist()
            patch_id = np.unique(ann_inst).tolist()[1:]
            type_map = np.zeros(ann_inst.shape)
            for v in patch_id:
                idn = nuclei_id.index(v)
                type_map[ann_inst == v] = mat["class"][idn]
            type_map = type_map.astype("int32")
            class_map.append(type_map)
        else:
            class_map.append(mat["type_map"])

    # load the predicted data
    data_path = TRAIN_DATA_PATH
    file_list = list(Path(data_path).glob("*.mat"))

    points = []
    predicted_class = []

    for file_path in file_list:
        mat = sio.loadmat(file_path)
        points.append(mat["inst_centroid"]),
        predicted_class.append(mat["inst_type"]),

    # create df with the data
    train_data = pd.DataFrame(
        {
            "points": points,
            "predicted_class": predicted_class,
            "class_map": class_map,
        }
    )

    # create the datamodule
    dm = LizardGraphDataModule(
        train_data=train_data,
        x_type="c",
        root="data/graph_hovernet",
        concep_data=CONSEP,
    )
    dm.setup()

    # train the models with different parameters
    for dim_h in [32, 64, 128]:
        for num_layers in [2, 4, 8, 16]:
            # create the model
            model = Graph(
                model="graph_sage",
                learning_rate=0.01,
                num_features=1,
                num_classes=NUM_CLASSES,
                max_epochs=args.max_epochs,
                dim_h=dim_h,
                num_layers=num_layers,
                class_weights=None,
            )

            device = torch.device("cuda")

            # create the trainer
            trainer = pl.Trainer.from_argparse_args(args)

            # # train the model
            trainer.fit(model, dm)

            # # save the model
            trainer.save_checkpoint(
                f"models/full_training_hover_net_lizard_graph_{args.max_epochs}epochs_{num_layers}layer_sage_{dim_h}h.ckpt"
            )
