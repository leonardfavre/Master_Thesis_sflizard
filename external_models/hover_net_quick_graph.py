import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.io as sio
import torch
from tqdm import tqdm

from sflizard import Graph, LizardGraphDataModule

# GRAPH

SEED = 303

CONSEP = False
TRUE_DATA_PATH = "../data/Lizard_dataset_split/patches/Lizard_Labels_train/"  # lizard
TRUE_VALID_DATA_PATH = (
    "../data/Lizard_dataset_split/patches/Lizard_Labels_valid/"  # lizard
)
TRAIN_DATA_PATH = "output/Lizard_train_out/mat/"  # Lizard
VALID_DATA_PATH = "output/Lizard_valid_out/mat/"  # Lizard
NUM_CLASSES = 7  # Lizard

# CONSEP = True
# TRUE_DATA_PATH = "../data/CoNSeP/Train/Labels/" # CoNSeP
# TRAIN_DATA_PATH = "output/CoNSeP_train_out/mat/" # CoNSeP
# NUM_CLASSES = 5  # CoNSeP


def get_df(true_path, data_path):
    file_list = list(Path(data_path).glob("*.mat"))

    class_map = []
    points = []
    predicted_class = []

    print("Loading data")
    for file_path in tqdm(file_list):
        mat = sio.loadmat(file_path)

        if len(mat["inst_centroid"]) > 0:
            points.append(mat["inst_centroid"])
            predicted_class.append(mat["inst_type"])

            mat = sio.loadmat(str(file_path).replace(data_path, true_path))
            if not CONSEP:
                ann_inst = mat["inst_map"]
                nuclei_id = np.squeeze(mat["nuclei_id"]).tolist()
                if type(nuclei_id) != list:
                    nuclei_id = [nuclei_id]
                patch_id = np.unique(ann_inst).tolist()[1:]
                type_map = np.zeros(ann_inst.shape)
                for v in patch_id:
                    idn = nuclei_id.index(v)
                    type_map[ann_inst == v] = mat["classes"][idn]
                type_map = type_map.astype("int32")
                class_map.append(type_map)
            else:
                class_map.append(mat["type_map"])
    # create df with the data
    df = pd.DataFrame(
        {
            "points": points,
            "predicted_classes": predicted_class,
            "class_map": class_map,
        }
    )
    return df


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

    valid_df = get_df(TRUE_VALID_DATA_PATH, VALID_DATA_PATH)
    train_df = get_df(TRUE_DATA_PATH, TRAIN_DATA_PATH)

    print("Creating datamodule")
    # create the datamodule
    dm = LizardGraphDataModule(
        train_data=train_df,
        valid_data=valid_df,
        x_type="c",
        root="data/graph_hovernet_lizard",
        consep_data=CONSEP,
    )
    dm.setup()

    # train the models with different parameters
    for dim_h in [16, 32, 64, 128]:  # [4, 8, 16, 32, 64, 128]:
        for num_layers in [2, 4, 8]:  # [2, 4, 8, 16]:
            print(
                "Creating model with dim_h = ", dim_h, " and num_layers = ", num_layers
            )
            # create the model
            model = Graph(
                model="graph_sage",
                learning_rate=0.01,
                num_features=1,
                num_classes=NUM_CLASSES,
                max_epochs=args.max_epochs,
                dim_h=dim_h,
                num_layers=num_layers,
            )

            acc_callback = pl.callbacks.ModelCheckpoint(
                dirpath="checkpoints/cp_acc",
                filename=f"fin_training_hover_net_lizard_graph_{args.max_epochs}epochs_{num_layers}layer_sage_{dim_h}h"
                + "-acc-{epoch}-{val_acc:.4f}",
                monitor="val_acc",
                mode="max",
                save_top_k=1,
            )

            acc_macro_callback = pl.callbacks.ModelCheckpoint(
                dirpath="checkpoints/cp_acc",
                filename=f"fin_training_hover_net_lizard_graph_{args.max_epochs}epochs_{num_layers}layer_sage_{dim_h}h"
                + "-accmacro-{epoch}-{val_acc_macro:.4f}",
                monitor="val_acc_macro",
                mode="max",
                save_top_k=1,
            )

            device = torch.device("cuda")

            # create the trainer
            trainer = pl.Trainer.from_argparse_args(
                args, callbacks=[acc_callback, acc_macro_callback]
            )

            # # train the model
            trainer.fit(model, dm)

            # # save the model
            trainer.save_checkpoint(
                f"checkpoints/fin_training_hover_net_lizard_graph_{args.max_epochs}epochs_{num_layers}layer_sage_{dim_h}h.ckpt"
            )
