"""Copyright (C) SquareFactory SA - All Rights Reserved.

This source code is protected under international copyright law. All rights 
reserved and protected by the copyright holders.
This file is confidential and only available to authorized individuals with the
permission of the copyright holders. If you encounter this file and do not have
permission, please contact the copyright holders and delete this file.
"""

import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import torch

import wandb
from sflizard import Graph, LizardDataModule, LizardGraphDataModule, Stardist

# default values

IN_CHANNELS = 3
N_RAYS = 32
TRAIN_DATA_PATH = "data/Lizard_dataset_extraction/data_0.9_split_train.pkl"
VALID_DATA_PATH = "data/Lizard_dataset_extraction/data_0.9_split_valid.pkl"
TEST_DATA_PATH = "data/Lizard_dataset_extraction/data_0.9_split_test.pkl"
MODEL = "graph_sage"
BATCH_SIZE = 8
NUM_WORKERS = 8
INPUT_SIZE = 540
LEARNING_RATE = 5e-4
SEED = 303
DEFAULT_ROOT_DIR = os.getcwd()
NUM_CLASSES = 7
LOSS_POWER_SCALER = 1.0

DIMH = 1024
NUM_LAYERS = 4
HEADS = 8
NUM_FEATURES = {
    "ll": 128,
    "ll+c": 135,
    "ll+c+x": 137,
    "4ll": 512,
    "4ll+c": 540,
    "4ll+c+x": 548,
}
STARDIST_CHECKPOINT = "models/stardist_1000epochs_0.0losspower_0.0005lr.ckpt"
X_TYPE = "ll+c"
DISTANCE = 45


def init_stardist_training(args, device, debug=False):
    """Init the training for the stardist model."""

    if debug:
        print("init_stardist_training: initialize dm...")

    # create the datamodule
    dm = LizardDataModule(
        train_data_path=args.train_data_path,
        valid_data_path=args.valid_data_path,
        test_data_path=args.test_data_path,
        annotation_target=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=args.input_size,
        seed=args.seed,
        aditional_args={"n_rays": N_RAYS},
    )
    if debug:
        print("init_stardist_training: dm initialized, setup...")
    dm.setup()

    if debug:
        print("init_stardist_training: initialization of dm done.")
        print("init_stardist_training: initialize model...")

    # create the model
    model = Stardist(
        learning_rate=args.learning_rate,
        input_size=args.input_size,
        in_channels=IN_CHANNELS,
        n_rays=N_RAYS,
        n_classes=args.num_classes,
        loss_power_scaler=args.loss_power_scaler,
        seed=args.seed,
        device=device,
        wandb_log=True,
        max_epochs=args.max_epochs,
    )

    loss_callback = pl.callbacks.ModelCheckpoint(
        dirpath="models/loss_cb",
        filename=f"final3-{args.model}-{args.loss_power_scaler}losspower_{args.learning_rate}lr-crop-cosine"
        + "-loss-{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    if debug:
        print("init_stardist_training: model initialized.")

    return dm, model, [loss_callback]


def init_graph_training(args):
    """Init the training for the graphSage model."""
    # get the train data
    train_data_path = Path(args.data_path)
    with train_data_path.open("rb") as f:
        train_data = pickle.load(f)
    train_data = train_data["annotations"]

    # get the test data
    test_data_path = Path(args.test_data_path)
    with test_data_path.open("rb") as f:
        test_data = pickle.load(f)
        test_data = test_data["annotations"]

    # create the datamodule
    dm = LizardGraphDataModule(
        train_data=train_data,
        test_data_path=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        stardist_checkpoint=STARDIST_CHECKPOINT,
        x_type=args.x_type,
        distance=args.distance,
    )
    dm.setup()

    # create the model
    model = Graph(
        model=args.model,
        learning_rate=args.learning_rate,
        num_features=NUM_FEATURES[args.x_type],
        num_classes=args.num_classes,
        seed=args.seed,
        max_epochs=args.max_epochs,
        dim_h=args.dimh,
        num_layers=args.num_layers,
        heads=args.heads,
    )

    loss_callback = pl.callbacks.ModelCheckpoint(
        dirpath="models/loss_cb_graph",
        filename=f"final2-{args.model}-{args.dimh}-{args.num_layers}-{args.x_type}-{args.distance}-{args.learning_rate}"
        + "-loss-{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    return dm, model, loss_callback


def full_training(args):
    """Train the model on the whole dataset."""

    # get the choosen device
    device = torch.device(
        "cuda"
        if (
            ("gpus" in args and args.gpus is not None and args.gpus > 0)
            or (
                "accelerator" in args
                and args.accelerator is not None
                and args.accelerator == "gpu"
            )
        )
        else "cpu"
    )

    if "stardist" in args.model:
        dm, model, callbacks = init_stardist_training(args, device)
    elif "graph" in args.model:
        dm, model, callbacks = init_graph_training(args)
    else:
        raise ValueError("Model not implemented.")

    # create the trainer
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)

    # # train the model
    trainer.fit(model, dm)

    # # save the model
    datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    if "stardist" in args.model:
        trainer.save_checkpoint(
            f"models/final3_stardist_crop-cosine_{args.max_epochs}epochs_{args.loss_power_scaler}losspower_{args.learning_rate}lr.ckpt"
        )
    else:
        trainer.save_checkpoint(
            f"models/{args.model}_{args.dimh}dh_{args.num_layers}lay_{args.x_type}_{args.distance}dist_{args.max_epochs}epochs_{args.learning_rate}lr.ckpt"
        )

    # run test on single GPU to avoir bias (see:https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metrics-in-distributed-data-parallel-ddp-mode)
    if args.gpus and args.gpus > 1:
        torch.distributed.destroy_process_group()
        if trainer.is_global_zero:
            trainer = pl.Trainer(gpus=1)

    # trainer.test(model, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "-tdp",
        "--train_data_path",
        type=str,
        default=TRAIN_DATA_PATH,
        help="Path to the .pkl file containing the train data.",
    )
    parser.add_argument(
        "-vdp",
        "--valid_data_path",
        type=str,
        default=VALID_DATA_PATH,
        help="Path to the .pkl file containing the validation data.",
    )
    parser.add_argument(
        "-tp",
        "--test_data_path",
        type=str,
        default=TEST_DATA_PATH,
        help="Path to the .pkl file containing the test data.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=MODEL,
        help="Model to train. Can be 'stardist' or ...",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size to use for the dataloaders.",
    )
    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of workers to use for the dataloaders.",
    )
    parser.add_argument(
        "-is",
        "--input_size",
        type=int,
        default=INPUT_SIZE,
        help="Input size to use for the dataloaders.",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=LEARNING_RATE,
        help="Learning rate to use for the optimizer.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=SEED,
        help="Seed to use for the dataloaders.",
    )
    parser.add_argument(
        "--default-root-dir",
        type=str,
        help="Directory to save the trained weights to.",
        default=DEFAULT_ROOT_DIR,
    )
    parser.add_argument(
        "-nc",
        "--num_classes",
        type=int,
        default=NUM_CLASSES,
        help="Number of classes to use for the stardist model.",
    )
    parser.add_argument(
        "-lps",
        "--loss_power_scaler",
        type=float,
        default=LOSS_POWER_SCALER,
        help="Loss scaler to use for the stardist model.",
    )
    parser.add_argument(
        "-dh",
        "--dimh",
        type=int,
        default=DIMH,
        help="Dimension of the hidden layer in the grap model.",
    )
    parser.add_argument(
        "-nl",
        "--num_layers",
        type=int,
        default=NUM_LAYERS,
        help="Number of layers in the grap model.",
    )
    parser.add_argument(
        "-he",
        "--heads",
        type=int,
        default=HEADS,
        help="Number of heads in the grap model.",
    )
    parser.add_argument(
        "-xt",
        "--x_type",
        type=str,
        default=X_TYPE,
        help="Type of the input in the grap model.",
    )
    parser.add_argument(
        "-d",
        "--distance",
        type=int,
        default=DISTANCE,
        help="Distance to use for the graph model.",
    )
    parser.add_argument(
        "-sn",
        "--save_name",
        type=str,
        default="",
        help="Name to add to the saved model.",
    )

    args = parser.parse_args()

    # wandb logging (see: https://wandb.ai)
    wandb_config = {
        "model": args.model,
        "learning_rate": args.learning_rate,
        "epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "num_classes": args.num_classes,
        "loss_power_scaler": args.loss_power_scaler,
        "dimh": args.dimh,
        "num_layers": args.num_layers,
        "data": "shuffle",
        "save_name": args.save_name,
    }
    wandb.init(project="sflizard", entity="leonardfavre", config=wandb_config)

    # Set seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    full_training(args)
