"""Copyright (C) SquareFactory SA - All Rights Reserved.

This source code is protected under international copyright law. All rights 
reserved and protected by the copyright holders.
This file is confidential and only available to authorized individuals with the
permission of the copyright holders. If you encounter this file and do not have
permission, please contact the copyright holders and delete this file.
"""

import argparse
import os

import pytorch_lightning as pl
import torch

from sflizard import Graph, LizardDataModule, LizardGraphDataModule, Stardist


def init_stardist_training(args):
    """Init the training for the stardist model."""

    IN_CHANNELS = 3
    N_RAYS = 32

    # create the datamodule
    dm = LizardDataModule(
        data_path=args.data_path,
        annotation_target="stardist",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=args.input_size,
        seed=args.seed,
        aditional_args={"n_rays": N_RAYS},
    )
    dm.setup()

    # create the model
    model = Stardist(
        learning_rate=args.learning_rate,
        input_size=args.input_size,
        in_channels=IN_CHANNELS,
        n_rays=N_RAYS,
        seed=args.seed,
    )

    return dm, model


def init_graph_training(args):
    """Init the training for the graphSage model."""

    NUM_FEATURES = 32
    NUM_CLASSES = 7
    DIM_H = 256
    NUM_LAYERS = 8

    # create the datamodule
    dm = LizardGraphDataModule(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    dm.setup()

    # create the model
    model = Graph(
        model=args.model,
        learning_rate=args.learning_rate,
        num_features=NUM_FEATURES,
        num_classes=NUM_CLASSES,
        seed=args.seed,
        max_epochs=args.max_epochs,
        dim_h=DIM_H,
        num_layers=NUM_LAYERS,
    )

    return dm, model


def full_training(args):
    """Train the model on the whole dataset."""

    if args.model == "stardist":
        dm, model = init_stardist_training(args)
    elif "graph" in args.model:
        dm, model = init_graph_training(args)
    else:
        raise ValueError("Model not implemented.")

    # create the trainer
    trainer = pl.Trainer.from_argparse_args(args)

    # # train the model
    trainer.fit(model, dm)

    # # save the model
    trainer.save_checkpoint(f"models/full_training_{args.model}_{args.max_epochs}.ckpt")

    # run test on single GPU to avoir bias (see:https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metrics-in-distributed-data-parallel-ddp-mode)
    if args.gpus and args.gpus > 1:
        torch.distributed.destroy_process_group()
        if trainer.is_global_zero:
            trainer = pl.Trainer(gpus=1)

    trainer.test(model, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument(
        "-dp",
        "--data_path",
        type=str,
        help="Path to the .pkl file containing the data.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="stardist",
        help="Model to train. Can be 'stardist' or ...",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=32,
        help="Batch size to use for the dataloaders.",
    )
    parser.add_argument(
        "-nw",
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers to use for the dataloaders.",
    )
    parser.add_argument(
        "-is",
        "--input_size",
        type=int,
        default=540,
        help="Input size to use for the dataloaders.",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use for the optimizer.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=303,
        help="Seed to use for the dataloaders.",
    )
    parser.add_argument(
        "--default-root-dir",
        type=str,
        help="Directory to save the trained weights to.",
        default=os.getcwd(),
    )
    args = parser.parse_args()

    # Set seed for reproducibility

    pl.seed_everything(args.seed, workers=True)

    full_training(args)
