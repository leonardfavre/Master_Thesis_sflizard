import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union

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
BATCH_SIZE = 64
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
CUSTOM_INPUT_LAYER = 0
CUSTOM_OUTPUT_LAYER = 0
CUSTOM_INPUT_HIDDEN = 8
CUSTOM_OUTPUT_HIDDEN = NUM_CLASSES
CUSTOM_WIDE_CONNECTIONS = False
DROPOUT = 0.0

NUM_FEATURES = {
    "c": 7,
    "c+x": 9,
    "ll": 128,
    "ll+c": 135,
    "ll+c+x": 137,
    "4ll": 512,
    "4ll+c": 540,
    "4ll+c+x": 542,
}
STARDIST_CHECKPOINT = (
    "models/final3_stardist_crop-cosine_200epochs_1.0losspower_0.0005lr.ckpt"
)
X_TYPE = "4ll+c"
DISTANCE = 45


def init_stardist_training(
    args: argparse.Namespace, device: Union[str, torch.device], debug: bool = False
) -> Tuple[LizardDataModule, Stardist, List[pl.callbacks.Callback]]:
    """Init the training for the stardist model.

    Args:
        args (argparse.Namespace): the arguments from the command line.
        device (Union[str, torch.device]): the device to use.
        debug (bool): if True, print debug messages.

    Returns:
        tuple: tuple containing:
            dm (LizardDataModule): the datamodule.
            model (Stardist): the model.
            callbacks (List[pl.callbacks.Callback]): the callbacks.

    Raises:
        None.
    """

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

    return (
        dm,
        model,
        [loss_callback],
    )


def init_graph_training(
    args: argparse.Namespace,
) -> Tuple[pl.LightningDataModule, Graph, List[pl.callbacks.Callback]]:
    """Init the training for the graphSage model.

    Args:
        args (argparse.Namespace): the arguments from the command line.

    Returns:
        tuple: tuple containing:
            dm (LizardGraphDataModule): the datamodule.
            model (Graph): the model.
            callbacks (List[pl.callbacks.Callback]): the callbacks.

    Raises:
        None.
    """
    # get the train data
    train_data_path = Path(args.train_data_path)
    with train_data_path.open("rb") as f:
        train_data = pickle.load(f)
    # train_data = train_data["annotations"]

    # get the valid data
    valid_data_path = Path(args.valid_data_path)
    with valid_data_path.open("rb") as f:
        valid_data = pickle.load(f)
        # valid_data = valid_data["annotations"]

    # get the test data
    test_data_path = Path(args.test_data_path)
    with test_data_path.open("rb") as f:
        test_data = pickle.load(f)
        # test_data = test_data["annotations"]

    # create the datamodule
    dm = LizardGraphDataModule(
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        stardist_checkpoint=STARDIST_CHECKPOINT,
        x_type=args.x_type,
        distance=args.distance,
        light=True,
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
        custom_input_layer=args.custom_input_layer,
        custom_input_hidden=args.custom_input_hidden,
        custom_output_layer=args.custom_output_layer,
        custom_output_hidden=args.custom_output_hidden,
        custom_wide_connections=args.custom_wide_connections,
        wandb_log=True,
        dropout=args.dropout,
    )

    print(f"\nwide: {args.custom_wide_connections}\n")

    if args.dropout == 0.0:
        if args.model == "graph_gat":
            name = f"{args.model}-{args.dimh}-{args.num_layers}-{args.x_type}-{args.distance}-{args.heads}-{args.learning_rate}"
        elif args.model == "graph_custom" and not args.custom_wide_connections:
            name = f"{args.model}-{args.dimh}-{args.num_layers}-{args.x_type}-{args.distance}-{args.custom_input_layer}-{args.custom_input_hidden}-{args.custom_output_layer}-{args.custom_output_hidden}-{args.learning_rate}"
        elif args.model == "graph_custom" and args.custom_wide_connections:
            name = f"{args.model}-{args.dimh}-{args.num_layers}-{args.x_type}-{args.distance}-{args.custom_input_layer}-{args.custom_input_hidden}-{args.custom_output_layer}-{args.custom_output_hidden}-wide-{args.learning_rate}"
        else:
            name = f"{args.model}-{args.dimh}-{args.num_layers}-{args.x_type}-{args.distance}-{args.learning_rate}"
    else:
        if args.model == "graph_gat":
            name = f"{args.model}-{args.dimh}-{args.num_layers}-{args.x_type}-{args.distance}-{args.heads}-{args.dropout}-{args.learning_rate}"
        elif args.model == "graph_custom" and not args.custom_wide_connections:
            name = f"{args.model}-{args.dimh}-{args.num_layers}-{args.x_type}-{args.distance}-{args.dropout}-{args.custom_input_layer}-{args.custom_input_hidden}-{args.custom_output_layer}-{args.custom_output_hidden}-{args.learning_rate}"
        elif args.model == "graph_custom" and args.custom_wide_connections:
            name = f"{args.model}-{args.dimh}-{args.num_layers}-{args.x_type}-{args.distance}-{args.dropout}-{args.custom_input_layer}-{args.custom_input_hidden}-{args.custom_output_layer}-{args.custom_output_hidden}-wide-{args.learning_rate}"
        else:
            name = f"{args.model}-{args.dimh}-{args.num_layers}-{args.x_type}-{args.distance}-{args.dropout}-{args.learning_rate}"

    print(f"\nname: {name}\n")

    loss_callback = pl.callbacks.ModelCheckpoint(
        dirpath="models/loss_cb_graph",
        filename=name + "-loss-{epoch}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    acc_callback = pl.callbacks.ModelCheckpoint(
        dirpath="models/cp_acc_graph",
        filename=name + "-acc-{epoch}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    acc_macro_callback = pl.callbacks.ModelCheckpoint(
        dirpath="models/cp_acc_graph",
        filename=name + "-accmacro-{epoch}-{val_acc_macro:.4f}",
        monitor="val_acc_macro",
        mode="max",
        save_top_k=1,
    )

    return dm, model, [loss_callback, acc_callback, acc_macro_callback]


def full_training(args: argparse.Namespace) -> None:
    """Train the model on the whole dataset.

    Args:
        args (argparse.Namespace): the arguments from the command line.

    Returns:
        None.

    Raises:
        ValueError: if the model is not implemented.
    """

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
        if args.dropout == 0.0:
            if args.model == "graph_gat":
                name = f"{args.model}-{args.dimh}-{args.num_layers}-{args.x_type}-{args.distance}-{args.heads}-{args.learning_rate}"
            elif args.model == "graph_custom" and not args.custom_wide_connections:
                name = f"{args.model}-{args.dimh}-{args.num_layers}-{args.x_type}-{args.distance}-{args.custom_input_layer}-{args.custom_input_hidden}-{args.custom_output_layer}-{args.custom_output_hidden}-{args.learning_rate}"
            elif args.model == "graph_custom" and args.custom_wide_connections:
                name = f"{args.model}-{args.dimh}-{args.num_layers}-{args.x_type}-{args.distance}-{args.custom_input_layer}-{args.custom_input_hidden}-{args.custom_output_layer}-{args.custom_output_hidden}-wide-{args.learning_rate}"
            else:
                name = f"{args.model}-{args.dimh}-{args.num_layers}-{args.x_type}-{args.distance}-{args.learning_rate}"
        else:
            if args.model == "graph_gat":
                name = f"{args.model}-{args.dimh}-{args.num_layers}-{args.x_type}-{args.distance}-{args.heads}-{args.dropout}-{args.learning_rate}"
            elif args.model == "graph_custom" and not args.custom_wide_connections:
                name = f"{args.model}-{args.dimh}-{args.num_layers}-{args.x_type}-{args.distance}-{args.dropout}-{args.custom_input_layer}-{args.custom_input_hidden}-{args.custom_output_layer}-{args.custom_output_hidden}-{args.learning_rate}"
            elif args.model == "graph_custom" and args.custom_wide_connections:
                name = f"{args.model}-{args.dimh}-{args.num_layers}-{args.x_type}-{args.distance}-{args.dropout}-{args.custom_input_layer}-{args.custom_input_hidden}-{args.custom_output_layer}-{args.custom_output_hidden}-wide-{args.learning_rate}"
            else:
                name = f"{args.model}-{args.dimh}-{args.num_layers}-{args.x_type}-{args.distance}-{args.dropout}-{args.learning_rate}"

        trainer.save_checkpoint(f"models/{name}-{args.max_epochs}.ckpt")

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
        "-cil",
        "--custom_input_layer",
        type=int,
        default=CUSTOM_INPUT_LAYER,
        help="Custom linear input layer number in the custom graph model.",
    )
    parser.add_argument(
        "-cih",
        "--custom_input_hidden",
        type=int,
        default=CUSTOM_INPUT_HIDDEN,
        help="Custom linear input hidden layer size in the custom graph model.",
    )
    parser.add_argument(
        "-col",
        "--custom_output_layer",
        type=int,
        default=CUSTOM_OUTPUT_LAYER,
        help="Custom linear output layer number in the custom graph model.",
    )
    parser.add_argument(
        "-coh",
        "--custom_output_hidden",
        type=int,
        default=CUSTOM_OUTPUT_HIDDEN,
        help="Custom linear output hidden layer size in the custom graph model.",
    )
    parser.add_argument(
        "-cwc",
        "--custom_wide_connections",
        type=bool,
        default=CUSTOM_WIDE_CONNECTIONS,
        help="Custom wide connections in the custom graph model.",
    )
    parser.add_argument(
        "-do",
        "--dropout",
        type=float,
        default=DROPOUT,
        help="Dropout to use for the graph model.",
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
        "heads": args.heads,
        "x_type": args.x_type,
        "data": "shuffle",
        "save_name": args.save_name,
    }
    wandb.init(project="sflizard", entity="leonardfavre", config=wandb_config)

    # Set seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    full_training(args)
