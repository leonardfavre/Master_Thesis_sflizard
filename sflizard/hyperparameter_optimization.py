import os
import time
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from sflizard import Graph, LizardGraphDataModule


def train_from_config(config, checkpoint_dir=None):
    """Perform a training."""
    config = Namespace(**config)

    print("############################ setting up dm")
    dm = LizardGraphDataModule(
        data_path=config.data_path,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    dm.setup()
    print("done setting up dm")

    metrics = {"loss": "val_loss", "acc": "val_acc"}
    callbacks = [
        TuneReportCheckpointCallback(
            metrics, on="validation_end", filename="checkpoint"
        )
    ]
    model = Graph(
        model=config.model_name,
        learning_rate=config.learning_rate,
        num_features=32,
        num_classes=7,
        seed=config.seed,
        max_epochs=config.max_epochs,
        dim_h=config.dim_h,
        num_layers=config.num_layers,
    )

    trainer = pl.Trainer.from_argparse_args(
        config,
        callbacks=callbacks,
        max_epochs=config.max_epochs,
        gpus=1,
        # enable_progress_bar=False,
    )

    # Train and testing

    trainer.fit(model, dm)


if __name__ == "__main__":

    config = {
        "model_name": tune.choice(
            [
                "graph_gatv2",
                "graph_gat",
                "graph_rgcn",
                "graph_rgcn_fast",
                "graph_sage",
            ]
        ),
        "learning_rate": tune.loguniform(1e-6, 5e-1),
        "batch_size": 16,
        "scheduler": tune.choice(["cosine", "step", "lambda"]),
        "data_path": None,
        "num_workers": 8,
        "seed": 303,
        "default-root-dir": os.getcwd(),
        "max_epochs": 20,
        "dim_h": tune.choice([8, 16, 32, 64]),
        "num_layers": tune.choice([0, 1, 2, 4]),
    }
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    config["data_path"] = args.data_path

    analysis = tune.run(
        train_from_config,
        metric="loss",
        mode="min",
        config=config,
        num_samples=args.num_samples,
        name="lizardGraph_hyperparameter_optim",
        resources_per_trial={"cpu": 16, "gpu": 1},
    )
    analysis.trial_dataframes.to_csv(
        f"lizardGraph_hyperparameter_optim{str(time.asctime()).replace('' ,'_').replace(':','_')}.csv"
    )
    print(f"The best model was saved in {analysis.get_best_checkpoint()}")
