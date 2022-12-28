import os
import time
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from sflizard import Graph, LizardGraphDataModule, init_stardist_training


def train_graph_from_config(config, checkpoint_dir=None):
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


def train_stardist_from_config(config, checkpoint_dir=None):
    """Perform a training."""
    config = Namespace(**config)

    dm, model = init_stardist_training(config, "cuda", debug=True)

    # metrics = {"loss": "val_loss"}
    # callbacks = [
    #     TuneReportCheckpointCallback(
    #         metrics, on="validation_end", filename="checkpoint"
    #     )
    # ]

    print("############################ setting up trainer")

    # args = {
    #     "logger": True,
    #     "enable_checkpointing": True,
    #     "default_root_dir": None,
    #     "gradient_clip_val": None,
    #     "gradient_clip_algorithm": None,
    #     "num_nodes": 1,
    #     "num_processes": None,
    #     "devices": "1",
    #     "gpus": None,
    #     "auto_select_gpus": False,
    #     "tpu_cores": None,
    #     "ipus": None,
    #     "enable_progress_bar": True,
    #     "overfit_batches": 0.0,
    #     "track_grad_norm": -1,
    #     "check_val_every_n_epoch": 1,
    #     "fast_dev_run": False,
    #     "accumulate_grad_batches": None,
    #     "max_epochs": 1,
    #     "min_epochs": None,
    #     "max_steps": -1,
    #     "min_steps": None,
    #     "max_time": None,
    #     "limit_train_batches": None,
    #     "limit_val_batches": None,
    #     "limit_test_batches": None,
    #     "limit_predict_batches": None,
    #     "val_check_interval": None,
    #     "log_every_n_steps": 50,
    #     "accelerator": "gpu",
    #     "strategy": None,
    #     "sync_batchnorm": False,
    #     "precision": 32,
    #     "enable_model_summary": True,
    #     "weights_save_path": None,
    #     "num_sanity_val_steps": 2,
    #     "resume_from_checkpoint": None,
    #     "profiler": None,
    #     "benchmark": None,
    #     "deterministic": None,
    #     "reload_dataloaders_every_n_epochs": 0,
    #     "auto_lr_find": False,
    #     "replace_sampler_ddp": True,
    #     "detect_anomaly": False,
    #     "auto_scale_batch_size": False,
    #     "plugins": None,
    #     "amp_backend": "native",
    #     "amp_level": None,
    #     "move_metrics_to_cpu": False,
    #     "multiple_trainloader_mode": "max_size_cycle",
    #     "data_path": "/home/leo/projects/Master_Thesis_LeonardFavre/data_540_200_v3.pkl",
    #     "model": "stardist_class",
    #     "batch_size": 4,
    #     "num_workers": 16,
    #     "input_size": 540,
    #     "learning_rate": 0.0001,
    #     "seed": 303,
    #     "dimh": 32,
    #     "num_layers": 1,
    #     "heads": 8,
    # }
    # args = Namespace(**args)
    trainer = pl.Trainer.from_argparse_args(
        config,
        # # callbacks=callbacks,
        # # max_epochs=config.max_epochs,
        # # gpus=1,
        # accelerator='gpu',
        # devices=1,
        # enable_progress_bar=True,
    )

    print("############################ training")

    # Train and testing
    trainer.fit(model, dm)


if __name__ == "__main__":

    graph_config = {
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
        "num_workers": 1,
        "seed": 303,
        "default-root-dir": os.getcwd(),
        "max_epochs": 20,
        "dim_h": tune.choice([8, 16, 32, 64]),
        "num_layers": tune.choice([0, 1, 2, 4]),
    }
    stardist_config = {
        "model": "stardist_class",
        "learning_rate": tune.loguniform(1e-6, 5e-1),
        "batch_size": 4,
        # "scheduler": tune.choice(["cosine", "step", "lambda"]),
        "data_path": None,
        "num_workers": 1,
        "input_size": 540,
        "seed": 303,
        "default-root-dir": os.getcwd(),
        "max_epochs": 1,
        "accelerator": "gpu",
        "devices": 1,
    }
    parser = ArgumentParser()
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    stardist_config["data_path"] = args.data_path
    graph_config["data_path"] = args.data_path

    target = args.target

    analysis = tune.run(
        train_stardist_from_config if target == "stardist" else train_graph_from_config,
        metric="loss",
        mode="min",
        config=stardist_config if target == "stardist" else graph_config,
        num_samples=args.num_samples,
        name=f"lizard_{target}_hyperparameter_optim",
        resources_per_trial={"cpu": 1, "gpu": 1},
        verbose=0,
    )
    analysis.trial_dataframes.to_csv(
        f"lizard_{target}_hyperparameter_optim{str(time.asctime()).replace('' ,'_').replace(':','_')}.csv"
    )
    print(f"The best model was saved in {analysis.get_best_checkpoint()}")
