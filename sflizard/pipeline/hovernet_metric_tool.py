import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm

from sflizard import Graph, LizardGraphDataModule

# config of training
TRAIN_DATA_PATH = "data/Lizard_dataset_extraction/data_0.9_split_train.pkl"
VALID_DATA_PATH = "data/Lizard_dataset_extraction/data_0.9_split_valid.pkl"
TEST_DATA_PATH = "data/Lizard_dataset_extraction/data_0.9_split_test.pkl"
SEED = 303
STARDIST_CHECKPOINT = (
    "models/final3_stardist_crop-cosine_200epochs_1.0losspower_0.0005lr.ckpt"
)
CHECKPOINT_PATH = ["models/"]  # , "models/cp_acc_graph/", "models/loss_cb_graph/"]
TRUE_DATA_PATH_START = "data/Lizard_dataset_split/patches/Lizard_Labels_"


class HoverNetMetricTool:
    def __init__(
        self,
        mode: str = "valid",
        weights_selector: dict = {
            "model": [],
            "dimh": [],
            "num_layers": [],
            "heads": [],
        },
        distance: int = 45,
        x_type: str = "ll",
    ) -> None:
        self.mode = mode
        self.device = "cuda"
        self.base_save_path = f"output/graph/{self.mode}/{distance}/{x_type}"
        Path(self.base_save_path).mkdir(parents=True, exist_ok=True)
        self.log_file = Path(self.base_save_path) / "log.txt"
        Path(self.log_file).touch(exist_ok=True)
        self.distance = distance
        self.x_type = x_type

        # create the datamodule
        print("\nLoading data...")
        # get the valid data
        with Path(VALID_DATA_PATH).open("rb") as f:
            valid_data = pickle.load(f)
        with Path(TEST_DATA_PATH).open("rb") as f:
            test_data = pickle.load(f)
        dm = LizardGraphDataModule(
            valid_data=valid_data,
            test_data=test_data,
            batch_size=4,
            num_workers=4,
            seed=SEED,
            stardist_checkpoint=STARDIST_CHECKPOINT,
            x_type=x_type,
            distance=distance,
            light=False,
        )
        dm.setup()

        # get the dataloader
        if mode == "test":
            print("-- test mode")
            self.dataloader = dm.val_dataloader()
        elif mode == "valid":
            print("-- validation mode")
            self.dataloader = dm.train_dataloader()
        print("Data loaded.")

        # get the checkpoints
        print("\nGetting checkpoints...")
        weights_paths = self.get_weights_path(weights_selector)
        print(f"Checkpoints found: {len(weights_paths)}")
        for wp in weights_paths:
            print(f" -- {wp}")
        with self.log_file.open("a") as f:
            f.write(f"{len(weights_paths)} checkpoints found:")
            for wp in weights_paths:
                f.write(f"\n -- {wp}")

        # run the conversion for each model
        print("\nRunning inference and metrics...")
        with self.log_file.open("a") as f:
            f.write("\n\nInference result:")
        for wp in weights_paths:
            print(f"\n -- {wp}...")
            try:
                # load graph model
                graph_model = self.init_graph_inference(weights_paths[wp])
                # run the inference on data
                self.save_mat(graph_model, wp)
                # run the hovernet metric tool
                result = self.run_hovernet_metric_tool(wp)
                # clean the folder
                self.clean_folder(wp)
                print(f"{result}...done.\n")
                with self.log_file.open("a") as f:
                    f.write(f"\n{wp}: {result}")
                # clean the folder

            except RuntimeError as e:
                print(f"RuntimeError... see log in {self.base_save_path}/log.txt")
                # save error to file
                with self.log_file.open("a") as f:
                    f.write(f"\n{wp}: RuntimeError.\n\n{str(e)}\nskiped.")
                print("...skipped.")
            except ValueError as e:
                print(f"ValueError... see log in {self.base_save_path}/log.txt")
                # save error to file
                with self.log_file.open("a") as f:
                    f.write(f"\n{wp}: ValueError.\n\n{str(e)}\nskiped.")
                print("...skipped.")

        print("\nAll done.")

    def init_graph_inference(self, weights_path: str) -> None:
        print("Loading graph model...")
        model = Graph.load_from_checkpoint(
            weights_path,
        )
        graph = model.model.to(self.device)
        print("Graph model loaded.")
        return graph

    def get_weights_path(self, weights_selector: dict) -> dict:
        weights_path = {}
        available_checkpoints = []
        for path in CHECKPOINT_PATH:
            available_checkpoints += list(Path(path).glob("*.ckpt"))
        for model in weights_selector["model"]:
            for dimh in weights_selector["dimh"]:
                for num_layers in weights_selector["num_layers"]:
                    for heads in weights_selector["heads"]:
                        if (
                            model != "graph_gat"
                            and heads != weights_selector["heads"][0]
                        ):
                            continue
                        for checkpoint in available_checkpoints:
                            if (
                                f"{model}-{dimh}-{num_layers}-{self.x_type}-{self.distance}"
                                in str(checkpoint)
                                and model != "graph_gat"
                            ):
                                ckpt = checkpoint
                                wp = f"{model}-{dimh}-{num_layers}"
                            elif (
                                f"{model}-{dimh}-{num_layers}-{self.x_type}-{self.distance}"#-{heads}"
                                in str(checkpoint)
                                and model == "graph_gat"
                            ):
                                ckpt = checkpoint
                                wp = f"{model}-{dimh}-{num_layers}-{heads}"
                            else:
                                continue
                            if "loss" in str(ckpt):
                                weights_path[f"{wp}-loss"] = ckpt
                            elif "acc_macro" in str(ckpt):
                                weights_path[f"{wp}-acc_macro"] = ckpt
                            elif "acc" in str(ckpt):
                                weights_path[f"{wp}-acc"] = ckpt
                            else:
                                weights_path[f"{wp}-final"] = ckpt
        return weights_path

    def save_mat(self, graph_model: torch.nn.Module, save_folder: str) -> None:
        # create a folder for the results
        save_path = self.base_save_path + f"/{save_folder}/"
        Path(save_path).mkdir(parents=True, exist_ok=True)

        loader = iter(self.dataloader)
        # convert data
        for i in tqdm(range(len(loader))):  # type: ignore

            # get next test batch
            batch = next(loader)
            batch = batch.to(self.device)
            for b in range(len(batch)):
                # run inference
                x, edge_index = batch[b].x, batch[b].edge_index
                with torch.no_grad():
                    out = graph_model(x, edge_index)
                    graph_pred = out.argmax(-1)

                # get correct format data
                inst_map = np.squeeze(batch[b].inst_map.astype(np.int32))
                inst_centroid = batch[b].pos.cpu().numpy().astype(np.float64)
                inst_type = np.expand_dims(
                    graph_pred.cpu().numpy().astype(np.int64), axis=1
                )
                inst_uid = np.array(
                    [[inst_map[int(i[0])][int(i[1])]] for i in inst_centroid]
                ).astype(np.int32)
                inst_centroid[:, [1, 0]] = inst_centroid[:, [0, 1]]
                mat = {
                    "inst_map": inst_map,
                    "inst_uid": inst_uid,
                    "inst_type": inst_type,
                    "inst_centroid": inst_centroid,
                }

                # save the results
                sio.savemat(f"{save_path}{batch[b].image_idx}.mat", mat)

    def run_hovernet_metric_tool(self, save_folder: str) -> str:
        save_path = self.base_save_path + f"/{save_folder}/"
        compute_stat_cmd = f"python external_models/hover_net/compute_stats.py --pred_dir {save_path} --true_dir {TRUE_DATA_PATH_START}{self.mode}/ --mode type"
        command = f"conda activate hovernet; {compute_stat_cmd}; conda activate TM"
        ret = subprocess.run(command, capture_output=True, shell=True)
        result = ret.stdout.decode()
        return result

    def clean_folder(self, save_folder: str) -> None:
        save_path = self.base_save_path + f"/{save_folder}/"
        shutil.rmtree(save_path)
