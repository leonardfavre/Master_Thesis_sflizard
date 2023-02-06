import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

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
CHECKPOINT_PATH = ["models/", "models/cp_acc_graph/", "models/loss_cb_graph/"]
TRUE_DATA_PATH_START = "data/Lizard_dataset_split/patches/Lizard_Labels_"

TEST_DROPOUT = True


class HoverNetMetricTool:
    """Tool to evaluate the performance of Graph model on the Lizard dataset using hovernet compute_stats tool."""

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
        x_type: str = "4ll",
        paths: dict = {},
    ) -> None:
        """Tool to evaluate the performance of Graph model on the Lizard dataset using hovernet compute_stats tool.

        Args:
            mode (str): "valid" or "test" depending on the dataset to use.
            weights_selector (dict): dict of list of model, dimh, num_layers and heads to test.
            distance (int): distance used in creation of graph.
            x_type (str): type of node feature vector.
            paths (dict): dict of paths to the model checkpoints to test.

        Returns:
            None.

        Raises:
            None.
        """
        self.mode = mode
        self.device = "cuda"

        if len(paths) == 0:
            quick_run = False
        else:
            quick_run = True

        if not quick_run:
            # create the output directory
            self.base_save_path = f"output/graph/{self.mode}/{distance}/{x_type}"
            Path(self.base_save_path).mkdir(parents=True, exist_ok=True)
            self.log_file = Path(self.base_save_path) / "log.txt"
            Path(self.log_file).touch(exist_ok=True)

            # result table to store results for easier analysis
            self.__init_result_table(weights_selector)
        else:
            self.base_save_path = "output/graph/manual"
            Path(self.base_save_path).mkdir(parents=True, exist_ok=True)

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
            self.dataloader = dm.val_dataloader()  # type: ignore
        elif mode == "valid":
            print("-- validation mode")
            self.dataloader = dm.train_dataloader()  # type: ignore
        print("Data loaded.")

        if not quick_run:
            # get the checkpoints
            print("\nGetting checkpoints...")
            weights_paths = self.__get_weights_path(weights_selector)
            print(f"Checkpoints found: {len(weights_paths)}")
            for wp in weights_paths:
                print(f" -- {wp}")
            with self.log_file.open("a") as f:  # type: ignore
                f.write(f"{len(weights_paths)} checkpoints found:")  # type: ignore
                for wp in weights_paths:
                    f.write(f"\n -- {wp}")  # type: ignore
        else:
            weights_paths = paths

        # run the conversion for each model
        print("\nRunning inference and metrics...")
        if not quick_run:
            with self.log_file.open("a") as f:  # type: ignore
                f.write("\n\nInference result:")  # type: ignore
        for wp in weights_paths:
            print(f"\n -- {wp}...")
            try:
                # load graph model
                graph_model = self.__init_graph_inference(weights_paths[wp])
                # run the inference on data
                self.__save_mat(graph_model, wp)
                # run the hovernet metric tool
                result = self.__run_hovernet_metric_tool(wp)
                print(f"{result}...done.\n")
                if not quick_run:
                    self.__save_result_in_table(wp, result)
                    # clean the folder
                    self.__clean_folder(wp)
                    with self.log_file.open("a") as f:  # type: ignore
                        f.write(f"\n{wp}: {result}")  # type: ignore

            except RuntimeError as e:
                if not quick_run:
                    print(f"RuntimeError... see log in {self.base_save_path}/log.txt")
                    # save error to file
                    with self.log_file.open("a") as f:  # type: ignore
                        f.write(f"\n{wp}: RuntimeError.\n\n{str(e)}\nskiped.")  # type: ignore
                else:
                    print(e)
                print("...skipped.")
            except ValueError as e:
                if not quick_run:
                    print(f"ValueError... see log in {self.base_save_path}/log.txt")
                    # save error to file
                    with self.log_file.open("a") as f:  # type: ignore
                        f.write(f"\n{wp}: ValueError.\n\n{str(e)}\nskiped.")  # type: ignore
                else:
                    print(e)
                print("...skipped.")
            except FileNotFoundError as e:
                if not quick_run:
                    print(
                        f"FileNotFoundError... see log in {self.base_save_path}/log.txt"
                    )
                    # save error to file
                    with self.log_file.open("a") as f:  # type: ignore
                        f.write(f"\n{wp}: FileNotFoundError.\n\n{str(e)}\nskiped.")  # type: ignore
                else:
                    print(e)
                print("...skipped.")

        # save the result table
        if not quick_run:
            self.__save_result_to_file()

        print("\nAll done.")

    def __init_graph_inference(self, weights_path: str) -> torch.nn.Module:
        """Initialize the graph model for inference.

        Args:
            weights_path (str): path to the checkpoint.

        Returns:
            None.

        Raises:
            None.
        """
        print("Loading graph model...")
        model = Graph.load_from_checkpoint(
            weights_path,
            wandb_log=False,
        )
        graph = model.model.to(self.device)
        print("Graph model loaded.")
        return graph

    def __get_weights_path(self, weights_selector: dict) -> dict:
        """Looks for the weights path for each model available and return the one that matches the selection.

        Args:
            weights_selector (dict): dictionary with the model parameters to select.

        Returns:
            weights_path (dict): dictionary with the weights path to test.

        Raises:
            None.
        """

        def save_path(ckpt: Path, wp: str, weights_path: dict) -> dict:
            if "loss" in str(ckpt):
                weights_path[f"{wp}-loss"] = ckpt
            elif "acc_macro" in str(ckpt):
                weights_path[f"{wp}-acc_macro"] = ckpt
            elif "acc" in str(ckpt):
                weights_path[f"{wp}-acc"] = ckpt
            else:
                weights_path[f"{wp}-final"] = ckpt
            return weights_path

        weights_path: Dict[Any, Any] = {}
        available_checkpoints = []
        for path in CHECKPOINT_PATH:
            available_checkpoints += list(Path(path).glob("*.ckpt"))
        for model in weights_selector["model"]:
            for dimh in weights_selector["dimh"]:
                for num_layers in weights_selector["num_layers"]:
                    if model == "graph_gat":
                        for heads in weights_selector["heads"]:
                            if "custom_combinations" in weights_selector:
                                for comb in weights_selector["custom_combinations"]:
                                    for checkpoint in available_checkpoints:
                                        if (
                                            f"{model}-{dimh}-{num_layers}-{self.x_type}-{self.distance}-{heads}-{comb}"
                                            in str(checkpoint)
                                        ):
                                            wp = f"{model}-{dimh}-{num_layers}-{heads}-{comb}"
                                            weights_path = save_path(
                                                checkpoint, wp, weights_path
                                            )
                            else:
                                for checkpoint in available_checkpoints:
                                    if (
                                        f"{model}-{dimh}-{num_layers}-{self.x_type}-{self.distance}-{heads}"
                                        in str(checkpoint)
                                    ):
                                        wp = f"{model}-{dimh}-{num_layers}-{heads}"
                                        weights_path = save_path(
                                            checkpoint, wp, weights_path
                                        )
                    elif "custom_combinations" in weights_selector:
                        for comb in weights_selector["custom_combinations"]:
                            for checkpoint in available_checkpoints:
                                if (
                                    f"{model}-{dimh}-{num_layers}-{self.x_type}-{self.distance}-{comb}"
                                    in str(checkpoint)
                                ):
                                    wp = f"{model}-{dimh}-{num_layers}-{comb}"
                                    weights_path = save_path(
                                        checkpoint, wp, weights_path
                                    )
                    else:
                        for checkpoint in available_checkpoints:
                            if (
                                f"{model}-{dimh}-{num_layers}-{self.x_type}-{self.distance}"
                                in str(checkpoint)
                            ):
                                wp = f"{model}-{dimh}-{num_layers}"
                                weights_path = save_path(checkpoint, wp, weights_path)
        return weights_path

    def __save_mat(self, graph_model: torch.nn.Module, save_folder: str) -> None:
        """Run the inference on the test data and save the results in a .mat file.

        Args:
            graph_model (torch.nn.Module): graph model to use for inference.
            save_folder (str): folder to save the results.

        Returns:
            None.

        Raises:
            None.
        """
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

    def __run_hovernet_metric_tool(self, save_folder: str) -> str:
        """Run the hovernet metric tool to compute the metrics.

        Args:
            save_folder (str): folder to save the results.

        Returns:
            result (str): string with the results metrics.

        Raises:
            None.
        """
        save_path = self.base_save_path + f"/{save_folder}/"
        compute_stat_cmd = f"python external_models/hover_net/compute_stats.py --pred_dir {save_path} --true_dir {TRUE_DATA_PATH_START}{self.mode}/ --mode type"
        command = f"conda activate hovernet; {compute_stat_cmd}; conda activate TM"
        ret = subprocess.run(command, capture_output=True, shell=True)  # nosec
        result = ret.stdout.decode()
        return result

    def __clean_folder(self, save_folder: str) -> None:
        """Clean the folder with the results to save disk space.

        Args:
            save_folder (str): folder to save the results.

        Returns:
            None.

        Raises:
            None.
        """
        save_path = self.base_save_path + f"/{save_folder}/"
        shutil.rmtree(save_path)

    def __init_result_table(self, weights_selector: dict) -> None:
        """Initialize the result table to save the results.

        Args:
            weights_selector (dict): dictionary with the weights to use.

        Returns:
            None.

        Raises:
            None.
        """
        self.result_table: Dict[Any, Any] = {}
        for model in weights_selector["model"]:
            self.result_table[model] = {}
            for ckpt in ["final", "acc", "acc_macro", "loss"]:
                self.result_table[model][ckpt] = {}
                if "custom_combinations" in weights_selector:
                    for c in weights_selector["custom_combinations"]:
                        if model == "graph_gat":
                            for head in weights_selector["heads"]:
                                self.result_table[model][ckpt][c][head] = {}
                                for dh in weights_selector["dimh"]:
                                    self.result_table[model][ckpt][c][head][dh] = {}
                        else:
                            self.result_table[model][ckpt][c] = {}
                            for dh in weights_selector["dimh"]:
                                self.result_table[model][ckpt][c][dh] = {}
                elif model == "graph_gat":
                    for head in weights_selector["heads"]:
                        self.result_table[model][ckpt][head] = {}
                        for dh in weights_selector["dimh"]:
                            self.result_table[model][ckpt][head][dh] = {}
                else:
                    for dh in weights_selector["dimh"]:
                        self.result_table[model][ckpt][dh] = {}
        self.result_file = Path(self.base_save_path) / "result_table.pkl"
        Path(self.result_file).touch(exist_ok=True)

    def __save_result_in_table(self, save_folder: str, result: str) -> None:
        """Save the result of a model in the result table.

        Args:
            save_folder (str): folder to save the results.
            result (str): string with the results metrics.

        Returns:
            None.

        Raises:
            None.
        """
        # save result in result table
        selector = save_folder.split("-")
        model = selector[0]
        dimh = int(selector[1])
        num_layers = int(selector[2])
        if model == "graph_custom":
            if "wide" in selector:
                if TEST_DROPOUT:
                    combination = "-".join(selector[4:9])
                    ckpt = selector[9]
                else:
                    combination = "-".join(selector[3:8])
                    ckpt = selector[8]
            else:
                combination = "-".join(selector[3:7])
                ckpt = selector[7]
            self.result_table[model][ckpt][combination][dimh][num_layers] = result
        elif model == "graph_sage" and TEST_DROPOUT:
            combination = selector[3]
            ckpt = selector[4]
            self.result_table[model][ckpt][combination][dimh][num_layers] = result
        elif model == "graph_gat":
            head = int(selector[3])
            if TEST_DROPOUT:
                combination = selector[4]
                ckpt = selector[5]
                self.result_table[model][ckpt][combination][head][dimh][
                    num_layers
                ] = result
            else:
                ckpt = selector[4]
                self.result_table[model][ckpt][head][dimh][num_layers] = result
        else:
            ckpt = selector[3]
            self.result_table[model][ckpt][dimh][num_layers] = result

    def __save_result_to_file(self) -> None:
        """Save the result table to a file in a good format to use later.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        with open(self.result_file, "a") as f:
            f.write("{\n")
            for m in self.result_table:
                # f.write(f"-------------------\n# {m}\n")
                for c in self.result_table[m]:
                    if m == "graph_gat" or m == "graph_custom":
                        for h in self.result_table[m][c]:
                            f.write(f'   "{m}_{c}_{h}": [\n')
                            for dh in self.result_table[m][c][h]:
                                f.write(f"      [ # {dh}\n")
                                for nl in self.result_table[m][c][h][dh]:
                                    val = self.result_table[m][c][h][dh][nl].split(
                                        "\n"
                                    )[1]
                                    val = val.replace("  ", ", ")
                                    f.write(f"         {val}, # {nl}\n")
                                f.write("      ],\n")
                            f.write("   ],\n")
                    else:
                        f.write(f'   "{m}_{c}": [\n')
                        for dh in self.result_table[m][c]:
                            f.write(f"      [ # {dh}\n")
                            for nl in self.result_table[m][c][dh]:
                                val = self.result_table[m][c][dh][nl].split("\n")[1]
                                val = val.replace("  ", ", ")
                                f.write(f"         {val}, # {nl}\n")
                            f.write("      ],\n")
                        f.write("   ],\n")
            f.write("}\n")
