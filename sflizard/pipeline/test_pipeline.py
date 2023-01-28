import copy
from typing import List, Union

import numpy as np
import torch
from rich.console import Console
from tqdm import tqdm

from sflizard import (
    Graph,
    LizardDataModule,
    ReportGenerator,
    SegmentationMetricTool,
    Stardist,
    get_class_map_from_graph,
    get_graph_for_inference,
    improve_class_map,
    merge_stardist_class_together,
    rotate_and_pred,
)

X_TYPE = {
    128: "ll",
    135: "ll+c",
    137: "ll+c+x",
    512: "4ll",
    540: "4ll+c",
    548: "4ll+c+x",
}
STAR_4_IMPROVEMENT = False


class TestPipeline:
    """A pipeline to test the model."""

    def __init__(
        self,
        valid_data_path: str,
        test_data_path: str,
        stardist_weights_path: str,
        graph_weights_path: List[str],
        graph_distance: int,
        n_rays: int,
        n_classes: int,
        batch_size: int,
        seed: int,
        mode: str,
    ) -> None:
        """Init the pipeline.

        Args:
            valid_data_path (str): The path to the valid data.
            test_data_path (str): The path to the test data.
            stardist_weights_path (str): The path to the stardist weights.
            graph_weights_path (List[str]): The path to the graph weights.
            graph_distance (int): The distance to use for the graph.
            n_rays (int): The number of rays to use for stardist.
            n_classes (int): The number of classes.
            batch_size (int): The batch size to use.
            seed (int): The seed to use.
            mode (str): The mode to use (test or valid).

        Returns:
            None.

        Raises:
            None.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device:", self.device)
        self.n_classes = n_classes
        self.n_rays = n_rays
        self.__init_dataloader(valid_data_path, test_data_path, seed, batch_size, mode)
        self.__init_stardist_inference(stardist_weights_path)

        # graph initialization
        self.compute_graph = graph_weights_path is not None
        if self.compute_graph:
            self.__init_graph_inference(graph_weights_path)
        self.graph_distance = graph_distance

    def __init_dataloader(
        self,
        valid_data_path: str,
        test_data_path: str,
        seed: int = 303,
        batch_size: int = 1,
        mode: str = "valid",
    ) -> None:
        """Init the dataloader.

        Args:
            valid_data_path (str): The path to the valid data.
            test_data_path (str): The path to the test data.
            seed (int): The seed to use.
            batch_size (int): The batch size to use.
            mode (str): The mode to use (test or valid).

        Returns:
            None.

        Raises:
            None.
        """
        aditional_args = {"n_rays": self.n_rays}

        print("Loading data...")
        # create the datamodule
        dm = LizardDataModule(
            train_data_path=None,
            valid_data_path=valid_data_path,
            test_data_path=test_data_path,
            batch_size=batch_size,
            annotation_target="stardist" if self.n_classes == 1 else "stardist_class",
            seed=seed,
            aditional_args=aditional_args,
        )
        dm.setup()

        # get the dataloader
        if mode == "test":
            print("test mode")
            self.dataloader = iter(dm.test_dataloader())
        elif mode == "valid":
            print("validation mode")
            self.dataloader = iter(dm.val_dataloader())
        print("Data loaded.")

    def __init_stardist_inference(
        self,
        weights_path: str,
    ) -> None:
        """Init the stardist model for inference.

        Args:
            weights_path (str): The path to the stardist weights.

        Returns:
            None.

        Raises:
            None.
        """
        print("Loading stardist model...")
        model = Stardist.load_from_checkpoint(
            weights_path,
            wandb_log=False,
        )
        self.stardist_weights_path = weights_path

        self.stardist = model.model.to(self.device)
        self.stardist_layer = copy.deepcopy(model.model).to(self.device)
        self.stardist_layer.output_last_layer = True
        print("Stardist model loaded.")
        self.classification = self.n_classes > 1

    def __init_graph_inference(self, weights_path: Union[List[str], str]) -> None:
        """Init the graph model for inference.

        Args:
            weights_path (List[str]): The path to the graph weights.

        Returns:
            None.

        Raises:
            None.
        """
        print("Loading graph model...")
        if isinstance(weights_path, str):
            weights_path = [weights_path]
        self.graph = {}
        for w in weights_path:
            model = Graph.load_from_checkpoint(
                w,
            )
            self.x_type = X_TYPE[model.num_features]
            self.graph[w] = model.model.to(self.device)
        print("Graph model loaded.")

    def test(self, output_dir=None, imgs_to_display=0) -> None:
        """Run the pipeline and test the model.

        Args:
            output_dir (str): The path to the output directory for report and images.
            imgs_to_display (int): The number of images to display in the report.

        Returns:
            None.

        Raises:
            None.
        """
        print("Testing...")

        # init metric tool
        self.star_smt = SegmentationMetricTool(self.n_classes, self.device)

        if self.compute_graph:
            self.graph_smt = {}
            for g in self.graph:
                self.graph_smt[g] = SegmentationMetricTool(self.n_classes, self.device)

        # init report tool
        if output_dir is not None:
            self.report_generator = ReportGenerator(
                output_dir, imgs_to_display, self.n_classes
            )

        for i in tqdm(range(len(self.dataloader))):  # type: ignore

            # if i > 3:
            #     break

            # get next test batch
            batch = next(self.dataloader)
            for b in range(len(batch)):
                batch[b] = batch[b].to(self.device)
            if self.classification:
                inputs, obj_probabilities, distances, classes = batch
                true_class_map = classes.int()
            else:
                inputs, obj_probabilities, distances = batch

            # orignal images
            images = inputs

            # stardist true mask
            true_mask = self.stardist.compute_star_label(
                inputs, distances, obj_probabilities
            )

            # stardist predicted mask
            with torch.no_grad():
                if self.classification:
                    dist, prob, clas = self.stardist(inputs)
                else:
                    dist, prob = self.stardist(inputs)
            pred_mask = self.stardist.compute_star_label(inputs, dist, prob)

            # 4 times stardist to improve mask
            if STAR_4_IMPROVEMENT:
                # rotate input
                pred_90_mask, clas_90 = rotate_and_pred(self.stardist, inputs, 1)
                pred_180_mask, clas_180 = rotate_and_pred(self.stardist, inputs, 2)
                pred_270_mask, clas_270 = rotate_and_pred(self.stardist, inputs, 3)

            # stardist class mask
            if self.classification:

                def get_class_pred(clas, pred_mask):
                    pred_class_map = torch.argmax(clas, dim=1)
                    pred_class_map_improved = torch.Tensor(
                        np.array(
                            [
                                improve_class_map(pred_class_map[b].cpu(), pred_mask[b])
                                for b in range(len(pred_class_map))
                            ]
                        )
                    )
                    return pred_class_map_improved, pred_class_map

                pred_class_map_improved, pred_class_map = get_class_pred(
                    clas, pred_mask
                )
                if STAR_4_IMPROVEMENT:
                    class_pred_90, _ = get_class_pred(clas_90, pred_90_mask)
                    class_pred_180, _ = get_class_pred(clas_180, pred_180_mask)
                    class_pred_270, _ = get_class_pred(clas_270, pred_270_mask)
                    pred_class_map_improved = merge_stardist_class_together(
                        pred_class_map_improved,
                        class_pred_90,
                        class_pred_180,
                        class_pred_270,
                    )
            else:
                pred_class_map_improved = None

            if self.compute_graph:

                # graph predicted mask
                graphs = get_graph_for_inference(
                    inputs, self.graph_distance, self.stardist_weights_path, self.x_type
                )

                graphs_class_map = {}

                with torch.no_grad():
                    for g in self.graph:
                        graph_pred = []
                        for j in range(len(graphs)):
                            if len(graphs[j]["x"]) > 0:
                                out = self.graph[g](
                                    graphs[j]["x"].to(self.device),
                                    graphs[j]["edge_index"].to(self.device),
                                )
                                graph_pred.append(out.argmax(-1))
                            else:
                                graph_pred.append(None)

                        graphs_class_map[g] = get_class_map_from_graph(
                            graphs, pred_mask, graph_pred, pred_class_map_improved
                        )

            # metrics gattering
            self.star_smt.add_batch(i, true_mask, pred_mask)

            if self.classification:
                self.star_smt.add_batch_class(
                    i,
                    true_class_map.detach().int().cpu().numpy(),
                    pred_class_map_improved.detach().int().cpu().numpy(),
                )
                if self.compute_graph:
                    for g in self.graph_smt:
                        self.graph_smt[g].add_batch_class(
                            i,
                            true_class_map.detach().int().cpu().numpy(),
                            np.array(graphs_class_map[g]),
                        )

            if output_dir is not None:
                self.report_generator.add_batch(
                    images,
                    true_mask,
                    pred_mask,
                    true_class_map,
                    pred_class_map,
                    pred_class_map_improved,
                    graphs,
                    graphs_class_map[list(graphs_class_map.keys())[0]],
                )

        # compute metrics
        self.star_smt.compute_metrics()
        if self.compute_graph:
            for g in self.graph_smt:
                self.graph_smt[g].compute_metrics()

        print("Test done.\n\nResults:\n")

        # log
        console = Console()

        self.star_smt.log_results(console)
        if self.compute_graph:
            print("\nGraph results:\n")
            for g in self.graph_smt:
                self.graph_smt[g].log_results(console)

        # generate report
        if output_dir is not None:
            self.report_generator.add_final_metrics(
                self.star_smt.seg_metrics[0],
                self.star_smt.classes_metrics if self.classification else None,
                self.graph_smt[list(self.graph_smt.keys())[0]].classes_metrics
                if self.compute_graph
                else None,
                self.star_smt.seg_metrics if self.classification else None,
                self.graph_smt[list(self.graph_smt.keys())[0]].seg_metrics
                if self.compute_graph
                else None,
            )

            self.report_generator.generate_md()
