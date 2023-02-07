import copy
from typing import List

import numpy as np
import torch
import torchmetrics
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
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
    log_confmat,
)

X_TYPE = {
    128: "ll",
    135: "ll+c",
    137: "ll+c+x",
    512: "4ll",
    540: "4ll+c",
    548: "4ll+c+x",
}


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
            seed (int): The seed to use for constant randomization.
            mode (str): The mode to use (test or valid).

        Returns:
            None.

        Raises:
            None.
        """
        # log
        self.console = Console()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.n_classes = n_classes
        self.n_rays = n_rays
        self.stardist_weights_path = stardist_weights_path
        self.graph_weights_path = graph_weights_path
        if isinstance(self.graph_weights_path, str):
            self.graph_weights_path = [self.graph_weights_path]
        self.graph_distance = graph_distance

        self.__log_config(mode, seed)
        self.__init_dataloader(valid_data_path, test_data_path, seed, batch_size, mode)
        self.__init_stardist_inference()

        # graph initialization
        self.compute_graph = self.graph_weights_path is not None
        if self.compute_graph:
            self.__init_graph_inference()

    def __log_config(self, mode: str, seed: int) -> None:
        """Log the config of the pipeline.

        Args:
            mode (str): The mode to use (test or valid).
            seed (int): The seed to use for constant randomization.

        Returns:
            None.

        Raises:
            None.
        """
        config = Table(title="Config")
        config.add_column("Arg", justify="right", style="cyan", no_wrap=True)
        config.add_column("Value", style="magenta")
        config.add_row("Mode", mode)
        config.add_row("Device", self.device)
        config.add_row("N classes", str(self.n_classes))
        config.add_row("Stardist ckpt", self.stardist_weights_path)
        config.add_row("N rays", str(self.n_rays))
        if self.graph_weights_path is not None:
            for i in range(len(self.graph_weights_path)):
                config.add_row(f"Graph ckpt {i}", self.graph_weights_path[i])
            config.add_row("Graph distance", str(self.graph_distance))
        config.add_row("Seed", str(seed))

        self.console.print(config)

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

        self.console.print("\n\n[bold cyan]### Loading data...[/bold cyan]")
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
            self.dataloader = iter(dm.test_dataloader())
        elif mode == "valid":
            self.dataloader = iter(dm.val_dataloader())
        self.console.print("Data loaded.")

    def __init_stardist_inference(
        self,
    ) -> None:
        """Init the stardist model for inference.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        self.console.print("\n\n[bold cyan]### Loading stardist model... [/bold cyan]")
        model = Stardist.load_from_checkpoint(
            self.stardist_weights_path,
            wandb_log=False,
        )
        self.stardist = model.model.to(self.device)
        self.stardist_layer = copy.deepcopy(model.model).to(self.device)
        self.stardist_layer.output_last_layer = True
        self.console.print("Stardist model loaded.")
        self.classification = self.n_classes > 1
        self.star_confmat = torchmetrics.classification.MulticlassConfusionMatrix(
            num_classes=self.n_classes
        )
        self.star_confmat_norm = torchmetrics.classification.MulticlassConfusionMatrix(
            num_classes=self.n_classes, normalize="true"
        )

    def __init_graph_inference(self) -> None:
        """Init the graph model for inference.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        self.console.print("\n\n[bold cyan]### Loading graph model...[/bold cyan]")

        self.graph = {}
        self.graph_confmat = {}
        self.graph_confmat_norm = {}
        for w in self.graph_weights_path:
            model = Graph.load_from_checkpoint(
                w,
                wandb_log=False,
            )
            self.x_type = X_TYPE[model.num_features]
            self.graph[w] = model.model.to(self.device)
            self.graph_confmat[
                w
            ] = torchmetrics.classification.MulticlassConfusionMatrix(
                num_classes=self.n_classes
            )
            self.graph_confmat_norm[
                w
            ] = torchmetrics.classification.MulticlassConfusionMatrix(
                num_classes=self.n_classes, normalize="true"
            )
        self.console.print("Graph model loaded.")

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
        self.console.print("\n\n[bold cyan]### Testing...[/bold cyan]")

        # init metric tool
        self.star_smt = SegmentationMetricTool(
            self.n_classes, self.device, self.console
        )

        if self.compute_graph:
            self.graph_smt = {}
            for g in self.graph:
                self.graph_smt[g] = SegmentationMetricTool(
                    self.n_classes, self.device, self.console
                )

        # init report tool
        if output_dir is not None:
            self.report_generator = ReportGenerator(
                output_dir, imgs_to_display, self.n_classes, self.console
            )

        for i in tqdm(range(len(self.dataloader))):  # type: ignore

            # get next test batch
            batch = next(self.dataloader)
            for b in range(len(batch)):
                batch[b] = batch[b].to(self.device)
            if self.classification:
                inputs, obj_probabilities, distances, classes = batch
                true_class_map = classes.detach().cpu().numpy().astype("int32")
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
            pred_mask, points = self.stardist.compute_star_label(
                inputs, dist, prob, get_points=True
            )

            # stardist class mask
            if self.classification:

                def get_class_pred(clas, pred_mask):
                    pred_class_map = (
                        clas.argmax(1).detach().cpu().numpy().astype("int32")
                    )

                    pred_class_map_improved = np.array(
                        [
                            improve_class_map(
                                pred_class_map[b], pred_mask[b], points[b]
                            )
                            for b in range(len(pred_class_map))
                        ]
                    )
                    pred_class_map_improved = pred_class_map_improved.astype("int32")
                    return pred_class_map_improved, pred_class_map

                pred_class_map_improved, pred_class_map = get_class_pred(
                    clas, pred_mask
                )
            else:
                pred_class_map_improved = None

            graphs = None
            graphs_class_map = {}
            if self.compute_graph:

                # graph predicted mask
                graphs = get_graph_for_inference(
                    inputs, self.graph_distance, self.stardist_weights_path, self.x_type
                )

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
                                self.console.print("empty graph")
                                graph_pred.append(None)
                        graphs_class_map[g] = get_class_map_from_graph(
                            graphs, pred_mask, graph_pred, pred_class_map_improved
                        )

            # get per-cell class list for confusion matrix
            if self.classification:
                for b in range(len(pred_class_map_improved)):
                    star_pred_b = torch.Tensor(
                        [pred_class_map_improved[b][p[0]][p[1]] for p in points[b]]
                    )
                    true_b = torch.Tensor(
                        [true_class_map[b][p[0]][p[1]] for p in points[b]]
                    )
                    self.star_confmat(star_pred_b, true_b)
                    self.star_confmat_norm(star_pred_b, true_b)
                    for g in self.graph:
                        graph_pred_b = torch.Tensor(
                            [graphs_class_map[g][b][p[0]][p[1]] for p in points[b]]
                        )
                        self.graph_confmat[g](graph_pred_b, true_b)
                        self.graph_confmat_norm[g](graph_pred_b, true_b)

            # metrics gattering
            self.star_smt.add_batch(i, true_mask, pred_mask)

            if self.classification:
                self.star_smt.add_batch_class(
                    i,
                    true_class_map,
                    pred_class_map_improved,
                )
                if self.compute_graph:
                    for g in self.graph_smt:
                        self.graph_smt[g].add_batch(i, true_mask, pred_mask)
                        self.graph_smt[g].add_batch_class(
                            i,
                            true_class_map,
                            graphs_class_map[g],
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
                    graphs_class_map[list(graphs_class_map.keys())[0]]  # type: ignore
                    if self.compute_graph
                    else None,
                )

        print(self.star_confmat.compute())

        # compute metrics
        self.console.print("Computing metrics...")
        self.star_smt.compute_metrics()
        if self.compute_graph:
            for g in self.graph_smt:
                self.graph_smt[g].compute_metrics()

        self.console.print("Test done.\n\nResults:\n")

        self.console.print(Markdown("\n ## Stardist results:\n"))
        self.star_smt.log_results()
        log_confmat(
            self.star_confmat.compute(),
            "Stardist classification confusion matrix",
            self.console,
        )
        log_confmat(
            self.star_confmat_norm.compute(),
            "Stardist classification confusion matrix (normalized over targets)",
            self.console,
        )
        if self.compute_graph:
            self.console.print(Markdown("\n## Graph results:\n"))
            for g in self.graph_smt:
                self.console.print(Markdown(f"\n### {g} results:\n"))
                self.graph_smt[g].log_results()
                log_confmat(
                    self.graph_confmat[g].compute(),
                    "Graph classification confusion matrix",
                    self.console,
                )
                log_confmat(
                    self.graph_confmat_norm[g].compute(),
                    "Graph classification confusion matrix (normalized over targets)",
                    self.console,
                )

        # generate report
        if output_dir is not None:
            self.console.print("\n\n[bold cyan]### Generating report...[/bold cyan]")
            self.report_generator.add_final_metrics(
                self.star_smt.seg_metrics[0],
                self.star_smt.seg_metrics if self.classification else None,
                self.graph_smt[list(self.graph_smt.keys())[0]].seg_metrics
                if self.compute_graph
                else None,
                self.star_confmat.compute(),
                self.star_confmat_norm.compute(),
                self.graph_confmat[list(self.graph_confmat.keys())[0]].compute()
                if self.compute_graph
                else None,
                self.graph_confmat_norm[
                    list(self.graph_confmat_norm.keys())[0]
                ].compute()
                if self.compute_graph
                else None,
            )

            self.report_generator.generate_md()
            self.console.print("Report generated.")
