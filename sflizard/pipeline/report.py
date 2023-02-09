from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torchvision.transforms as T
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import ImageDraw
from rich.console import Console
from stardist.matching import matching
from tqdm import tqdm

from sflizard import get_class_color, get_class_name


class ReportGenerator:
    """MD report generator."""

    def __init__(
        self,
        output_dir: str,
        imgs_to_display: int,
        n_classes: int,
        console: Console,
    ) -> None:
        """Report generator.

        Args:
            output_dir (str): The output directory.
            imgs_to_display (int): The number of images to display.
            n_classes (int): The number of classes.
            console (Console): The console.

        Returns:
            None.

        Raises:
            None.
        """
        self.console = console
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if self.output_dir[-1] != "/":
            self.output_dir += "/"

        self.imgs_to_display = imgs_to_display

        self.n_classes = n_classes
        self.class_name = get_class_name()
        self.class_color = get_class_color()

        # images and masks
        self.images: List[Any] = []
        self.true_masks: List[Any] = []
        self.predicted_masks: List[Any] = []

        self.true_class_map: List[Any] = []
        self.pred_class_map: List[Any] = []
        self.pred_class_map_improved: List[Any] = []

        self.graphs_images: List[Any] = []
        self.graphs_class_map: List[Any] = []

        # metrics
        self.single_segmentation_metric: List[Any] = []
        self.segmentation_metric: Union[Dict[Any, Any], None] = None
        self.segmentation_classification_metric: Union[Dict[Any, Any], None] = None
        self.graph_segmentation_classification_metric: Union[
            Dict[Any, Any], None
        ] = None

    def add_batch(
        self,
        images: list,
        true_masks: Union[np.ndarray, list],
        pred_masks: Union[np.ndarray, list],
        true_class_map: Union[np.ndarray, list, None] = None,
        pred_class_map: Union[np.ndarray, list, None] = None,
        pred_class_map_improved: Union[np.ndarray, list, None] = None,
        graphs: list = None,
        graphs_class_map: list = None,
    ) -> None:
        """Add a batch to the report.

        Args:
            images (list): The images.
            true_masks (list): The true masks.
            pred_masks (list): The predicted masks.
            true_class_map (list): The true class map.
            pred_class_map (list): The predicted class map.
            pred_class_map_improved (list): The improved predicted class map.
            graphs (list): The graphs.
            graphs_class_map (list): The graphs class map.

        Returns:
            None.

        Raises:
            None.
        """
        if self.imgs_to_display > 0:
            for i in range(min(self.imgs_to_display, len(images))):
                # save images and masks
                self.images.append(images[i].cpu())
                self.true_masks.append(true_masks[i])
                self.predicted_masks.append(pred_masks[i])
                self.single_segmentation_metric.append(
                    matching(true_masks[i], pred_masks[i])
                )
                if (
                    true_class_map is not None
                    and pred_class_map is not None
                    and pred_class_map_improved is not None
                ):
                    self.true_class_map.append(true_class_map[i])
                    self.pred_class_map.append(pred_class_map[i])
                    self.pred_class_map_improved.append(pred_class_map_improved[i])
                    if graphs is not None and graphs_class_map is not None:
                        self.graphs_images.append(
                            self._draw_graph(graphs[i], pred_class_map_improved[i])
                        )
                        self.graphs_class_map.append(graphs_class_map[i])
            self.imgs_to_display = max(0, self.imgs_to_display - len(images))

    def add_final_metrics(
        self,
        segmentation_metric: Union[Dict[Any, Any], None],
        segmentation_classification_metric: Union[Dict[Any, Any], None],
        graph_segmentation_classification_metric: Union[Dict[Any, Any], None],
        star_confmat: torch.Tensor,
        star_confmat_norm: torch.Tensor,
        graph_confmat: Union[torch.Tensor, None],
        graph_confmat_norm: Union[torch.Tensor, None],
    ) -> None:
        """Add final metrics to the report.

        Args:
            segmentation_metric (dict): The segmentation metric.
            segmentation_classification_metric (dict): The segmentation classification metric.
            graph_segmentation_classification_metric (dict): The graph segmentation classification metric.
            star_confmat (torch.Tensor): The stardist confusion matrix.
            star_confmat_norm (torch.Tensor): The normalized stardist confusion matrix.
            graph_confmat (torch.Tensor): The graph confusion matrix.
            graph_confmat_norm (torch.Tensor): The normalized graph confusion matrix.

        Returns:
            None.

        Raises:
            None.
        """
        self.segmentation_metric = segmentation_metric
        self.segmentation_classification_metric = segmentation_classification_metric
        self.graph_segmentation_classification_metric = (
            graph_segmentation_classification_metric
        )
        self.star_confmat = star_confmat
        self.star_confmat_norm = star_confmat_norm
        self.graph_confmat = graph_confmat
        self.graph_confmat_norm = graph_confmat_norm

    def generate_md(self) -> None:
        """Generate a markdown file with the report.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        self.console.print("Plotting...")

        # create subdirectories
        Path(self.output_dir + "/images").mkdir(parents=True, exist_ok=True)

        # generate images
        self._generate_images()

        # create MD report containing images and metrics
        md = "# Test results\n\n"

        # add metrics
        if self.segmentation_metric is not None:
            md = self._get_simple_metric_table(
                "Segmentation Metrics", self.segmentation_metric, md
            )
        if self.segmentation_classification_metric is not None:
            md = self._get_per_class_metric_table(
                "Segmentation Metrics per class",
                self.segmentation_classification_metric,
                md,
            )
        if self.star_confmat is not None and self.star_confmat_norm is not None:
            md = self._get_confusion_matrix(
                "Confusion Matrix", self.star_confmat, md, False
            )
            md = self._get_confusion_matrix(
                "Confusion Matrix normalized", self.star_confmat_norm, md, True
            )

        if self.graph_segmentation_classification_metric is not None:
            md = self._get_per_class_metric_table(
                "Segmentation Metrics per class after graph improvement",
                self.graph_segmentation_classification_metric,
                md,
            )

        if self.graph_confmat is not None and self.graph_confmat_norm is not None:
            md = self._get_confusion_matrix(
                "Confusion Matrix after graph improvement",
                self.graph_confmat,
                md,
                False,
            )
            md = self._get_confusion_matrix(
                "Confusion Matrix normalized after graph improvement",
                self.graph_confmat_norm,
                md,
                True,
            )

        if len(self.images) > 0:
            md += "## Images\n\n"
            for i in range(len(self.images)):
                md += "### Image " + str(i + 1) + "\n\n"
                md += f"![](images/test_image_{i}.png)\n"
                md = self._get_simple_metric_table(
                    "Personal segmentation Metrics",
                    self.single_segmentation_metric[i],
                    md,
                )
                if self.n_classes > 1:
                    md += f"\n![](images/test_image_{i}_classes.png)\n"
                    md += f"\n![](images/test_image_{i}_graph.png)\n"
                    md += f"\n![](images/test_image_{i}_diff.png)\n"
        with open(f"{self.output_dir}test_results.md", "w") as f:
            f.write(md)
        self.console.print("Done.")

    def _get_simple_metric_table(self, title: str, metrics: dict, md: str) -> str:
        """Get a simple metric table.

        Args:
            title (str): The title of the table.
            metrics (dict): The metrics.
            md (str): The markdown string.

        Returns:
            md (str): The markdown string with the added table.

        Raises:
            None.
        """
        md += f"## {title}\n\n"
        md += "| Metric | Value |\n"
        md += "| :--- | :---: |\n"
        md += f"| FP | {metrics.fp} |\n"  # type: ignore
        md += f"| TP | {metrics.tp} |\n"  # type: ignore
        md += f"| FN | {metrics.fn} |\n"  # type: ignore
        md += f"| Precision | {metrics.precision} |\n"  # type: ignore
        md += f"| Recall | {metrics.recall} |\n"  # type: ignore
        md += f"| Accuracy | {metrics.accuracy} |\n"  # type: ignore
        md += f"| F1 | {metrics.f1} |\n"  # type: ignore
        md += f"| n_true | {metrics.n_true} |\n"  # type: ignore
        md += f"| n_pred | {metrics.n_pred} |\n"  # type: ignore
        md += f"| mean_true_score | {metrics.mean_true_score} |\n"  # type: ignore
        md += f"| mean_matched_score | {metrics.mean_matched_score} |\n"  # type: ignore
        md += f"| panoptic_quality | {metrics.panoptic_quality} |\n"  # type: ignore
        return md

    def _get_per_class_metric_table(self, title: str, metrics: dict, md: str) -> str:
        """Get a per class metric table.

        Args:
            title (str): The title of the table.
            metrics (dict): The metrics.
            md (str): The markdown string.

        Returns:
            md (str): The markdown string with the added table.

        Raises:
            None.
        """
        md += f"## {title}\n\n"
        md += "| Metric |"
        for i in range(1, self.n_classes):
            md += f" {self.class_name[i]} |"
        md += "\n"
        md += "| :--- |"
        for i in range(1, self.n_classes):
            md += " :---: |"
        md += "\n"
        md += (
            "| FP |"
            + " ".join([f"{metrics[i].fp:.4f} | " for i in range(1, self.n_classes)])
            + "\n"
        )
        md += (
            "| TP |"
            + " ".join([f"{metrics[i].tp:.4f} | " for i in range(1, self.n_classes)])
            + "\n"
        )
        md += (
            "| FN |"
            + " ".join([f"{metrics[i].fn:.4f} | " for i in range(1, self.n_classes)])
            + "\n"
        )
        md += (
            "| Precision |"
            + " ".join(
                [f"{metrics[i].precision:.4f} | " for i in range(1, self.n_classes)]
            )
            + "\n"
        )
        md += (
            "| Recall |"
            + " ".join(
                [f"{metrics[i].recall:.4f} | " for i in range(1, self.n_classes)]
            )
            + "\n"
        )
        md += (
            "| Accuracy |"
            + " ".join(
                [f"{metrics[i].accuracy:.4f} | " for i in range(1, self.n_classes)]
            )
            + "\n"
        )
        md += (
            "| F1 |"
            + " ".join([f"{metrics[i].f1:.4f} | " for i in range(1, self.n_classes)])
            + "\n"
        )
        md += (
            "| n_true |"
            + " ".join(
                [f"{metrics[i].n_true:.4f} | " for i in range(1, self.n_classes)]
            )
            + "\n"
        )
        md += (
            "| n_pred |"
            + " ".join(
                [f"{metrics[i].n_pred:.4f} | " for i in range(1, self.n_classes)]
            )
            + "\n"
        )
        md += (
            "| mean_true_score |"
            + " ".join(
                [
                    f"{metrics[i].mean_true_score:.4f} | "
                    for i in range(1, self.n_classes)
                ]
            )
            + "\n"
        )
        md += (
            "| mean_matched_score |"
            + " ".join(
                [
                    f"{metrics[i].mean_matched_score:.4f} | "
                    for i in range(1, self.n_classes)
                ]
            )
            + "\n"
        )
        md += (
            "| panoptic_quality |"
            + " ".join(
                [
                    f"{metrics[i].panoptic_quality:.4f} | "
                    for i in range(1, self.n_classes)
                ]
            )
            + "\n"
        )

        return md

    def _get_confusion_matrix(
        self, title: str, confmat: torch.tensor, md: str, norm: bool
    ) -> str:
        """Generate confusion matrix table.

        Args:
            title (str): The title of the table.
            confmat (torch.tensor): The confusion matrix.
            md (str): The markdown string.
            norm (bool): Whether to normalize the confusion matrix.

        Returns:
            md (str): The markdown string with the added table.

        Raises:
            None.
        """
        cm = confmat.cpu().numpy()
        index = list(self.class_name.values())
        df = pd.DataFrame(cm, index=index, columns=index)
        if norm:
            ax = sn.heatmap(
                df, annot=True, fmt=".2f", cmap="YlGnBu", cbar=False, vmin=0, vmax=1
            )
        else:
            ax = sn.heatmap(df, annot=True, fmt="g", cmap="YlGnBu", cbar=False)
        ax.set_title(title)
        ax.figsize = (5, 10)
        # ax.set(xlabel=index, ylabel=index)
        save_name = title.replace(" ", "_")
        plt.tight_layout()
        ax.figure.savefig(f"{self.output_dir}images/{save_name}.png", dpi=300)
        ax.figure.clf()
        md += f"## {title}\n\n"
        md += f"![](images/{save_name}.png)\n\n"
        return md

    def _generate_images(self) -> None:
        """Generate images for the report.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        self.console.print("Generating images for the report...")
        for i in tqdm(range(len(self.images))):
            # save images and segmentation masks
            self._save_images(
                [
                    self.images[i].permute(1, 2, 0),
                    self.true_masks[i],
                    self.predicted_masks[i],
                ],
                ["Image", "True instance map", "Stardist instance map"],
                f"test_image_{i}",
            )

            if self.n_classes > 1:
                # save true and predicted classes
                self._save_images(
                    [
                        self.true_class_map[i],
                        self.pred_class_map[i],
                        self.pred_class_map_improved[i],
                    ],
                    [
                        "True class map",
                        "Stardist class map",
                        "Stardist class map x instance map",
                    ],
                    f"test_image_{i}_classes",
                    legend=True,
                )

                # save graph of class map
                if (
                    self.graphs_class_map is not None
                    and len(self.graphs_class_map) > i
                    and self.graphs_images is not None
                    and len(self.graphs_images) > i
                ):
                    self._save_images(
                        [self.graphs_images[i], self.graphs_class_map[i]],
                        ["Graph", "Graph class map"],
                        f"test_image_{i}_graph",
                        legend=True,
                    )

                # save differences between true and predicted maps
                diff_imgs = {}
                diff_imgs["Stardist class map differences"] = np.zeros_like(
                    self.true_class_map[i]
                )
                diff_imgs["Stardist class map differences"][
                    self.true_class_map[i] != self.pred_class_map_improved[i]
                ] = 1
                if (
                    self.graphs_class_map is not None
                    and len(self.graphs_class_map) > i
                    and self.graphs_images is not None
                    and len(self.graphs_images) > i
                ):
                    diff_imgs["graph class map differences"] = np.zeros_like(
                        self.true_class_map[i]
                    )
                    diff_imgs["graph class map differences"][
                        self.true_class_map[i] != self.graphs_class_map[i]
                    ] = 1
                    diff_imgs["Stardist/graph class map differences"] = np.zeros_like(
                        self.true_class_map[i]
                    )
                    diff_imgs["Stardist/graph class map differences"][
                        diff_imgs["Stardist class map differences"]
                        > diff_imgs["graph class map differences"]
                    ] = 1
                    diff_imgs["Stardist/graph class map differences"][
                        diff_imgs["Stardist class map differences"]
                        < diff_imgs["graph class map differences"]
                    ] = 2
                    self._save_images(
                        list(diff_imgs.values()),
                        list(diff_imgs.keys()),
                        f"test_image_{i}_diff",
                    )

    def _save_images(
        self, imgs: list, titles: list, file_name: str, legend: bool = False
    ) -> None:
        """Save images to png files.

        Args:
            imgs (list): list of images to plot.
            titles (list): list of titles for each image.
            file_name (str): name of the file to save.
            legend (bool): if True, plot a legend (used for class maps).

        Returns:
            None.

        Raises:
            None.
        """

        fig, ax = plt.subplots(1, len(imgs), figsize=(5 * len(imgs), 5))
        for i in range(len(imgs)):
            if legend:
                ax[i].imshow(imgs[i], cmap=ListedColormap(self.class_color))
            else:
                ax[i].imshow(imgs[i])
            ax[i].set_title(titles[i])
            ax[i].axis("off")

        if legend:
            fig.legend(
                handles=[
                    Patch(color=self.class_color[i], label=self.class_name[i])
                    for i in range(1, self.n_classes)
                ],
                loc="lower center",
                ncol=self.n_classes,
            )
        # saving
        plt.savefig(f"{self.output_dir}images/{file_name}.png")
        plt.close()

    def _draw_graph(
        self,
        graph: dict,
        class_map: np.ndarray,
    ) -> np.ndarray:
        """Draw graph on class map.

        Args:
            graph (dict): graph to draw.
            class_map (np.ndarray): class map.

        Returns:
            img (np.ndarray): class map with graph drawn.

        Raises:
            None.
        """
        p = graph["pos"]
        ei = graph["edge_index"]
        transform = T.ToPILImage()
        img = transform(class_map.astype(np.uint8))
        draw = ImageDraw.Draw(img)
        for e in range(ei.shape[1]):
            draw.line(
                (p[ei[0][e]][1], p[ei[0][e]][0], p[ei[1][e]][1], p[ei[1][e]][0]),
                fill=(5),
            )
        img = np.array(img)
        return img
