from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from stardist.matching import matching
import torchvision.transforms as T
from PIL import ImageDraw

from sflizard import get_class_color, get_class_name


class ReportGenerator:
    """MD report generator."""

    def __init__(
        self, 
        output_dir: str, 
        imgs_to_display: int, 
        n_classes: int,
    )-> None:
        """Report generator.
        
        Args:
            output_dir (str): The output directory.
            imgs_to_display (int): The number of images to display.
            n_classes (int): The number of classes.
            
        Returns:
            None.
            
        Raises:
            None.
        """
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if self.output_dir[-1] != "/":
            self.output_dir += "/"

        self.imgs_to_display = imgs_to_display

        self.n_classes = n_classes
        self.class_name = get_class_name()
        self.class_color = get_class_color()
        

        # images and masks
        self.images = []
        self.true_masks = []
        self.predicted_masks = []

        self.true_class_map = []
        self.pred_class_map = []
        self.pred_class_map_improved = []

        self.graphs_images = []
        self.graphs_class_map = []

        # metrics
        self.single_segmentation_metric = []
        self.segmentation_metric = None
        self.classification_metric = None
        self.graph_classification_metric = None
        self.segmentation_classification_metric = None
        self.graph_segmentation_classification_metric = None
        
    def add_batch(
        self, 
        images: torch.Tensor,
        true_masks: np.array,
        pred_masks: np.array,
        true_class_map: torch.Tensor=None,
        pred_class_map: torch.Tensor=None,
        pred_class_map_improved: torch.Tensor=None,
        graphs: list=None,
        graphs_class_map: np.array=None,
    ) -> None:
        """Add a batch to the report.
        
        Args:
            images (torch.Tensor): The images.
            true_masks (np.array): The true masks.
            pred_masks (np.array): The predicted masks.
            true_class_map (torch.Tensor): The true class map.
            pred_class_map (torch.Tensor): The predicted class map.
            pred_class_map_improved (torch.Tensor): The improved predicted class map.
            graphs (list): The graphs.
            graphs_class_map (np.array): The graphs class map.
            
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
                self.single_segmentation_metric.append(matching(true_masks[i], pred_masks[i]))
                if true_class_map is not None:
                    self.true_class_map.append(true_class_map[i].cpu().numpy())
                    self.pred_class_map.append(pred_class_map[i].cpu().numpy())
                    self.pred_class_map_improved.append(pred_class_map_improved[i].cpu().numpy())
                    if graphs is not None:
                        self.graphs_images.append(
                            self._draw_graph(graphs[i], pred_class_map_improved[i].cpu().numpy())
                        )
                        self.graphs_class_map.append(graphs_class_map[i])
            self.imgs_to_display = max(0, self.imgs_to_display - len(images))

    def add_final_metrics(
        self, 
        segmentation_metric: dict, 
        classification_metric: dict,
        graph_classification_metric: dict,
        segmentation_classification_metric: dict,
        graph_segmentation_classification_metric: dict,
    ) -> None:
        """Add final metrics to the report.
        
        Args:
            segmentation_metric (dict): The segmentation metric.
            classification_metric (dict): The classification metric.
            graph_classification_metric (dict): The graph classification metric.
            segmentation_classification_metric (dict): The segmentation classification metric.
            graph_segmentation_classification_metric (dict): The graph segmentation classification metric.
            
        Returns:
            None.
            
        Raises:
            None.
        """
        self.segmentation_metric = segmentation_metric
        self.classification_metric = classification_metric
        self.graph_classification_metric = graph_classification_metric
        self.segmentation_classification_metric = segmentation_classification_metric
        self.graph_segmentation_classification_metric = graph_segmentation_classification_metric

    def generate_md(self) -> None:
        """Generate a markdown file with the report.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        print("Plotting...")

        # create subdirectories
        Path(self.output_dir + "/images").mkdir(parents=True, exist_ok=True)

        # generate images
        self._generate_images()

        # create MD report containing images and metrics
        md = "# Test results\n\n"

        # add metrics
        md = self._get_simple_metric_table("Segmentation Metrics", self.segmentation_metric, md)
        if self.segmentation_classification_metric is not None:
            md = self._get_per_class_metric_table("Segmentation Metrics per class", self.segmentation_classification_metric, md)
        if self.graph_segmentation_classification_metric is not None:
            md = self._get_per_class_metric_table("Segmentation Metrics per class after graph improvement", self.graph_segmentation_classification_metric, md)

        if self.n_classes > 1:
            md += "## Classification metrics\n\n"
            md += "| Metric | Value |\n"
            md += "| :--- | :---: |\n"
            for metric in self.classification_metric:
                md += f"| {metric} | {self.classification_metric[metric].compute().item()} |\n"
            md += "\n"
        if self.graph_classification_metric is not None:
            md += "## Classification metrics after graph improvement\n\n"
            md += "| Metric | Value |\n"
            md += "| :--- | :---: |\n"
            for metric in self.graph_classification_metric:
                md += f"| {metric} | {self.graph_classification_metric[metric].compute().item()} |\n"
            md += "\n"

        if len(self.images) > 0:
            md += "## Images\n\n"
            for i in range(len(self.images)):
                md += "### Image " + str(i + 1) + "\n\n"
                md += f"![](images/test_image_{i}.png)\n"
                md = self._get_simple_metric_table("Personal segmentation Metrics", self.single_segmentation_metric[i], md)
                if self.n_classes > 1:
                    md += f"\n![](images/test_image_{i}_classes.png)\n"
                    md += f"\n![](images/test_image_{i}_graph.png)\n"
                    md += f"\n![](images/test_image_{i}_diff.png)\n"
        with open(f"{self.output_dir}test_results.md", "w") as f:
            f.write(md)
        print("Done.")

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
        md += f"| FP | {metrics.fp} |\n"
        md += f"| TP | {metrics.tp} |\n"
        md += f"| FN | {metrics.fn} |\n"
        md += f"| Precision | {metrics.precision} |\n"
        md += f"| Recall | {metrics.recall} |\n"
        md += f"| Accuracy | {metrics.accuracy} |\n"
        md += f"| F1 | {metrics.f1} |\n"
        md += f"| n_true | {metrics.n_true} |\n"
        md += f"| n_pred | {metrics.n_pred} |\n"
        md += f"| mean_true_score | {metrics.mean_true_score} |\n"
        md += f"| mean_matched_score | {metrics.mean_matched_score} |\n"
        md += (
            f"| panoptic_quality | {metrics.panoptic_quality} |\n"
        )
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
            md += f" {get_class_name()[i]} |"
        md += "\n"
        md += "| :--- |"
        for i in range(1, self.n_classes):
            md += " :---: |"
        md += "\n"
        md += "| FP |" + ' '.join([f"{metrics[i].fp:.4f} | " for i in range(1, self.n_classes)]) + "\n"
        md += f"| TP |" + ' '.join([f"{metrics[i].tp:.4f} | " for i in range(1, self.n_classes)]) + "\n"
        md += f"| FN |" + ' '.join([f"{metrics[i].fn:.4f} | " for i in range(1, self.n_classes)]) + "\n"
        md += f"| Precision |" + ' '.join([f"{metrics[i].precision:.4f} | " for i in range(1, self.n_classes)]) + "\n"
        md += f"| Recall |" + ' '.join([f"{metrics[i].recall:.4f} | " for i in range(1, self.n_classes)]) + "\n"
        md += f"| Accuracy |" + ' '.join([f"{metrics[i].accuracy:.4f} | " for i in range(1, self.n_classes)]) + "\n"
        md += f"| F1 |" + ' '.join([f"{metrics[i].f1:.4f} | " for i in range(1, self.n_classes)]) + "\n"
        md += f"| n_true |" + ' '.join([f"{metrics[i].n_true:.4f} | " for i in range(1, self.n_classes)]) + "\n"
        md += f"| n_pred |" + ' '.join([f"{metrics[i].n_pred:.4f} | " for i in range(1, self.n_classes)]) + "\n"
        md += f"| mean_true_score |" + ' '.join([f"{metrics[i].mean_true_score:.4f} | " for i in range(1, self.n_classes)]) + "\n"
        md += f"| mean_matched_score |" + ' '.join([f"{metrics[i].mean_matched_score:.4f} | " for i in range(1, self.n_classes)]) + "\n"
        md += (
            f"| panoptic_quality |" + ' '.join([f"{metrics[i].panoptic_quality:.4f} | " for i in range(1, self.n_classes)]) + "\n"
        )
        
        return md

    def _generate_images(self)-> None:
        """Generate images for the report.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        for i in range(len(self.images)):
            # save images and segmentation masks
            self._save_images(
                [
                    self.images[i].permute(1, 2, 0),
                    self.true_masks[i],
                    self.predicted_masks[i],
                ],
                ["Image", "True mask", "Predicted mask"],
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
                    ["True Class map", "Predicted Class map", "Predicted class map improved"],
                    f"test_image_{i}_classes",
                    legend=True,
                )

                # save graph of class map
                self._save_images(
                    [self.graphs_images[i], self.graphs_class_map[i]],
                    ["Graph", "Graph Class map"],
                    f"test_image_{i}_graph",
                    legend=True,
                )

                # save differences between true and predicted maps
                diff_imgs = {}
                diff_imgs["class mask differences"] = np.zeros_like(
                    self.true_class_map[i]
                )
                diff_imgs["class mask differences"][
                    self.true_class_map[i] != self.pred_class_map_improved[i]
                ] = 1
                diff_imgs["graph class mask differences"] = np.zeros_like(
                    self.true_class_map[i]
                )
                diff_imgs["graph class mask differences"][
                    self.true_class_map[i] != self.graphs_class_map[i]
                ] = 1
                diff_imgs["class/graph mask differences"] = np.zeros_like(
                    self.true_class_map[i]
                )
                diff_imgs["class/graph mask differences"][
                    diff_imgs["class mask differences"]
                    > diff_imgs["graph class mask differences"]
                ] = 1
                diff_imgs["class/graph mask differences"][
                    diff_imgs["class mask differences"]
                    < diff_imgs["graph class mask differences"]
                ] = 2
                self._save_images(
                    list(diff_imgs.values()),
                    list(diff_imgs.keys()),
                    f"test_image_{i}_diff",
                )

    def _save_images(
        self, 
        imgs: list, 
        titles: list, 
        file_name: str, 
        legend: bool=False
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
        class_map: np.array,
    ) -> np.array:
        """Draw graph on class map.

        Args:
            graph (dict): graph to draw.
            class_map (np.array): class map.

        Returns:
            img (np.array): class map with graph drawn.

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