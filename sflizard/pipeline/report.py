from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from sflizard import get_class_color, get_class_name


class ReportGenerator:
    def __init__(
        self,
        images,
        true_masks,
        predicted_masks,
        image_metric,
        test_values,
        test_classes,
        output_dir,
        true_classes=None,
        predicted_classes=None,
        improved_class=None,
        graphs=None,
        graphs_masks=None,
    ):

        self.images = images
        self.true_masks = true_masks
        self.predicted_masks = predicted_masks
        self.image_metric = image_metric
        self.test_values = test_values
        self.test_classes = test_classes
        self.output_dir = output_dir
        self.true_classes = true_classes
        self.predicted_classes = predicted_classes
        self.improved_class = improved_class
        self.graphs = graphs
        self.graphs_masks = graphs_masks

        if self.true_classes is not None:
            self.classification = True

        if self.output_dir is None:
            self.output_dir = Path(__file__).parents[2] / "output/report/"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if self.output_dir[-1] != "/":
            self.output_dir += "/"

        self.class_name = get_class_name()
        self.class_color = get_class_color()
        self.n_classes = len(self.class_name) + 1 if self.classification else 1

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
        md += "## Global metrics\n\n"
        md += "| Metric | Value |\n"
        md += "| :--- | :---: |\n"
        for metric in self.test_values:
            md += (
                f"| {metric} | {torch.tensor(self.test_values[metric]).mean():.4f} |\n"
            )
        if self.classification:
            md += "## Classification metrics\n\n"
            md += "| Metric | Value |\n"
            md += "| :--- | :---: |\n"
            for metric in self.test_classes:
                md += f"| {metric} | {self.test_classes[metric].compute().item()} |\n"
        if len(self.images) > 0:
            md += "## Images\n\n"
            for i in range(len(self.images)):
                md += "### Image " + str(i + 1) + "\n\n"
                md += f"![](images/test_image_{i}.png)\n"
                md += "| Metric | Value |\n"
                md += "| :--- | :---: |\n"
                md += f"| FP | {self.image_metric[i].fp} |\n"
                md += f"| TP | {self.image_metric[i].tp} |\n"
                md += f"| FN | {self.image_metric[i].fn} |\n"
                md += f"| Precision | {self.image_metric[i].precision} |\n"
                md += f"| Recall | {self.image_metric[i].recall} |\n"
                md += f"| Accuracy | {self.image_metric[i].accuracy} |\n"
                md += f"| F1 | {self.image_metric[i].f1} |\n"
                md += f"| n_true | {self.image_metric[i].n_true} |\n"
                md += f"| n_pred | {self.image_metric[i].n_pred} |\n"
                md += f"| mean_true_score | {self.image_metric[i].mean_true_score} |\n"
                md += f"| mean_matched_score | {self.image_metric[i].mean_matched_score} |\n"
                md += (
                    f"| panoptic_quality | {self.image_metric[i].panoptic_quality} |\n"
                )
                if self.classification:
                    md += f"\n![](images/test_image_{i}_classes.png)\n"
                    md += f"\n![](images/test_image_{i}_graph.png)\n"
                    md += f"\n![](images/test_image_{i}_diff.png)\n"
        with open(f"{self.output_dir}test_results.md", "w") as f:
            f.write(md)
        print("Done.")

    def _generate_images(self):
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

            if self.classification:
                # save true and predicted classes
                self._save_images(
                    [
                        self.true_classes[i],
                        self.predicted_classes[i],
                        self.improved_class[i],
                    ],
                    ["True Class", "Predicted Class", "Predicted class after cleaning"],
                    f"test_image_{i}_classes",
                    legend=True,
                )

                # save graph of class map
                self._save_images(
                    [self.graphs[i], self.graphs_masks[i]],
                    ["Graph", "Graph with mask"],
                    f"test_image_{i}_graph",
                    legend=True,
                )

                # save differences between true and predicted maps
                diff_imgs = {}
                diff_imgs["class mask differences"] = np.zeros_like(
                    self.true_classes[i]
                )
                diff_imgs["class mask differences"][
                    self.true_classes[i] != self.improved_class[i]
                ] = 1
                diff_imgs["graph class mask differences"] = np.zeros_like(
                    self.true_classes[i]
                )
                diff_imgs["graph class mask differences"][
                    self.true_classes[i] != self.graphs_masks[i]
                ] = 1
                diff_imgs["class/graph mask differences"] = np.zeros_like(
                    self.true_classes[i]
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

    def _save_images(self, imgs, titles, file_name, legend=False):
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
