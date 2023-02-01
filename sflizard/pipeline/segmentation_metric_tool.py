from typing import Any, Dict

import numpy as np
from rich.console import Console
from rich.table import Table
from stardist.matching import matching_dataset

from sflizard import get_class_name

PRINT_LATEX_STRING = False


class SegmentationMetricTool:
    """A tool to compute metrics for segmentation."""

    def __init__(self, n_classes: int, device: str, console: Console) -> None:
        """Init the metric tool.

        Args:
            n_classes (int): The number of classes.
            device (str): The device to use.
            console (Console): The rich console.

        Returns:
            None.

        Raises:
            None.
        """
        self.n_classes = n_classes
        self.device = device
        self.console = console

        # init tests for segmentation for each class
        self.seg_metrics: Dict[int, Any] = {}

        # init data holders
        self.true_masks = None
        self.predicted_masks = None
        self.true_class_map: Dict[int, Any] = {}
        self.pred_class_map: Dict[int, Any] = {}

    def add_batch(
        self,
        batch_idx: int,
        true_masks: np.array,
        pred_masks: np.array,
    ) -> None:
        """Add a batch to the metric tool.

        Args:
            batch_idx (int): The batch index.
            true_masks (np.array): The true masks.
            pred_masks (np.array): The predicted masks.

        Returns:
            None.

        Raises:
            None.
        """
        # compute metrics for each class
        if batch_idx == 0:
            self.true_masks = true_masks
            self.predicted_masks = pred_masks
        else:
            self.true_masks = np.concatenate(
                [
                    self.true_masks,
                    true_masks,
                ]
            )
            self.predicted_masks = np.concatenate(
                [
                    self.predicted_masks,
                    pred_masks,
                ]
            )

    def add_batch_class(
        self,
        batch_idx: int,
        true_class_map: np.array,
        pred_class_map: np.array,
    ) -> None:
        """Add a batch to the metric tool.

        Args:
            batch_idx (int): The batch index.
            true_class_map (np.array): The true class map.
            pred_class_map (np.array): The predicted class map.

        Returns:
            None.

        Raises:
            None.
        """

        for j in range(1, self.n_classes):
            # save the true class map
            ct = np.copy(true_class_map)
            ct[ct != j] = 0
            if batch_idx == 0:
                self.true_class_map[j] = ct
            else:
                self.true_class_map[j] = np.concatenate([self.true_class_map[j], ct])

            # save the predicted class map
            cp = np.copy(pred_class_map)
            cp[cp != j] = 0
            if batch_idx == 0:
                self.pred_class_map[j] = cp
            else:
                self.pred_class_map[j] = np.concatenate([self.pred_class_map[j], cp])

    def compute_metrics(self) -> None:
        """Compute the metrics.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        # compute segmentation metrics
        if self.true_masks is not None and self.predicted_masks is not None:
            self.console.print("Computing global segmentation metrics...")
            self.seg_metrics[0] = matching_dataset(
                self.true_masks, self.predicted_masks, show_progress=True, parallel=True
            )

        # compute metrics for each class
        for i in range(1, self.n_classes):
            self.console.print(
                f"Computing class {get_class_name()[i]} segmentation metrics..."
            )
            # compute metrics for each class
            tm = np.copy(self.true_masks)
            tm[self.true_class_map[i] == 0] = 0
            pm = np.copy(self.predicted_masks)
            pm[self.pred_class_map[i] == 0] = 0
            self.seg_metrics[i] = matching_dataset(
                tm, pm, show_progress=True, parallel=True
            )

    def log_results(self) -> None:
        """Log the results in rich tables.

        Args:
            None.

        Returns:
            None.

        Raises:
            None.
        """
        latex_table_str = []
        # log segmentation metrics
        if 0 in self.seg_metrics:
            table = Table(title="Segmentation metrics")
            table.add_column("Metric", justify="center")
            table.add_column("Value", justify="center")
            table.add_row(
                "precision",
                f"{self.seg_metrics[0].precision:.4f}",
            )
            table.add_row(
                "recall",
                f"{self.seg_metrics[0].recall:.4f}",
            )
            table.add_row(
                "acc",
                f"{self.seg_metrics[0].accuracy:.4f}",
            )
            table.add_row(
                "f1",
                f"{self.seg_metrics[0].f1:.4f}",
            )
            table.add_row(
                "panoptic_quality",
                f"{self.seg_metrics[0].panoptic_quality:.4f}",
            )
            self.console.print(table)

            # save latex table
            if PRINT_LATEX_STRING:
                table_str = r"\begin{center}" + "\n" + r"\begin{tabular}{|c c|}"
                table_str += "\nSegmentation metrics\\\\ \n \\hline \n"
                table_str += "Metric & Result \\\\ \n \\hline \n"
                table_str += f"Precision & {self.seg_metrics[0].precision:.4f} \\\\ \n"
                table_str += f"Recall & {self.seg_metrics[0].recall:.4f} \\\\ \n"
                table_str += f"Acc & {self.seg_metrics[0].accuracy:.4f} \\\\ \n"
                table_str += f"F1 & {self.seg_metrics[0].f1:.4f} \\\\ \n"
                table_str += f"Panoptic quality & {self.seg_metrics[0].panoptic_quality:.4f} \\\\ \n"
                table_str += (
                    r"\hline" + "\n" + r"\end{tabular}" + "\n" + r"\end{center}"
                )
                latex_table_str.append(table_str)

        if self.n_classes > 1:

            # log per class segmentation metrics
            table = Table(title="Per class metrics")
            table.add_column("Class", justify="center")
            for i in range(1, self.n_classes):
                table.add_column(f"{get_class_name()[i]}", justify="center")
            table.add_row(
                "precision",
                *[
                    f"{self.seg_metrics[i].precision:.4f}"
                    for i in range(1, self.n_classes)
                ],
            )
            table.add_row(
                "recall",
                *[
                    f"{self.seg_metrics[i].recall:.4f}"
                    for i in range(1, self.n_classes)
                ],
            )
            table.add_row(
                "acc",
                *[
                    f"{self.seg_metrics[i].accuracy:.4f}"
                    for i in range(1, self.n_classes)
                ],
            )
            table.add_row(
                "f1",
                *[f"{self.seg_metrics[i].f1:.4f}" for i in range(1, self.n_classes)],
            )
            table.add_row(
                "panoptic_quality",
                *[
                    f"{self.seg_metrics[i].panoptic_quality:.4f}"
                    for i in range(1, self.n_classes)
                ],
            )
            self.console.print(table)

            # save latex table
            if PRINT_LATEX_STRING:
                table_str = (
                    r"\begin{center}" + "\n" + r"\begin{tabular}{|c c c c c c c|}"
                )
                table_str += "\nPer class metrics\\\\ \n \\hline \n"
                table_str += "Class "
                for i in range(1, self.n_classes):
                    table_str += f"& {get_class_name()[i]} "
                table_str += "\\\\ \n \\hline \n"
                table_str += "Precision "
                for i in range(1, self.n_classes):
                    table_str += f"& {self.seg_metrics[i].precision:.4f} "
                table_str += "\\\\ \n"
                table_str += "Recall "
                for i in range(1, self.n_classes):
                    table_str += f"& {self.seg_metrics[i].recall:.4f} "
                table_str += "\\\\ \n"
                table_str += "Accuracy "
                for i in range(1, self.n_classes):
                    table_str += f"& {self.seg_metrics[i].accuracy:.4f} "
                table_str += "\\\\ \n"
                table_str += "F1 "
                for i in range(1, self.n_classes):
                    table_str += f"& {self.seg_metrics[i].f1:.4f} "
                table_str += "\\\\ \n"
                table_str += "Panoptic quality "
                for i in range(1, self.n_classes):
                    table_str += f"& {self.seg_metrics[i].panoptic_quality:.4f} "
                table_str += "\\\\ \n"
                table_str += (
                    r"\hline" + "\n" + r"\end{tabular}" + "\n" + r"\end{center}"
                )
                latex_table_str.append(table_str)

        if PRINT_LATEX_STRING:
            for table_str in latex_table_str:
                print(table_str)
