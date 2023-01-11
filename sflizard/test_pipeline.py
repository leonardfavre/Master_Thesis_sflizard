import argparse
import copy

import numpy as np
import torch
import torchmetrics
import torchvision.transforms as T
from PIL import ImageDraw
from rich.console import Console
from rich.table import Table
from stardist.matching import matching, matching_dataset
from tqdm import tqdm

from sflizard import (
    Graph,
    LizardDataModule,
    ReportGenerator,
    Stardist,
    get_class_name,
    get_graph_for_inference,
)

VALID_DATA_PATH = "data/Lizard_dataset_extraction/data_0.9_split_valid.pkl"
TEST_DATA_PATH = "data/Lizard_dataset_extraction/data_0.9_split_test.pkl"
STARDIST_WEIGHTS_PATH = "models/stardist_1000epochs_0.0losspower_0.0005lr.ckpt"
STAR_4_IMPROVEMENT = False
GRAPH_WEIGHTS_PATH = None  # "models/full_training_graph_test_2000_08413.ckpt"
N_RAYS = 32
N_CLASSES = 7
BATCH_SIZE = 4
SEED = 303
OUTPUT_DIR = None  # "./pipeline_output/full_training_stardist_class_1000_111_1e-4/"
IMGS_TO_DISPLAY = 10
X_TYPE = {
    128: "ll",
    135: "ll+c",
    137: "ll+c+x",
    512: "4ll",
    540: "4ll+c",
    548: "4ll+c+x",
}
DISTANCE = 45
MODE = "test"


class TestPipeline:
    def __init__(
        self,
        valid_data_path: str,
        test_data_path: str,
        stardist_weights_path: str,
        graph_weights_path: str,
        n_rays: int,
        n_classes: int,
        batch_size: int,
        seed: int,
        mode: str,
    ):
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

    def __init_dataloader(
        self,
        valid_data_path: str,
        test_data_path: str,
        seed: int = 303,
        batch_size: int = 1,
        mode: str = "valid",
    ) -> None:
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
            self.dataloader = iter(dm.val_dataloader())
        print("Data loaded.")

    def __init_stardist_inference(
        self,
        weights_path: str,
    ) -> None:
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

    def __init_graph_inference(self, weights_path: str) -> None:
        print("Loading graph model...")
        model = Graph.load_from_checkpoint(
            weights_path,
        )
        self.x_type = X_TYPE[model.num_features]
        self.graph = model.model.to(self.device)
        print("Graph model loaded.")

    def test(self, output_dir=None, imgs_to_display=0) -> None:
        print("Testing...")

        # init test metrics
        self.__init_test_metrics()

        # results list
        if self.classification:
            true_classes = [None]
            predicted_classes = [None]
            if self.compute_graph:
                predicted_graph_classes = [None]

        # init lists to store images and masks
        if output_dir is not None and imgs_to_display > 0:
            o_images = []
            o_true_masks = []
            o_predicted_masks = []
            o_metrics = []
            if self.classification:
                o_true_classes = []
                o_predicted_classes = []
                o_improved_classes = []
                if self.compute_graph:
                    o_graphs = []
                    o_graphs_masks = []

        for i in tqdm(range(len(self.dataloader))):  # type: ignore

            # if i > 3:
            #     break

            # get next test batch
            batch = next(self.dataloader)
            for b in range(len(batch)):
                batch[b] = batch[b].to(self.device)
            if self.classification:
                inputs, obj_probabilities, distances, classes = batch
                classes = classes.int()
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

                def rotate_and_pred(inputs, angle):
                    inputs_rotated = torch.rot90(inputs, angle, [2, 3])
                    with torch.no_grad():
                        dist, prob, c = self.stardist(inputs_rotated)
                    pred_mask_rotated = self.stardist.compute_star_label(
                        inputs_rotated, dist, prob
                    )
                    # return rotated np array
                    return np.rot90(pred_mask_rotated, -angle, [1, 2]), torch.rot90(
                        c.cpu(), -angle, [2, 3]
                    )

                # rotate input
                pred_90_mask, clas_90 = rotate_and_pred(inputs, 1)
                pred_180_mask, clas_180 = rotate_and_pred(inputs, 2)
                pred_270_mask, clas_270 = rotate_and_pred(inputs, 3)

            # stardist class mask
            if self.classification:

                def get_class_pred(clas, pred_mask):
                    best_clas = torch.argmax(clas, dim=1)
                    class_pred = torch.Tensor(
                        np.array(
                            [
                                self.__improve_class_map(
                                    best_clas[b].cpu(), pred_mask[b]
                                )
                                for b in range(len(best_clas))
                            ]
                        )
                    )
                    return class_pred, best_clas

                class_pred, best_clas = get_class_pred(clas, pred_mask)
                if STAR_4_IMPROVEMENT:
                    class_pred_90, _ = get_class_pred(clas_90, pred_90_mask)
                    class_pred_180, _ = get_class_pred(clas_180, pred_180_mask)
                    class_pred_270, _ = get_class_pred(clas_270, pred_270_mask)
                    class_pred = self._merge_stardist_class_together(
                        class_pred, class_pred_90, class_pred_180, class_pred_270
                    )

            if self.compute_graph:
                # graph predicted mask

                graph = get_graph_for_inference(
                    inputs, args.distance, self.stardist_weights_path, self.x_type
                )

                with torch.no_grad():
                    graph_pred = []
                    for j in range(len(graph)):
                        if len(graph[j]["x"]) > 0:
                            out = self.graph(
                                graph[j]["x"].to(self.device),
                                graph[j]["edge_index"].to(self.device),
                                # graph[j]["edge_attr"].to(self.device),
                            )
                            graph_pred.append(out.argmax(-1))  # + 1)
                        else:
                            graph_pred.append(None)

                    graph_pred_mask = self._get_class_map_from_graph(
                        graph, pred_mask, graph_pred, class_pred
                    )

            # metrics gattering
            if i == 0:
                true_masks = true_mask
                predicted_masks = pred_mask
                if self.compute_graph:
                    graphs_masks = graph_pred_mask
            else:
                true_masks = np.concatenate(
                    [
                        true_masks,
                        true_mask,
                    ]
                )
                predicted_masks = np.concatenate(
                    [
                        predicted_masks,
                        pred_mask,
                    ]
                )
                if self.compute_graph:
                    graphs_masks = np.concatenate(
                        [
                            graphs_masks,
                            graph_pred_mask,
                        ]
                    )

            if self.classification:
                # compute classification metrics
                for metric in self.test_classes:
                    self.test_classes[metric](
                        torch.Tensor(class_pred).int().to(self.device), classes
                    )
                # compote per class segmentation metrics
                # true_classes = np.array([[
                #     torch.tensor(classes).int().cpu().numpy()[torch.tensor(classes).int().cpu().numpy() != j]
                # ] for j in range(1, self.n_classes)])
                # predicted_classes = np.array([[
                #     torch.tensor(class_pred).int().cpu().numpy()[torch.tensor(class_pred).int().cpu().numpy() != j]
                # ] for j in range(1, self.n_classes)])

                for j in range(1, self.n_classes):
                    ct = classes.clone().detach().int().cpu().numpy()
                    ct[ct != j] = 0
                    if i == 0:
                        true_classes.append(ct)
                    else:
                        true_classes[j] = np.concatenate([true_classes[j], ct])
                    cp = class_pred.clone().detach().int().cpu().numpy()
                    cp[cp != j] = 0
                    if i == 0:
                        predicted_classes.append(cp)
                    else:
                        predicted_classes[j] = np.concatenate(
                            [predicted_classes[j], cp]
                        )
                    if self.compute_graph:
                        gp = np.array(graph_pred_mask)
                        gp[gp != j] = 0
                        if i == 0:
                            predicted_graph_classes.append(gp)
                        else:
                            predicted_graph_classes[j] = np.concatenate(
                                [predicted_graph_classes[j], gp]
                            )

            # compute per image metrics for images to display
            if output_dir is not None and imgs_to_display > 0:
                for j in range(min(imgs_to_display, len(images))):
                    o_images.append(images[j].cpu())
                    o_true_masks.append(true_mask[j])
                    o_predicted_masks.append(pred_mask[j])
                    o_metrics.append(matching(true_mask[j], pred_mask[j]))
                    if self.classification:
                        o_true_classes.append(classes[j].cpu().numpy())
                        o_predicted_classes.append(best_clas[j].cpu().numpy())
                        o_improved_classes.append(class_pred[j].cpu().numpy())
                        if self.compute_graph:
                            o_graphs.append(
                                self._draw_graph(graph[j], class_pred[j].cpu().numpy())
                            )
                            o_graphs_masks.append(graph_pred_mask[j])
                imgs_to_display = max(0, imgs_to_display - len(images))

        # compute metrics
        self.test_metrics = matching_dataset(
            true_masks, predicted_masks, show_progress=False
        )
        if self.compute_graph:
            self.test_graph_metrics = matching_dataset(
                true_masks, graphs_masks, show_progress=False
            )

        if self.classification:
            self.classification_metrics = {}
            for i in range(1, self.n_classes):
                self.classification_metrics[i] = matching_dataset(
                    true_classes[i], predicted_classes[i], show_progress=False
                )
            if self.compute_graph:
                self.classification_graph_metrics = {}
                for i in range(1, self.n_classes):
                    self.classification_graph_metrics[i] = matching_dataset(
                        true_classes[i], predicted_graph_classes[i], show_progress=False
                    )

        print("Test done.\n\nResults:\n")

        # compute mean metrics and log
        console = Console()
        table = Table(title="Segmentation metrics")
        table.add_column("Metric", justify="center")
        table.add_column("Value", justify="center")

        table.add_row(
            "precision",
            f"{self.test_metrics.precision:.4f}",
        )
        table.add_row(
            "recall",
            f"{self.test_metrics.recall:.4f}",
        )
        table.add_row(
            "acc",
            f"{self.test_metrics.accuracy:.4f}",
        )
        table.add_row(
            "f1",
            f"{self.test_metrics.f1:.4f}",
        )
        table.add_row(
            "panoptic_quality",
            f"{self.test_metrics.panoptic_quality:.4f}",
        )

        console.print(table)

        if self.classification:
            table = Table(title="Classification metrics")
            table.add_column("metric \\ avg", justify="center")
            table.add_column("micro", justify="center")
            table.add_column("macro", justify="center")
            table.add_row(
                "Accuracy",
                str(self.test_classes["accuracy micro"].compute().item()),
                str(self.test_classes["accuracy macro"].compute().item()),
            )
            table.add_row(
                "F1",
                str(self.test_classes["f1 micro"].compute().item()),
                str(self.test_classes["f1 macro"].compute().item()),
            )
            console.print(table)

            # per class metrics
            table = self.__get_per_class_table(
                self.classification_metrics, "Per class metrics"
            )

            console.print(table)
            if self.compute_graph:
                # per class graph metrics
                table = self.__get_per_class_table(
                    self.classification_graph_metrics, "Per class metrics"
                )

                console.print(table)

        if output_dir is not None:
            # plot images and masks and save to file
            if self.classification and self.compute_graph:
                ReportGenerator(
                    o_images,
                    o_true_masks,
                    o_predicted_masks,
                    o_metrics,
                    self.test_values,
                    self.test_classes,
                    output_dir,
                    o_true_classes,
                    o_predicted_classes,
                    o_improved_classes,
                    o_graphs,
                    o_graphs_masks,
                ).generate_md()
            elif self.classification:
                ReportGenerator(
                    o_images,
                    o_true_masks,
                    o_predicted_masks,
                    o_metrics,
                    self.test_values,
                    self.test_classes,
                    output_dir,
                    o_true_classes,
                    o_predicted_classes,
                    o_improved_classes,
                ).generate_md()
            else:
                ReportGenerator(
                    o_images,
                    o_true_masks,
                    o_predicted_masks,
                    o_metrics,
                    self.test_values,
                    output_dir=output_dir,
                ).generate_md()

    def __get_per_class_table(self, metrics, title):
        table = Table(title=title)
        table.add_column("Class", justify="center")
        for i in range(1, self.n_classes):
            table.add_column(f"{get_class_name()[i]}", justify="center")
        table.add_row(
            "precision",
            *[f"{metrics[i].precision:.4f}" for i in range(1, self.n_classes)],
        )
        table.add_row(
            "recall",
            *[f"{metrics[i].recall:.4f}" for i in range(1, self.n_classes)],
        )
        table.add_row(
            "acc",
            *[f"{metrics[i].accuracy:.4f}" for i in range(1, self.n_classes)],
        )
        table.add_row(
            "f1",
            *[f"{metrics[i].f1:.4f}" for i in range(1, self.n_classes)],
        )
        table.add_row(
            "panoptic_quality",
            *[f"{metrics[i].panoptic_quality:.4f}" for i in range(1, self.n_classes)],
        )
        return table

    def __improve_class_map(self, class_map, predicted_masks):
        # add one dim if only 2 dims
        improved_class_map = np.zeros_like(class_map)
        for i in range(1, np.unique(predicted_masks).shape[0]):
            present_class = np.unique(class_map[predicted_masks == i])
            if len(present_class) == 0:
                best_class = 0
            for j in range(len(present_class)):
                best_class = present_class[j]
                if best_class != 0:
                    break
            improved_class_map[predicted_masks == i] = best_class
        return improved_class_map

    def _get_class_map_from_graph(self, graph, predicted_masks, graph_pred, class_pred):
        class_maps = []
        for idx, pm in enumerate(predicted_masks):
            if graph_pred[idx] is None:
                class_maps.append(class_pred[idx].int().cpu().numpy())
            else:
                class_map = np.zeros_like(pm)
                graph_points = graph[idx]["pos"].int().cpu().numpy()
                for i in range(1, np.unique(pm).shape[0]):
                    # get all points in mask with value i
                    points = np.argwhere(pm == i)
                    # get the point in the graph included in the points
                    for idp, p in enumerate(graph_points):
                        if (p == points).all(axis=1).any():
                            class_map[pm == i] = (
                                graph_pred[idx][idp].int().cpu().numpy()
                            )
                            break
                class_maps.append(class_map)
        return class_maps

    def _draw_graph(self, graph, class_map):
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

    def __init_test_metrics(self) -> None:

        # init tests for stardist segmentation
        self.test_values: dict[str, list] = {
            "precision": [],
            "recall": [],
            "acc": [],
            "f1": [],
            "panoptic_quality": [],
        }
        # init tests for stardist classification
        if self.classification:
            self.test_classes = {}
            self.test_classes["accuracy micro"] = torchmetrics.Accuracy(
                num_classes=self.n_classes, mdmc_average="global"
            ).to(self.device)
            self.test_classes["f1 micro"] = torchmetrics.F1Score(
                num_classes=self.n_classes, mdmc_average="global"
            ).to(self.device)
            self.test_classes["accuracy macro"] = torchmetrics.Accuracy(
                num_classes=self.n_classes, average="macro", mdmc_average="global"
            ).to(self.device)
            self.test_classes["f1 macro"] = torchmetrics.F1Score(
                num_classes=self.n_classes, average="macro", mdmc_average="global"
            ).to(self.device)
            # init tests for stardist segmentation
            self.per_classes_test: dict[str, list] = {
                "precision": [[] for _ in range(self.n_classes)],
                "recall": [[] for _ in range(self.n_classes)],
                "acc": [[] for _ in range(self.n_classes)],
                "f1": [[] for _ in range(self.n_classes)],
                "panoptic_quality": [[] for _ in range(self.n_classes)],
            }

    def _merge_stardist_class_together(self, p0, p1, p2, p3):

        class_map = torch.stack((p0, p1, p2, p3), dim=3)
        class_map = class_map.numpy().astype(int)
        class_map = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), 3, class_map
        )
        class_map = torch.from_numpy(class_map)

        return class_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-vdp",
        "--valid_data_path",
        type=str,
        default=VALID_DATA_PATH,
        help="Path to the .pkl file containing the data.",
    )
    parser.add_argument(
        "-tdp",
        "--test_data_path",
        type=str,
        default=TEST_DATA_PATH,
        help="Path to the .pkl file containing the test data.",
    )
    parser.add_argument(
        "-swp",
        "--stardist_weights_path",
        type=str,
        default=STARDIST_WEIGHTS_PATH,
        help="Path to the file containing the stardist model weights.",
    )
    parser.add_argument(
        "-gwp",
        "--graph_weights_path",
        type=str,
        default=GRAPH_WEIGHTS_PATH,
        help="Path to the file containing the graph model weights.",
    )
    parser.add_argument(
        "-nr",
        "--n_rays",
        type=int,
        default=N_RAYS,
        help="Number of rays to use in the stardist model.",
    )
    parser.add_argument(
        "-nc",
        "--n_classes",
        type=int,
        default=N_CLASSES,
        help="Number of classes to use in the stardist model (1 = no classification).",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size to use during training.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=SEED,
        help="Seed to use for the data split.",
    )
    parser.add_argument(
        "-od",
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Path to the directory where the results will be saved.",
    )
    parser.add_argument(
        "-itd",
        "--imgs_to_display",
        type=int,
        default=IMGS_TO_DISPLAY,
        help="Number of images to display in the report.",
    )
    parser.add_argument(
        "-d",
        "--distance",
        type=int,
        default=DISTANCE,
        help="Distance to use for the graph.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default=MODE,
        help="Mode to use for the test ('valid' or 'test').",
    )

    args = parser.parse_args()

    print("Testing pipeline...")
    pipeline = TestPipeline(
        valid_data_path=args.valid_data_path,
        test_data_path=args.test_data_path,
        stardist_weights_path=args.stardist_weights_path,
        graph_weights_path=args.graph_weights_path,
        n_rays=args.n_rays,
        n_classes=args.n_classes,
        batch_size=args.batch_size,
        seed=args.seed,
        mode=args.mode,
    )
    pipeline.test(
        output_dir=args.output_dir,
        imgs_to_display=args.imgs_to_display,
    )
