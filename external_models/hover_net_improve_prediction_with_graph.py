import torch
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import pytorch_lightning as pl
from torch_geometric.nn import GraphSAGE
import scipy.io as sio
from pathlib import Path
import numpy as np

def get_edge_list(vertex, distance):
    # edge distanche
    def distance_between_vertex(v_i, v_j):
        distance = ((v_i[0] - v_j[0]) ** 2 + (v_i[1] - v_j[1]) ** 2) ** (0.5)
        return distance

    edge_list = [[], [], []]
    for i in range(vertex.shape[0]):
        for j in range(i + 1, vertex.shape[0]):
            dist = distance_between_vertex(vertex[i], vertex[j])
            if dist < distance:
                edge_list[0].append(i)
                edge_list[1].append(j)
                edge_list[0].append(j)
                edge_list[1].append(i)
                edge_list[2].append(dist)
                edge_list[2].append(dist)
    return edge_list

def get_graph_for_inference(points,
    predicted_class,
    distance):

    graph = {}
    # add class prediction info to graph
    graph["x"] = torch.Tensor(predicted_class)

    graph["pos"] = torch.Tensor(points)

    # compute edge information
    edge_list = get_edge_list(points, distance)

    # add edge information to graph
    graph["edge_index"] = torch.tensor([edge_list[0], edge_list[1]], dtype=torch.long)
    graph["edge_attr"] = torch.tensor(edge_list[2], dtype=torch.float)
    return graph

class Graph(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.01,
        num_features: int = 1,
        num_classes: int = 5,
        seed: int = 303,
        max_epochs: int = 20,
        dim_h: int = 32,
        num_layers: int = 2,
    ):

        super().__init__()
        
        self.learning_rate = learning_rate
        self.num_features = num_features
        self.num_classes = num_classes
        
        self.model = GraphSAGE(
            in_channels=self.num_features,
            hidden_channels=dim_h,
            num_layers=num_layers,
            out_channels=self.num_classes,
        )

        print(self.model)

        self.seed = seed
        self.max_epochs = max_epochs

        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, edge_index):
        """Forward pass."""
        return self.model(x, edge_index)

    def _step(self, batch, name):
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        label = batch.y
        label = label.long()

        outputs = self.model(x, edge_index)
        loss = self.loss(outputs, label)
        pred = outputs.argmax(-1)
        accuracy = (pred == label).sum() / pred.shape[0]

        if name == "train":
            self.log("train_loss", loss)
            self.log("train_acc", accuracy)
            return loss
        elif name == "val":
            return loss, accuracy
        elif name == "test":
            return loss, accuracy
        else:
            raise ValueError(f"Invalid step name given: {name}")

    def training_step(self, batch, batch_idx):
        """Training step."""
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Test step."""
        return self._step(batch, "test")

    def _epoch_end(self, outputs, name):
        """Epoch end."""

        loss = 0.0
        accuracy = 0
        batch_nbr = 0

        for lo, a in outputs:
            if lo == lo:
                loss += lo
            if a == a:
                accuracy += a
            batch_nbr += 1

        loss /= batch_nbr
        accuracy /= batch_nbr

        self.log(f"{name}_acc", accuracy)
        self.log(f"{name}_loss", loss)

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, "test")

    def configure_optimizers(self, scheduler="cosine"):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=5e-4,
        )
        schedulers = {
            "cosine": LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=int(self.max_epochs / 10),
                max_epochs=self.max_epochs,
            ),
            "step": torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[5, 10], gamma=0.1
            ),
            "lambda": torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda epoch: 0.95**epoch
            ),
        }

        if scheduler not in schedulers.keys():
            raise ValueError(
                f"Invalid scheduler given: {scheduler}. You can implement a new one by modifying the Classifier.configure_optimizers method."
            )

        scheduler = schedulers[scheduler]
        return [optimizer], [scheduler]


def init_graph_inference(weights_path: str, dim_h, num_layers) -> None:
        print("Loading graph model...")
        model = Graph.load_from_checkpoint(
            weights_path,
            num_features = 1,
            num_classes = 7,
            dim_h=dim_h,
            num_layers=num_layers,
        )
        graph = model.model
        print("Graph model loaded.")
        return graph

if __name__ == "__main__":
    device = "cuda"

    model_list = [
        # ["models/full_training_hover_net_graph_500epochs_1layer_sage_16h.ckpt", "1-16"],
        # ["models/full_training_hover_net_graph_500epochs_2layer_sage_16h.ckpt", "2-16"],
        # ["models/full_training_hover_net_graph_500epochs_3layer_sage_16h.ckpt", "3-16"],
        # ["models/full_training_hover_net_graph_500epochs_4layer_sage_16h.ckpt", "4-16"],
        # ["models/full_training_hover_net_graph_500epochs_5layer_sage_16h.ckpt", "5-16"],
        # ["models/full_training_hover_net_graph_500epochs_1layer_sage_32h.ckpt", "1-32"],
        # ["models/full_training_hover_net_graph_500epochs_2layer_sage_32h.ckpt", "2-32"],
        # ["models/full_training_hover_net_graph_500epochs_3layer_sage_32h.ckpt", "3-32"],
        # ["models/full_training_hover_net_graph_500epochs_4layer_sage_32h.ckpt", "4-32"],
        # ["models/full_training_hover_net_graph_500epochs_5layer_sage_32h.ckpt", "5-32"],
        # ["models/full_training_hover_net_graph_500epochs_1layer_sage_64h.ckpt", "1-64"],
        # ["models/full_training_hover_net_graph_500epochs_2layer_sage_64h.ckpt", "2-64"],
        # ["models/full_training_hover_net_graph_500epochs_3layer_sage_64h.ckpt", "3-64"],
        # ["models/full_training_hover_net_graph_500epochs_4layer_sage_64h.ckpt", "4-64"],
        # ["models/full_training_hover_net_graph_500epochs_5layer_sage_64h.ckpt", "5-64"],
        ["models/full_training_hover_net_lizard_graph_500epochs_2layer_sage_4h.ckpt", "2-4"],
        ["models/full_training_hover_net_lizard_graph_500epochs_4layer_sage_4h.ckpt", "4-4"],
        ["models/full_training_hover_net_lizard_graph_500epochs_8layer_sage_4h.ckpt", "8-4"],
        ["models/full_training_hover_net_lizard_graph_500epochs_16layer_sage_4h.ckpt", "16-4"],
        ["models/full_training_hover_net_lizard_graph_500epochs_2layer_sage_8h.ckpt", "2-8"],
        ["models/full_training_hover_net_lizard_graph_500epochs_4layer_sage_8h.ckpt", "4-8"],
        ["models/full_training_hover_net_lizard_graph_500epochs_8layer_sage_8h.ckpt", "8-8"],
        ["models/full_training_hover_net_lizard_graph_500epochs_16layer_sage_8h.ckpt", "16-8"],
        ["models/full_training_hover_net_lizard_graph_500epochs_2layer_sage_16h.ckpt", "2-16"],
        ["models/full_training_hover_net_lizard_graph_500epochs_4layer_sage_16h.ckpt", "4-16"],
        ["models/full_training_hover_net_lizard_graph_500epochs_8layer_sage_16h.ckpt", "8-16"],
        ["models/full_training_hover_net_lizard_graph_500epochs_16layer_sage_16h.ckpt", "16-16"],
        ["models/full_training_hover_net_lizard_graph_500epochs_2layer_sage_32h.ckpt", "2-32"],
        ["models/full_training_hover_net_lizard_graph_500epochs_4layer_sage_32h.ckpt", "4-32"],
        ["models/full_training_hover_net_lizard_graph_500epochs_8layer_sage_32h.ckpt", "8-32"],
        ["models/full_training_hover_net_lizard_graph_500epochs_16layer_sage_32h.ckpt", "16-32"],
        ["models/full_training_hover_net_lizard_graph_500epochs_2layer_sage_64h.ckpt", "2-64"],
        ["models/full_training_hover_net_lizard_graph_500epochs_4layer_sage_64h.ckpt", "4-64"],
        ["models/full_training_hover_net_lizard_graph_500epochs_8layer_sage_64h.ckpt", "8-64"],
        ["models/full_training_hover_net_lizard_graph_500epochs_16layer_sage_64h.ckpt", "16-64"],
        ["models/full_training_hover_net_lizard_graph_500epochs_2layer_sage_128h.ckpt", "2-128"],
        ["models/full_training_hover_net_lizard_graph_500epochs_4layer_sage_128h.ckpt", "4-128"],
        ["models/full_training_hover_net_lizard_graph_500epochs_8layer_sage_128h.ckpt", "8-128"],
        ["models/full_training_hover_net_lizard_graph_500epochs_16layer_sage_128h.ckpt", "16-128"],
    ]

    data_path = "hover_net/Lizard_test_out/mat/"
    file_list = list(Path(data_path).glob("*.mat"))

    for model_path, save_folder in model_list:

        dim_h = int(save_folder.split("-")[1])
        num_layers = int(save_folder.split("-")[0])

        model = init_graph_inference(model_path, dim_h, num_layers)
        model.to(device)

        save_path = f"hover_net/Lizard_test_out/graph/{save_folder}/"
        Path(save_path).mkdir(parents=True, exist_ok=True)


        for file_path in file_list:
            base_name = file_path.stem
            pred = sio.loadmat(file_path)

            points = pred["inst_centroid"]
            predicted_class = pred["inst_type"]
            # graph predicted mask
            graph = get_graph_for_inference(points,predicted_class, 45)
            with torch.no_grad():
                out = (
                        model(
                            graph["x"].to(device),
                            graph["edge_index"].to(device),
                        )
                    ) 
                graph_pred = out.argmax(-1)
            pred["inst_type"] = graph_pred.cpu().numpy()
            pred["inst_type"] = np.reshape(pred["inst_type"], (pred["inst_type"].shape[0], 1))

            sio.savemat(f"{save_path}{base_name}.mat", pred)