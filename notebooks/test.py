
import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path
import torch
from sflizard import LizardGraphDataModule, get_graph
from tqdm import tqdm

STARDIST_CHECKPOINT = "models/final3_stardist_crop-cosine_200epochs_1.0losspower_0.0005lr.ckpt"
CHECKPOINT_PATH = ["models/", "models/cp_acc_graph/", "models/loss_cb_graph/"]

def test1():

    # get the train data
    train_data_path = Path("data/Lizard_dataset_extraction/data_0.9_split_train.pkl")
    with train_data_path.open("rb") as f:
        train_data = pickle.load(f)
    # train_data = train_data["annotations"]

    # get the valid data
    valid_data_path = Path("data/Lizard_dataset_extraction/data_0.9_split_valid.pkl")
    with valid_data_path.open("rb") as f:
        valid_data = pickle.load(f)
        # valid_data = valid_data["annotations"]

    # get the test data
    test_data_path = Path("data/Lizard_dataset_extraction/data_0.9_split_test.pkl")
    with test_data_path.open("rb") as f:
        test_data = pickle.load(f)
        # test_data = test_data["annotations"]

    # dm = LizardGraphDataModule(
    #     train_data=train_data,
    #     valid_data=valid_data,
    #     test_data=test_data,
    #     batch_size=1,
    #     num_workers=8,
    #     seed=303,
    #     stardist_checkpoint=STARDIST_CHECKPOINT,
    #     x_type="4ll+c",
    #     distance=45,
    # )
    # dm.setup()

    # dataloader = iter(dm.train_dataloader())

    # batch = next(dataloader)

    # print(batch[0].y)

    train_df = train_data["annotations"]
    images = train_data["images"]

    image = (
        torch.tensor(images[train_df.iloc[0].id]).permute(2, 0, 1)
        if images is not None
        else None
    )
    print("image", image is not None)
    class_map = (
        train_df.iloc[0].class_map if "class_map" in train_df.columns else None
    )
    print("class_map", class_map is not None)
    inst_map = (
        train_df.iloc[0].inst_map if "inst_map" in train_df.columns else None
    )
    print("inst_map", inst_map is not None)
    predicted_classes = (
        train_df.iloc[0].predicted_classes
        if "predicted_classes" in train_df.columns
        else None
    )
    print("predicted_classes", predicted_classes is not None)
    points = train_df.iloc[0].points if "points" in train_df.columns else None
    print("points", points is not None)

    graph = get_graph(
        inst_map=inst_map,
        points=points,
        predicted_classes=predicted_classes,
        true_class_map=class_map,
        n_rays=32,
        distance=45,
        stardist_checkpoint="models/final3_stardist_crop-cosine_200epochs_1.0losspower_0.0005lr.ckpt",
        image=image,
        x_type="4ll+c",
        consep_data=False,
    )

    print(graph["x"].shape)
    print(graph["y"].shape)
    print(graph["y"])

def test2():
    for path in CHECKPOINT_PATH:
        available_checkpoints = list(Path(path).glob("*.ckpt"))
        for checkpoint_path in tqdm(available_checkpoints):
            checkpoint = torch.load(checkpoint_path)
            if "hyper_parameters" in checkpoint.keys():
                if "model" in checkpoint["hyper_parameters"] and checkpoint["hyper_parameters"]["model"] == "graph_gat":
                    path_list = str(checkpoint_path).split("-")
                    path_list.insert(5, str(checkpoint["hyper_parameters"]["heads"]))
                    new_path = "-".join(path_list)
                    os.rename(checkpoint_path, new_path)


test2()