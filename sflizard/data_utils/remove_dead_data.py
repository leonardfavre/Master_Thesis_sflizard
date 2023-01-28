import pickle
from pathlib import Path

TRAIN_DATA_PATH = "data/Lizard_dataset_extraction/data_0.9_split_train.pkl"
VALID_DATA_PATH = "data/Lizard_dataset_extraction/data_0.9_split_valid.pkl"
TEST_DATA_PATH = "data/Lizard_dataset_extraction/data_0.9_split_test.pkl"

TARGET = TRAIN_DATA_PATH

DEAD_IMAGES = [
    "crag_27_0_960",
    "crag_27_200_960",
    "crag_27_400_960",
    "crag_27_600_960",
    "crag_27_800_960",
    "crag_27_976_960",
    "crag_34_400_0",
    "crag_34_600_0",
    "crag_34_600_200",
    "crag_34_600_400",
    "crag_34_800_0",
    "crag_34_800_200",
    "crag_34_800_400",
    "crag_34_976_0",
    "crag_34_976_200",
    "crag_34_976_400",
    "crag_34_976_600",
    "crag_28_976_600",
    "crag_54_976_0",
    "crag_25_400_200",
    "crag_25_400_400",
    "crag_25_600_0",
    "crag_25_600_200",
    "crag_25_600_400",
    "crag_25_600_600",
    "crag_25_600_800",
    "crag_25_800_0",
    "crag_25_800_200",
    "crag_25_800_400",
    "crag_25_800_600",
    "crag_25_800_800",
    "crag_25_976_0",
    "crag_25_976_200",
    "crag_25_976_400",
    "crag_25_976_600",
    "crag_25_976_800",
    "crag_25_800_960",
    "crag_25_976_960",
]


def remove_dead_data() -> None:
    """Remove dead data from the dataset."""

    data_path = Path(TARGET)
    with data_path.open("rb") as f:
        data = pickle.load(f)

    images = data["images"]
    to_remove = []
    for d in images.keys():
        if d in DEAD_IMAGES:
            to_remove.append(d)
    for d in to_remove:
        del data["images"][d]

    df = data["annotations"]
    df = df[~df.id.isin(DEAD_IMAGES)]

    clean_data = {}
    clean_data["images"] = images
    clean_data["annotations"] = df

    with open(TARGET, "wb") as f:
        pickle.dump(clean_data, f)


if __name__ == "__main__":
    remove_dead_data()
