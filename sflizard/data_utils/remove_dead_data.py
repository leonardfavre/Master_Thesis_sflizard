import pickle
from pathlib import Path

TRAIN_DATA_PATH = "data/Lizard_dataset_extraction/data_0.9_split_train.pkl"
VALID_DATA_PATH = "data/Lizard_dataset_extraction/data_0.9_split_valid.pkl"
TEST_DATA_PATH = "data/Lizard_dataset_extraction/data_0.9_split_test.pkl"

TARGET = VALID_DATA_PATH

DEAD_IMAGES = [
    "crag_63_200_0",
    "crag_63_0_400",
    "crag_63_0_200",
    "crag_63_0_0",
    "crag_43_976_960",
    "crag_43_800_960",
    "crag_43_600_960",
    "crag_43_976_800",
    "crag_43_800_800",
]



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