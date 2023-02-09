import shutil
import subprocess
from pathlib import Path

TEST_MODELS = 3
# 0: CoNSeP HoverNet vs HoverNet+Graph
# 1: Lizard HoverNet vs HoverNet+Graph
# 2: Lizard Stardist vs Stardist+Graph
# 3: Lizard HoverNet vs Stardist+Graph


# CoNSeP HoverNet vs HoverNet+Graph
if TEST_MODELS == 0:
    MODEL_1_MAT_FOLDER = "../external_models/output/CoNSeP_test_out/mat/"
    MODEL_2_MAT_FOLDER = "../external_models/output/CoNSeP_test_out/graph/2-16/"
    TRUE_DATA_PATH_START = "../data/CoNSeP/Test/Labels/"
    sub_folders = ["0", "1", "2", "3", "4"]

# Lizard HoverNet vs HoverNet+Graph
elif TEST_MODELS == 1:
    MODEL_1_MAT_FOLDER = "../external_models/output/Lizard_test_out/mat/"
    MODEL_2_MAT_FOLDER = (
        "../external_models/output/Lizard_test_out/graph/2-64-acc-macro-1/"
    )
    TRUE_DATA_PATH_START = "../data/Lizard_dataset_split/patches/Lizard_Labels_test/"
    sub_folders = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Lizard Stardist vs Stardist+Graph
elif TEST_MODELS == 2:
    MODEL_1_MAT_FOLDER = "../output/graph/manual/stardist/"
    MODEL_2_MAT_FOLDER = "../output/graph/manual/mod1-b-0.0-t/"
    TRUE_DATA_PATH_START = "../data/Lizard_dataset_split/patches/Lizard_Labels_test/"
    sub_folders = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Lizard HoverNet vs Stardist+Graph
elif TEST_MODELS == 3:
    MODEL_1_MAT_FOLDER = "../external_models/output/Lizard_test_out/mat/"
    MODEL_2_MAT_FOLDER = "../output/graph/manual/mod1-b-0.0-t/"
    TRUE_DATA_PATH_START = "../data/Lizard_dataset_split/patches/Lizard_Labels_test/"
    sub_folders = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

model_1_mat = list(Path(MODEL_1_MAT_FOLDER).glob("*.mat"))
model_2_mat = list(Path(MODEL_2_MAT_FOLDER).glob("*.mat"))
# true_mat = list(Path(TRUE_DATA_PATH_START).glob("*.mat"))

print(len(model_1_mat), len(model_2_mat))

# create temporary folder
for folder in ["model1", "model2"]:
    for sub_folder in sub_folders:
        Path(f"temp/{folder}/{sub_folder}").mkdir(parents=True, exist_ok=True)

for idx, (mat1, mat2) in enumerate(zip(model_1_mat, model_2_mat)):
    # copy mat files to temp folder
    shutil.copy(mat1, f"temp/model1/{sub_folders[idx%len(sub_folders)]}/{mat1.name}")
    shutil.copy(mat2, f"temp/model2/{sub_folders[idx%len(sub_folders)]}/{mat2.name}")


def run_hovernet_metric_tool(save_folder: str) -> str:
    compute_stat_cmd = f"python ../external_models/hover_net/compute_stats.py --pred_dir {save_folder} --true_dir {TRUE_DATA_PATH_START} --mode type"
    command = f"conda activate hovernet; {compute_stat_cmd}; conda activate TM"
    ret = subprocess.run(command, capture_output=True, shell=True)  # nosec
    result = ret.stdout.decode()
    return result


results = {}

for sub_folder in sub_folders:
    print(f"Processing {sub_folder}")
    result1 = run_hovernet_metric_tool(f"temp/model1/{sub_folder}")
    result2 = run_hovernet_metric_tool(f"temp/model2/{sub_folder}")
    # print(result1)
    # print(result2)
    result1 = float(result1.split("\n")[1].split("  ")[1])
    result2 = float(result2.split("\n")[1].split("  ")[1])
    results[sub_folder] = (result1, result2)

print("result 1")
for sub_folder in sub_folders:
    print(results[sub_folder][0])
print("result 2")
for sub_folder in sub_folders:
    print(results[sub_folder][1])

# clean temp folder
shutil.rmtree("temp")
