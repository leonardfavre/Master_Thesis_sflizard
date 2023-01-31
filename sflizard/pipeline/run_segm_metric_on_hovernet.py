import glob

import numpy as np
import scipy.io as sio
from rich.console import Console
from tqdm import tqdm

from sflizard import SegmentationMetricTool


def get_cm(mat, class_k, inst_map_k, nuclei_k):
    ann_inst = mat[inst_map_k]
    nuclei_id = np.squeeze(mat[nuclei_k]).tolist()
    if type(nuclei_id) != list:
        nuclei_id = [nuclei_id]
    patch_id = np.unique(ann_inst).tolist()[1:]
    if len([t for t in patch_id if t not in nuclei_id]) > 0:
        print(
            "ERROR: nuclei_id not in patch_id: %s"
            % ([t for t in patch_id if t not in nuclei_id])
        )
    type_map = np.zeros(ann_inst.shape)
    for v in patch_id:
        if v not in nuclei_id:
            type_map[ann_inst == v] = 0
        else:
            idn = nuclei_id.index(v)
            type_map[ann_inst == v] = mat[class_k][idn]
    type_map = type_map.astype("int32")
    return type_map


true_dir = "data/Lizard_dataset_split/patches/Lizard_Labels_test/"
pred_dir = (
    "external_models/output/Lizard_test_out/mat/"  # "output/graph/manual/mod1-test/"
)
smt = SegmentationMetricTool(n_classes=7, device="cuda")

file_list = glob.glob("%s/*mat" % (pred_dir))

for idx, file in tqdm(enumerate(file_list)):
    mat_pred = sio.loadmat(file)
    mat_true = sio.loadmat(file.replace(pred_dir, true_dir))

    smt.add_batch(
        idx,
        np.expand_dims(mat_true["inst_map"], axis=0),
        np.expand_dims(mat_pred["inst_map"], axis=0),
    )

    smt.add_batch_class(
        idx,
        np.expand_dims(get_cm(mat_true, "classes", "inst_map", "nuclei_id"), axis=0),
        np.expand_dims(get_cm(mat_pred, "inst_type", "inst_map", "inst_uid"), axis=0),
    )

smt.compute_metrics()
console = Console()
smt.log_results(console)
