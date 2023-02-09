import glob

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from rich.console import Console
from tqdm import tqdm

from sflizard import ReportGenerator, SegmentationMetricTool, get_class_color


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
img_dir = "data/Lizard_dataset_split/patches/Lizard_Images_test/"
pred_dir = (
    "external_models/output/Lizard_test_out/mat/"  # "output/graph/manual/mod1-test/"
)
console = Console()
smt = SegmentationMetricTool(n_classes=7, device="cuda", console=console)
report_generator = ReportGenerator(
    "./output/hovernet_pipeline/final_result/", 30, n_classes=7, console=console
)

file_list = glob.glob("%s/*mat" % (pred_dir))

color = get_class_color()

for idx, file in tqdm(enumerate(file_list)):
    mat_pred = sio.loadmat(file)
    mat_true = sio.loadmat(file.replace(pred_dir, true_dir))

    true_masks = np.expand_dims(mat_true["inst_map"], axis=0)
    pred_masks = np.expand_dims(mat_pred["inst_map"], axis=0)
    true_class_map = np.expand_dims(
        get_cm(mat_true, "classes", "inst_map", "nuclei_id"), axis=0
    )
    pred_class_map = np.expand_dims(
        get_cm(mat_pred, "inst_type", "inst_map", "inst_uid"), axis=0
    )

    # Save image
    # file_name = file.split("/")[-1].replace(".mat", "")
    # plt.imsave(f"./output/hovernet_pipeline/final_result/images/{file_name}.png", pred_class_map[0], cmap=ListedColormap(color))

    image = Image.open(file.replace(pred_dir, img_dir).replace(".mat", ".png"))
    images = [torch.Tensor(np.array(image).transpose(2, 0, 1))]

    smt.add_batch(
        idx,
        true_masks,
        pred_masks,
    )

    smt.add_batch_class(
        idx,
        true_class_map,
        pred_class_map,
    )
    report_generator.add_batch(
        images=images,
        true_masks=true_masks,
        pred_masks=pred_masks,
        true_class_map=true_class_map,
        pred_class_map=pred_class_map,
        pred_class_map_improved=pred_class_map,
    )

smt.compute_metrics()

smt.log_results()

report_generator.add_final_metrics(
    smt.seg_metrics[0],
    smt.seg_metrics,
    None,
    None,
    None,
    None,
    None,
)

report_generator.generate_md()
