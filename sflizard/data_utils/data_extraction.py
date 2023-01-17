"""Copyright (C) SquareFactory SA - All Rights Reserved.

This source code is protected under international copyright law. All rights 
reserved and protected by the copyright holders.
This file is confidential and only available to authorized individuals with the
permission of the copyright holders. If you encounter this file and do not have
permission, please contact the copyright holders and delete this file.
"""
import argparse
import glob
import pickle
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image
from rich import print
from tqdm import tqdm

SEED = 303
TRAIN_TEST_SPLIT = 0.9
IMAGE_EXTENSION = "png"
OUTPUT_BASE_NAME = "data_0.9_split"
PATCH_SIZE = 540
PATCH_STEP = 200
SEPARATE_SAVE = True


def extract_annotation_patches(annotation_file, annotations, patch_size, patch_step):
    """Extract patches from annotations.

    Args:
        annotation_file (str): path to the annotation file.
        annotations (pd.Dataframe): dataframe of annotations.
        patch_size (int): size of the patches.
        patch_step (int): step between patches.

    Returns:
        annotations (pd.Dataframe): dataframe of annotations with the new patches.

    Raises:
        None.

    """
    # load the annotation file
    mat_file = sio.loadmat(annotation_file)

    # extract patches for inst_map
    name = Path(annotation_file).stem.split(".")[0]
    inst_map = np.array(mat_file["inst_map"])

    inst_map_dict = extract_patches(inst_map, name, patch_size, patch_step)

    # update annotation data for each patches
    for key, value in tqdm(
        inst_map_dict.items(),
        desc="Extracting annotation patches",
        position=1,
        leave=False,
    ):
        patch_id = np.unique(value).tolist()[1:]
        # keep only values present in the patch
        nuclei_id = np.squeeze(mat_file["id"]).tolist()
        classes = np.squeeze(mat_file["class"]).tolist()
        nuclei_in_patch = []
        class_in_patch = []
        centroid_in_patch = []
        class_map = np.zeros(value.shape)
        for v in patch_id:
            idx = nuclei_id.index(v)
            class_i = mat_file["class"][idx]
            centroid_i = mat_file["centroid"][idx].copy()
            centroid_i[0] = min(max(0, centroid_i[0] - int(key.split("_")[3])), patch_size-1)
            centroid_i[1] = min(max(0, centroid_i[1] - int(key.split("_")[2])), patch_size-1)
            nuclei_in_patch.append(v)
            class_in_patch.append(class_i)
            centroid_in_patch.append(centroid_i)
            class_map[value == v] = class_i
        # add the new patch to the dataframe
        annotations = pd.concat(
            [
                annotations,
                pd.DataFrame(
                    {
                        "id": [key],
                        "inst_map": [value],
                        "class_map": [class_map],
                        "nuclei_id": [nuclei_in_patch],
                        "classes": [class_in_patch],
                        "centroid": [centroid_in_patch],
                    }
                ),
            ],
            ignore_index=True,
        )
    return annotations


def extract_image_patches(image_file, patch_size, patch_step):
    """Extract patches from an image.

    Args:
        image_file (str): path to the image file.
        patch_size (int): size of the patches.
        patch_step (int): step between patches.

    Returns:
        images (dict): dictionary of images with the new patches.

    Raises:
        None.

    """
    # load image
    image = Image.open(image_file)
    image = np.array(image)
    # extract patches
    name = Path(image_file).stem.split(".")[0]
    images_dict = extract_patches(image, name, patch_size, patch_step)
    return images_dict


def extract_patches(array, array_name, patch_size, patch_step):
    """Extract patches from an image or an instance map.

    Args:
        array (np.array): array to be patched.
        array_name (str): name of the array.
        patch_size (int): size of the patches.
        patch_step (int): step between patches.

    Returns:
        array_dict (dict): dictionary of array patches.

    Raises:
        None.

    """
    array_dict = {}
    # extract patches
    for i in range(0, array.shape[0], patch_step):
        for j in range(0, array.shape[1], patch_step):
            patch = array[i : i + patch_size, j : j + patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                array_dict[array_name + f"_{i}_{j}"] = patch
    # add missing patches
    if array.shape[0] % patch_step != 0:
        for j in range(0, array.shape[1], patch_step):
            patch = array[-patch_size:, j : j + patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                array_dict[array_name + f"_{array.shape[0] - patch_size}_{j}"] = patch
    if array.shape[1] % patch_step != 0:
        for i in range(0, array.shape[0], patch_step):
            patch = array[i : i + patch_size, -patch_size:]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                array_dict[array_name + f"_{i}_{array.shape[1] - patch_size}"] = patch
    if array.shape[0] % patch_step != 0 and array.shape[1] % patch_step != 0:
        patch = array[-patch_size:, -patch_size:]
        if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
            array_dict[
                array_name
                + f"_{array.shape[0] - patch_size}_{array.shape[1] - patch_size}"
            ] = patch

    return array_dict


def remove_missing_data(images, annotations, set_name):
    """Clean data by removing image without annotations and annotations without images.

    Args:
        images (dict): dictionary of images.
        annotations (pd.DataFrame): dataframe of annotations.
        set_name (str): name of the set.

    Returns:
        images (dict): dictionary of images.
        annotations (pd.DataFrame): dataframe of annotations.

    Raises:
        None.

    """
    # remove images without annotations
    if set(annotations.id) != set(images.keys()):
        missings = set(images.keys()).difference(annotations.id)
        print(f" > Images in {set_name} set with no annotation : {len(missings)}")
        for missing in missings:
            del images[missing]
    # remove annotations without images
    if set(annotations.id) != set(images.keys()):
        missings = set(annotations.id).difference(images.keys())
        print(f" > Annotations in {set_name} with no images : {len(missings)}")
        annotations = annotations[~annotations.id.isin(missings)]
    return images, annotations


def extract_data(args):
    """Extract data from the original dataset folder.

    Args:
        args (argparse.Namespace): arguments from the command line (see list bellow).

    Returns:
        None.

    Raises:
        None.
    """

    print("1. Extracting images from folder...\n")
    start = time.time()
    images_train = {}
    images_valid = {}
    images_test = {}
    image_files = []
    train_list = []
    valid_list = []
    test_list = []
    # Get all the images from the folder list
    for folder in args.images_path:
        # fix folder and extension format
        folder = folder + "/" if folder[-1] != "/" else folder
        imgs_ext = (
            "." + args.images_extension
            if args.images_extension[0] != "."
            else args.images_extension
        )
        # get the list of images in the folder
        image_files += glob.glob(f"{folder}*{imgs_ext}")

    # randomize images order
    random.Random(SEED).shuffle(image_files)

    # determine the number of images for training and testing
    test_size = len(image_files) - int(len(image_files) * args.train_test_split)
    train_size = int((len(image_files) - test_size) * args.train_test_split)

    # extract train patches
    for idx, image_file in enumerate(tqdm(image_files, desc="Extracting images")):
        # extract patches for training
        if idx < train_size:
            # extract patches from the images to have multiple images of same size:
            images_train.update(
                extract_image_patches(image_file, args.patch_size, args.patch_step)
            )
            train_list.append(image_file)
        # extract patches for validation
        elif idx < len(image_files) - test_size:
            # extract patches from the images to have multiple images of same size:
            images_valid.update(
                extract_image_patches(image_file, args.patch_size, args.patch_step)
            )
            valid_list.append(image_file)
        # extract patches for testing
        else:
            # extract patches from the images to have multiple images of same size:
            images_test.update(
                extract_image_patches(image_file, args.patch_size, args.patch_step)
            )
            test_list.append(image_file)

    # save the images lists
    with open(
        "data/Lizard_dataset_extraction/" + args.output_base_name + "_train_list.txt",
        "w",
    ) as f:
        for item in train_list:
            f.write(f"{item}\n")
    with open(
        "data/Lizard_dataset_extraction/" + args.output_base_name + "_valid_list.txt",
        "w",
    ) as f:
        for item in valid_list:
            f.write(f"{item}\n")
    with open(
        "data/Lizard_dataset_extraction/" + args.output_base_name + "_test_list.txt",
        "w",
    ) as f:
        for item in test_list:
            f.write(f"{item}\n")

    print(
        f"\nAll {len(images_train) + len(images_valid) + len(images_test)} images extracted! (in {time.time() - start:.2f} secs)\n"
    )

    print("2. Extracting annotations from folder...\n")
    start = time.time()

    annotations = pd.DataFrame(
        columns=[
            "id",
            "inst_map",
            "class_map",
            "nuclei_id",
            "classes",
            "centroid",
        ]
    )
    # Get all the annotations from the folder list
    for folder in args.matlab_file:
        # fix folder and extension format
        folder = folder + "/" if folder[-1] != "/" else folder
        # get the list of masks in the folder
        annotation_files = glob.glob(f"{folder}*.mat")
        for annotation_file in tqdm(
            annotation_files,
            desc=f"Extracting annotations from folder {folder}",
            position=0,
        ):
            # extract patches from the annotations to correspond to the images patches of same size:
            annotations = extract_annotation_patches(
                annotation_file, annotations, args.patch_size, args.patch_step
            )

    print(f"\nAll annotations extracted! (in {time.time() - start:.2f} secs)\n")

    print("3. Cleaning...\n")
    start = time.time()
    cleaned_train_data = {}
    cleaned_valid_data = {}
    cleaned_test_data = {}

    # split annotations
    annotations_train = annotations[annotations["id"].isin(images_train.keys())]
    annotations_valid = annotations[annotations["id"].isin(images_valid.keys())]
    annotations_test = annotations[annotations["id"].isin(images_test.keys())]

    # clean missing data
    images_train, annotations_train = remove_missing_data(
        images_train, annotations_train, "train"
    )
    images_valid, annotations_valid = remove_missing_data(
        images_valid, annotations_valid, "valid"
    )
    images_test, annotations_test = remove_missing_data(
        images_test, annotations_test, "test"
    )

    cleaned_train_data["images"] = images_train
    cleaned_train_data["annotations"] = annotations_train
    cleaned_valid_data["images"] = images_valid
    cleaned_valid_data["annotations"] = annotations_valid
    cleaned_test_data["images"] = images_test
    cleaned_test_data["annotations"] = annotations_test

    print(f"\nFiles cleaned! (in {time.time() - start:.2f} secs)\n")

    print("4. Report\n")

    print(
        f" > Number of images : {len(cleaned_train_data['images']) + len(cleaned_valid_data['images']) + len(cleaned_test_data['images'])}"
    )
    print(f" > Number of images in train set: {len(cleaned_train_data['images'])}")
    print(f" > Number of images in valid set: {len(cleaned_valid_data['images'])}")
    print(f" > Number of images in test set: {len(cleaned_test_data['images'])}")

    print("5. Saving File...\n")

    if SEPARATE_SAVE:
        save_images_train = (
            "data/Lizard_dataset_split/patches/Lizard_Images_train"
        )
        save_images_valid = (
            "data/Lizard_dataset_split/patches/Lizard_Images_valid"
        )
        save_images_test = "data/Lizard_dataset_split/patches/Lizard_Images_test"
        for path in [save_images_train, save_images_valid, save_images_test]:
            Path(path).mkdir(parents=True, exist_ok=True)
        for img in cleaned_train_data["images"]:
            im = Image.fromarray(cleaned_train_data["images"][img])
            im.save(save_images_train + f"/{img}.png")
        for img in cleaned_valid_data["images"]:
            im = Image.fromarray(cleaned_valid_data["images"][img])
            im.save(save_images_valid + f"/{img}.png")
        for img in cleaned_test_data["images"]:
            im = Image.fromarray(cleaned_test_data["images"][img])
            im.save(save_images_test + f"/{img}.png")
        print(f"Images saved in {save_images_train}, {save_images_valid} and {save_images_test} folders")

        save_train = "data/Lizard_dataset_split/patches/Lizard_Labels_train"
        save_valid = "data/Lizard_dataset_split/patches/Lizard_Labels_valid"
        save_test = "data/Lizard_dataset_split/patches/Lizard_Labels_test"
        for path in [save_train, save_valid, save_test]:
            Path(path).mkdir(parents=True, exist_ok=True)
        # save back to mat files
        for idx in cleaned_train_data["images"]:
            dic = cleaned_train_data["annotations"][cleaned_train_data["annotations"]["id"] == idx].to_dict(orient="list")
            for key in dic:
                dic[key] = dic[key][0]
            sio.savemat(
                save_train + f"/{idx}.mat",
                dic,
            )
        for idx in cleaned_valid_data["images"]:
            dic = cleaned_valid_data["annotations"][cleaned_valid_data["annotations"]["id"] == idx].to_dict(orient="list")
            for key in dic:
                dic[key] = dic[key][0]
            sio.savemat(
                save_valid + f"/{idx}.mat",
                dic,
            )
        for idx in cleaned_test_data["images"]:
            dic = cleaned_test_data["annotations"][cleaned_test_data["annotations"]["id"]  == idx].to_dict(orient="list")
            for key in dic:
                dic[key] = dic[key][0]
            sio.savemat(
                save_test + f"/{idx}.mat",
                dic,
            )
        print(f"Annotations saved in {save_train}, {save_valid} and {save_test} folders")
    else:
        save_train = (
            "data/Lizard_dataset_extraction/" + args.output_base_name + "_train.pkl"
        )
        save_valid = (
            "data/Lizard_dataset_extraction/" + args.output_base_name + "_valid.pkl"
        )
        save_test = "data/Lizard_dataset_extraction/" + args.output_base_name + "_test.pkl"
        with open(save_train, "wb") as f:
            pickle.dump(cleaned_train_data, f)
        print(f"Cleaned train data saved in {save_train}")
        with open(save_valid, "wb") as f:
            pickle.dump(cleaned_valid_data, f)
        print(f"Cleaned valid data saved in {save_valid}")
        with open(save_test, "wb") as f:
            pickle.dump(cleaned_test_data, f)
        print(f"Cleaned test data saved in {save_test}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ip",
        "--images_path",
        type=str,
        nargs="+",
        help="Path to the folder containing images.",
    )
    parser.add_argument(
        "-ie",
        "--images_extension",
        type=str,
        default=IMAGE_EXTENSION,
        help="Extension of the images.",
    )
    parser.add_argument(
        "-mf",
        "--matlab_file",
        type=str,
        nargs="+",
        help=" Path to the .mat file containing the annotations.",
    )
    parser.add_argument(
        "-of",
        "--output_base_name",
        type=str,
        help="path for the output file (must end in .pkl).",
        default=OUTPUT_BASE_NAME,
    )
    parser.add_argument(
        "-dsr",
        "--train_test_split",
        type=float,
        help="Ratio of the dataset to be used for training.",
        default=TRAIN_TEST_SPLIT,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="Seed for the random split.",
        default=SEED,
    )

    # arguments for patches extraction
    parser.add_argument(
        "--patch_size",
        type=int,
        default=PATCH_SIZE,
        help="Size of the window to extract patches from the images.",
    )
    parser.add_argument(
        "--patch_step",
        type=int,
        default=PATCH_STEP,
        help="Step size to extract patches from the images.",
    )

    args = parser.parse_args()

    extract_data(args)
