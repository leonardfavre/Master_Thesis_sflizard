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
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from PIL import Image
from rich import print
from tqdm import tqdm


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
        classes = []
        # bbox = []
        # centroids = []
        class_map = np.zeros(value.shape)
        for v in patch_id:
            idx = nuclei_id.index(v)
            # classes.append(mat_file["class"][idx])
            # bbox.append(mat_file["bbox"][idx])
            # centroids.append(mat_file["centroid"][idx])
            class_map[value == v] = mat_file["class"][idx]
        # add the new patch to the dataframe
        annotations = pd.concat(
            [
                annotations,
                pd.DataFrame(
                    {
                        "id": [key],
                        "inst_map": [value],
                        "class_map": [class_map],
                        "nuclei_id": [nuclei_id],
                        "classes": [classes],
                        # "bbox": [bbox],
                        # "centroids": [centroids],
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
    """Extract patches from an image.

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


def extract_data(args):
    """Extract data from the original dataset folder.

    Args:


    Returns:


    Raises:


    """

    print("1. Extracting images from folder...\n")
    start = time.time()
    images_train = {}
    images_test = {}
    image_files = []
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

    # determine the number of images for training and testing
    if args.train_test_split is None:
        train_size = int(len(image_files) * 0.8)
        test_size = len(image_files) - train_size
    else:
        train_size = int(len(image_files) * args.train_test_split)
        test_size = len(image_files) - train_size

    # extract train patches
    for idx, image_file in enumerate(tqdm(
        image_files, desc="Extracting images"
    )):
        # extract patches for training
        if idx < train_size:
            # extract patches from the images to have multiple images of same size:
            images_train.update(
                extract_image_patches(image_file, args.patch_size, args.patch_step)
            )
        # extract patches for testing
        else:
            # extract patches from the images to have multiple images of same size:
            images_test.update(
                extract_image_patches(image_file, args.patch_size, args.patch_step)
            )
    

    print(
        f"\nAll {len(images_train) + len(images_test)} images extracted! (in {time.time() - start:.2f} secs)\n"
    )

    print("2. Extracting annotations from folder...\n")
    start = time.time()

    annotations = pd.DataFrame(
        columns=[
            "id",
            "inst_map",
            "class_map",
            # "nuclei_id",
            # "classes",
            # "bboxs",
            # "centroids",
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
    cleaned_test_data = {}

    # split annotations
    annotations_train = annotations[annotations["id"].isin(images_train.keys())]
    annotations_test = annotations[annotations["id"].isin(images_test.keys())]

    # remove images with no annotations
    images_train, annotations_train = remove_missing_data(
        images_train, annotations_train, "train"
    )
    images_test, annotations_test = remove_missing_data(
        images_test, annotations_test, "test"
    )
    
    cleaned_train_data["images"] = images_train
    cleaned_train_data["annotations"] = annotations_train
    cleaned_test_data["images"] = images_test
    cleaned_test_data["annotations"] = annotations_test

    print(f"\nFiles cleaned! (in {time.time() - start:.2f} secs)\n")

    print("4. Report\n")

    print(f" > Number of images : {len(cleaned_train_data['images']) + len(cleaned_test_data['images'])}")
    print(f" > Number of images in train/validation set: {len(cleaned_train_data['images'])}")
    print(f" > Number of images in test set: {len(cleaned_test_data['images'])}")
    # print(
    #     f" > Number of nuclei : {np.sum([np.max(d) if len(d) > 0 else 0 for d in cleaned_data['annotations']['nuclei_id']])}"
    # )
    # print(
    #     f" > Number of classes : {len(np.unique(np.concatenate([d for d in cleaned_data['annotations']['classes']])))}"
    # )

    print("5. Saving File...\n")
    save_train = args.output_file + "_train.pkl"
    save_test = args.output_file + "_test.pkl"
    with open(save_train, "wb") as f:
        pickle.dump(cleaned_train_data, f)
    print(f"Cleaned train data saved in {save_train}")
    with open(save_test, "wb") as f:
        pickle.dump(cleaned_test_data, f)
    print(f"Cleaned test data saved in {save_test}")

def remove_missing_data(images, annotations, set_name):
    """Clean data.

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
        default="png",
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
        "--output_file",
        type=str,
        help="path for the output file (must end in .pkl).",
        default="data",
    )
    parser.add_argument(
        "-dsr",
        "--train_test_split",
        type=float,
        help="Ratio of the dataset to be used for training.",
        default=0.8,
    )

    # arguments for patches extraction
    parser.add_argument(
        "--patch_size",
        type=int,
        default=540,
        help="Size of the window to extract patches from the images.",
    )
    parser.add_argument(
        "--patch_step",
        type=int,
        default=200,
        help="Step size to extract patches from the images.",
    )

    args = parser.parse_args()

    extract_data(args)
