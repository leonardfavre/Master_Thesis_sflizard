"""Copyright (C) SquareFactory SA - All Rights Reserved.

This source code is protected under international copyright law. All rights 
reserved and protected by the copyright holders.
This file is confidential and only available to authorized individuals with the
permission of the copyright holders. If you encounter this file and do not have
permission, please contact the copyright holders and delete this file.
"""
from rich import print
import glob
import argparse
import time
from PIL import Image
from pathlib import Path
import scipy.io as sio
import numpy as np
import pickle


def extract_data(args):
    """Extract data from the original dataset folder.
    
    Args:
    
    
    Returns:
    
    
    Raises:
    
    
    """

    
    print("1. Extracting images from folder...\n")
    start = time.time()
    images = {}
    # Get all the images from the folder list
    for folder in args.images_path:
        # fix folder and extension format
        folder = folder + "/" if folder[-1] != "/" else folder
        imgs_ext = "." + args.images_extension if args.images_extension[0] != "." else args.images_extension
        # get the list of images in the folder
        image_files = glob.glob(f"{folder}*{imgs_ext}")
        for image_file in image_files:
            # load the images with its name as key (without extension)
            images[Path(image_file).stem.split('.')[0]] = Image.open(image_file)

    print(f"\nAll images extracted! (in {time.time() - start:.2f} secs)\n")

    print("2. Extracting annotations from folder...\n")
    start = time.time()
    annotations = {}
    # Get all the annotations from the folder list
    for folder in args.matlab_file:
        # fix folder and extension format
        folder = folder + "/" if folder[-1] != "/" else folder
        # get the list of masks in the folder
        annotation_files = glob.glob(f"{folder}*.mat")
        for annotation_file in annotation_files:
            # load the annotation file
            mat_file = sio.loadmat(annotation_file)
            ann_inst = mat_file["inst_map"]
            class_list = mat_file["class"]
            # create ann_type by replacing id in ann_inst with class_list
            ann_type = np.zeros_like(ann_inst)
            for i in range(len(class_list)):
                ann_type[ann_inst == i] = class_list[i]
            # save annotation with its name as key (without extension)
            annotations[Path(annotation_file).stem.split('.')[0]] = np.dstack([ann_inst, ann_type]).astype("int32")
    
    print(f"\nAll annotations extracted! (in {time.time() - start:.2f} secs)\n")

    print("3. Cleaning...\n")
    start = time.time()
    cleaned_data = {}
    # remove images with no annotations
    if set(annotations.keys()) != set(images.keys()):
        missings = set(images.keys()).difference(annotations.keys())
        print(f" > Images with no annotation : {len(missings)}")
        for missing in missings:
            del images[missing]
    # remove annotations without images
    if set(annotations.keys()) != set(images.keys()):
        missings = set(annotations.keys()).difference(images.keys())
        print(f" > annotations with no images : {len(missings)}")
        for missing in missings:
            del annotations[missing]
    cleaned_data["images"] = images
    cleaned_data["annotations"] = annotations

    print(f"\nFiles cleaned! (in {time.time() - start:.2f} secs)\n")

    print("4. Report\n")

    print(f" > Number of images : {len(cleaned_data['images'])}")
    print(f" > Number of nuclei : {np.sum([np.max(d[:,:,0]) for d in cleaned_data['annotations'].values()])}")
    print(f" > Number of classes : {len(np.unique(np.concatenate([np.unique(d[:,:,1]) for d in cleaned_data['annotations'].values()])))}")

    save = args.output_file[0]
    with open(save, "wb") as f:
        pickle.dump(cleaned_data, f)
    print(f"Cleaned data saved in {save}")

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
        nargs="+",
        help="path for the output file (must end in .pkl).",
        default=["data.pkl"],
    )

    args = parser.parse_args()

    extract_data(args)