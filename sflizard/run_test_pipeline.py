import argparse

from sflizard import TestPipeline

VALID_DATA_PATH = "data/Lizard_dataset_extraction/data_final_split_valid.pkl"
TEST_DATA_PATH = "data/Lizard_dataset_extraction/data_final_split_test.pkl"
STARDIST_WEIGHTS_PATH = (
    "weights/final3_stardist_crop-cosine_200epochs_1.0losspower_0.0005lr.ckpt"
)
GRAPH_WEIGHTS_PATH = "weights/graph_custom-1024-4-4ll-45-0-0-3-16-wide-0.0005-acc-epoch=103-val_acc=0.7817.ckpt"
N_RAYS = 32
N_CLASSES = 7
BATCH_SIZE = 1
SEED = 303
OUTPUT_DIR = "./output/stardist_pipeline/final_result/"
IMGS_TO_DISPLAY = 30

DISTANCE = 45
MODE = "test"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-vdp",
        "--valid_data_path",
        type=str,
        default=VALID_DATA_PATH,
        help="Path to the .pkl file containing the data.",
    )
    parser.add_argument(
        "-tdp",
        "--test_data_path",
        type=str,
        default=TEST_DATA_PATH,
        help="Path to the .pkl file containing the test data.",
    )
    parser.add_argument(
        "-swp",
        "--stardist_weights_path",
        type=str,
        default=STARDIST_WEIGHTS_PATH,
        help="Path to the file containing the stardist model weights.",
    )
    parser.add_argument(
        "-gwp",
        "--graph_weights_path",
        type=str,
        nargs="+",
        default=GRAPH_WEIGHTS_PATH,
        help="Path to the file containing the graph model weights.",
    )
    parser.add_argument(
        "-nr",
        "--n_rays",
        type=int,
        default=N_RAYS,
        help="Number of rays to use in the stardist model.",
    )
    parser.add_argument(
        "-nc",
        "--n_classes",
        type=int,
        default=N_CLASSES,
        help="Number of classes to use in the stardist model (1 = no classification).",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size to use during training.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=SEED,
        help="Seed to use for the data split.",
    )
    parser.add_argument(
        "-od",
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Path to the directory where the results will be saved.",
    )
    parser.add_argument(
        "-itd",
        "--imgs_to_display",
        type=int,
        default=IMGS_TO_DISPLAY,
        help="Number of images to display in the report.",
    )
    parser.add_argument(
        "-d",
        "--distance",
        type=int,
        default=DISTANCE,
        help="Distance to use for the graph.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default=MODE,
        help="Mode to use for the test ('valid' or 'test').",
    )

    args = parser.parse_args()

    with open("sflizard/pipeline/banner.txt", "r") as f:
        banner = f.read()
        print(banner)
    pipeline = TestPipeline(
        valid_data_path=args.valid_data_path,
        test_data_path=args.test_data_path,
        stardist_weights_path=args.stardist_weights_path,
        graph_weights_path=args.graph_weights_path,
        graph_distance=args.distance,
        n_rays=args.n_rays,
        n_classes=args.n_classes,
        batch_size=args.batch_size,
        seed=args.seed,
        mode=args.mode,
    )
    pipeline.test(
        output_dir=args.output_dir,
        imgs_to_display=args.imgs_to_display,
    )
    print("\nAll done!\n")
