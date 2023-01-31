from sflizard import HoverNetMetricTool

TEST_MODE = True

WEIGHTS_SELECTOR = {
    "model": ["graph_custom"],  # "graph_sage", "graph_gin", "graph_GCN"],
    "dimh": [1024],  # , 512, 1024],
    "num_layers": [4],
    "heads": [8],
    "custom_combinations": [
        # "1-0-0-0",
        # "2-540-0-0",
        # "2-1024-0-0",
        # "3-540-0-0",
        # "3-1024-0-0",
        # "0-0-1-0",
        # "0-0-2-7",
        # "0-0-2-16",
        # "0-0-3-7",
        # "0-0-3-16",
        "1-0-0-0-wide",
        "2-540-0-0-wide",
        "2-1024-0-0-wide",
        "3-540-0-0-wide",
        "3-1024-0-0-wide",
        "0-0-1-0-wide",
        "0-0-2-7-wide",
        "0-0-2-16-wide",
        "0-0-3-7-wide",
        "0-0-3-16-wide",
    ],
}

WEIGHTS_PATH = {
    "mod1-b-test": "models/graph_custom-1024-4-4ll-45-0-0-3-16-wide-0.0005-acc-epoch=103-val_acc=0.7817.ckpt",
}

if __name__ == "__main__":

    if not TEST_MODE:
        hmt = HoverNetMetricTool(
            mode="test",
            weights_selector=WEIGHTS_SELECTOR,
            distance=45,
            x_type="4ll",
        )

    else:
        hmt = HoverNetMetricTool(
            mode="test",
            weights_selector=[],
            distance=45,
            x_type="ll",
            paths=WEIGHTS_PATH,
        )
