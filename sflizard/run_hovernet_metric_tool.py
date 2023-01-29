from sflizard import HoverNetMetricTool

WEIGHTS_SELECTOR = {
    "model": ["graph_custom"],  # "graph_sage", "graph_gin", "graph_GCN"],
    "dimh": [256],  # , 512, 1024],
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

if __name__ == "__main__":

    hmt = HoverNetMetricTool(
        mode="valid",
        weights_selector=WEIGHTS_SELECTOR,
        distance=45,
        x_type="4ll+c",
    )
