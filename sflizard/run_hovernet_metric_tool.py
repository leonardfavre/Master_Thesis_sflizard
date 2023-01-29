from sflizard import HoverNetMetricTool

WEIGHTS_SELECTOR = {
    "model": ["graph_sage"],  # "graph_sage", "graph_gin", "graph_GCN"],
    "dimh": [256],  # , 512, 1024],
    "num_layers": [4],
    "heads": [1],
}

if __name__ == "__main__":

    hmt = HoverNetMetricTool(
        mode="test",
        weights_selector=WEIGHTS_SELECTOR,
        distance=45,
        x_type="4ll+c",
    )
