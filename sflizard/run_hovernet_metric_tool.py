from sflizard import HoverNetMetricTool

WEIGHTS_SELECTOR = {
    "model": ["graph_gat", "graph_sage", "graph_gin", "graph_GCN"],
    "dimh": [16, 32, 64, 128, 256],
    "num_layers": [2, 4, 8],
    "heads": [8],
}

if __name__ == "__main__":

    hmt = HoverNetMetricTool(
        mode="valid",
        weights_selector=WEIGHTS_SELECTOR,
        distance=45,
        x_type="c+x",
    )
