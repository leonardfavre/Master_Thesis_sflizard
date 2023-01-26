from sflizard import HoverNetMetricTool 

WEIGHTS_SELECTOR = {
    "model": ["graph_custom"],
    "dimh": [256, 512, 1024],
    "num_layers": [2, 4, 8],
}

if __name__ == "__main__":

    hmt = HoverNetMetricTool(
        mode = "valid",
        weights_selector = WEIGHTS_SELECTOR,
        distance = 45,
        x_type = "c",
    )
