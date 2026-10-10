"""MiDaS v2.1 small (Intel ISL, MIT licence): load the model and trace it into a flat graph for the exporter."""
import os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MIDAS_DIR = os.path.join(HERE, "vendor", "MiDaS")
URL = "https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt"
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_midas_small():
    """MidasNet_small (EfficientNet-Lite3 encoder) with the released weights, eval mode. Input: RGB normalised with MEAN/STD."""
    sys.path.insert(0, MIDAS_DIR)
    from midas.midas_net_custom import MidasNet_small
    m = MidasNet_small(None, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={"expand": True})
    m.load_state_dict(torch.hub.load_state_dict_from_url(URL, map_location="cpu"))
    return m.eval()
