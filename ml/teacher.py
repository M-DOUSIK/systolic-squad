"""Teacher for distillation: Depth Anything V2 Small, metric indoor (Hypersim), Apache-2.0
(github.com/DepthAnything/Depth-Anything-V2, checkpoint depth_anything_v2_metric_hypersim_vits.pth). Laptop only - never on the board."""
import os, sys
import numpy as np, torch, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
DA = os.path.join(HERE, "vendor", "Depth-Anything-V2")
sys.path.insert(0, os.path.join(DA, "metric_depth"))
from depth_anything_v2.dpt import DepthAnythingV2

_m = None
def load():
    global _m
    if _m is None:
        _m = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=20)
        _m.load_state_dict(torch.load(os.path.join(DA, "checkpoints", "depth_anything_v2_metric_hypersim_vits.pth"), map_location="cpu"))
        _m.eval()
    return _m

def depth(rgb_u8, size=392):
    """uint8 RGB HxWx3 -> metric depth (m), same HxW"""
    with torch.no_grad():
        return load().infer_image(cv2.cvtColor(np.asarray(rgb_u8), cv2.COLOR_RGB2BGR), input_size=size)
