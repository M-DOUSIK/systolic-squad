"""Host-side preprocessing: the host only resizes. Any RGB image -> centre crop to the model's aspect
ratio -> bilinear resize to preproc.width x preproc.height -> uint8 HWC RGB. Shared by host/app.py, ml/make_vectors.py and the
accuracy report so the board, the emulator and the golden vectors all see the same pixels."""
import numpy as np
from PIL import Image


def resize_rgb(img, W, H):
    """img: PIL.Image or uint8 HWC RGB array (any size). Returns uint8 [H][W][3]."""
    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img, dtype=np.uint8))
    img = img.convert("RGB")
    w, h = img.size
    target = W / H
    if w / h > target:                       # too wide: crop the sides
        nw = int(round(h * target)); x0 = (w - nw) // 2
        img = img.crop((x0, 0, x0 + nw, h))
    elif w / h < target:                     # too tall: crop top and bottom
        nh = int(round(w / target)); y0 = (h - nh) // 2
        img = img.crop((0, y0, w, y0 + nh))
    return np.asarray(img.resize((W, H), Image.BILINEAR), dtype=np.uint8)
