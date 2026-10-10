"""Shared helpers: load the FastDepth checkpoint and the NYU test images (preparation copied from the
vendor repo's val_transform: resize to 250 x 333, centre crop 228 x 304, resize to the target size; RGB scaled to 0..1, NO mean/std)."""
import glob, os, sys, warnings
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "ml", "vendor", "hagaik-fastdepth")
CKPT = os.path.join(VENDOR, "Weights", "FastDepth_L1_Best.pth")
NYU_DIR = os.path.join(ROOT, "ml", "data", "val", "official")


def load_model():
    import torch
    warnings.filterwarnings("ignore")
    sys.path.insert(0, VENDOR)
    import models
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    m = models.FastDepth()
    m.load_state_dict(ck["model_state_dict"])
    return m.eval()


def nyu_files():
    import h5py
    ok = []
    for f in sorted(glob.glob(os.path.join(NYU_DIR, "*.h5"))):
        try:
            with h5py.File(f, "r") as h:
                h["rgb"][:]; h["depth"][:]
            ok.append(f)
        except Exception:
            pass
    return ok


def load_raw(f):
    import h5py
    with h5py.File(f, "r") as h:
        return np.transpose(h["rgb"][:], (1, 2, 0)), h["depth"][:]


def prep(rgb, depth, W, H):
    """NYU frame (480x640) -> (uint8 HWC rgb at H x W, float32 depth at H x W) using the vendor val_transform geometry."""
    ri = Image.fromarray(rgb).resize((333, 250), Image.BILINEAR)
    di = Image.fromarray(depth.astype(np.float32), mode="F").resize((333, 250), Image.BILINEAR)
    l, t = (333 - 304) // 2, (250 - 228) // 2
    box = (l, t, l + 304, t + 228)
    ri, di = ri.crop(box), di.crop(box)
    ri = ri.resize((W, H), Image.BILINEAR)
    di = di.resize((W, H), Image.BILINEAR)
    return np.asarray(ri, dtype=np.uint8), np.asarray(di, dtype=np.float32)


def metrics(pred, gt):
    m = gt > 0.001
    p, g = np.maximum(pred[m], 1e-3), gt[m]
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    absrel = float(np.mean(np.abs(p - g) / g))
    d1 = float(np.mean(np.maximum(p / g, g / p) < 1.25))
    return rmse, absrel, d1


def float_forward(model, rgb_u8):
    import torch
    x = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    with torch.no_grad():
        return model(x)[0, 0].numpy()
