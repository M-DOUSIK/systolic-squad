"""Teacher labels for distillation (laptop only). For every training image: 304 x 228 RGB crop (FastDepth geometry), teacher depth
(Depth Anything V2 Small, metric indoor, ml/teacher.py) and, for NYU frames, the ground truth. Cached in ml/data/teacher/*.npz; images that
already have a cache file are skipped, so it can be re-run while more data downloads.
  python ml/make_teacher_labels.py [max_images]"""
import glob, os, sys, time
import numpy as np, torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "host")]
import fd_common as C
import teacher as T
from preprocess import resize_rgb

OUT = os.path.join(HERE, "data", "teacher")
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(8)
lim = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
NYU_STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 3       # consecutive NYU video frames are near-duplicates (and have ground truth)


def crop_nyu(rgb, depth):
    ri = Image.fromarray(rgb).resize((333, 250), Image.BILINEAR)
    di = Image.fromarray(depth.astype(np.float32), mode="F").resize((333, 250), Image.NEAREST)
    box = ((333 - 304) // 2, (250 - 228) // 2, (333 - 304) // 2 + 304, (250 - 228) // 2 + 228)
    return np.asarray(ri.crop(box)), np.asarray(di.crop(box))


jobs = []                                                          # web images first: they have no other label
for f in sorted(glob.glob(os.path.join(HERE, "data", "unlabeled", "*.jpg"))):
    jobs.append(("web", f, "web_" + os.path.basename(f)[:-4]))
for f in sorted(glob.glob(os.path.join(HERE, "data", "train_raw", "**", "*.h5"), recursive=True))[::NYU_STRIDE]:
    jobs.append(("nyu", f, "nyu_" + os.path.basename(os.path.dirname(f)) + "_" + os.path.basename(f)[:-3]))
done, t0 = 0, time.time()
for kind, f, name in jobs:
    dst = os.path.join(OUT, name + ".npz")
    if os.path.exists(dst):
        continue
    try:
        if kind == "nyu":
            rgb, gt = crop_nyu(*C.load_raw(f))
        else:
            rgb, gt = resize_rgb(Image.open(f), 304, 228), None
        td = T.depth(rgb, 392).astype(np.float16)
    except Exception as e:
        print("skip", f, e); continue
    np.savez_compressed(dst, rgb=rgb, teacher=td, gt=(gt.astype(np.float16) if gt is not None else np.zeros(0, np.float16)))
    done += 1
    if done % 50 == 0:
        print(f"{done} labelled, {(time.time() - t0) / done:.2f} s/img", flush=True)
    if done >= lim:
        break
print(f"teacher labels: {done} new, {len(glob.glob(os.path.join(OUT, '*.npz')))} total")
