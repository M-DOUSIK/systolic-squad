"""Compare student checkpoints at 128 x 96 (float): NYU test split vs ground truth, held-out Wikimedia people/indoor photos vs the teacher
(Depth Anything V2 Small metric indoor), and a picture: RGB | teacher | each checkpoint.
  python ml/compare_models.py ml/export/fastdepth_ft_128x96.pth ml/export/fastdepth_distill_128x96.pth
-> prints a table, writes ml/export/compare_models.png and compare_models.md"""
import glob, os, sys
import numpy as np, torch, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fd_common as C

W, H = 128, 96
ckpts = sys.argv[1:]
models = []
for c in ckpts:
    m = C.load_model()
    m.load_state_dict(torch.load(c, map_location="cpu", weights_only=False)["model_state_dict"])
    models.append((os.path.basename(c).replace("fastdepth_", "").replace("_128x96.pth", ""), m.eval()))

rows = []
nyu = C.nyu_files()
web = sorted(glob.glob(os.path.join(HERE, "data", "teacher", "web_*.npz")))[::10]          # the held-out web images (distill.py)
for name, m in models:
    a = np.mean([C.metrics(C.float_forward(m, x), g) for x, g in (C.prep(*C.load_raw(f), W, H) for f in nyu)], axis=0)
    b = []
    for f in web:
        z = np.load(f)
        x = cv2.resize(z["rgb"], (W, H), interpolation=cv2.INTER_AREA)
        t = cv2.resize(z["teacher"].astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        b.append(C.metrics(C.float_forward(m, x), t))
    b = np.mean(b, axis=0)
    rows.append((name, a, b))
    print(f"{name:12s} NYU (n={len(nyu)}): RMSE {a[0]:.3f} absrel {a[1]:.3f} d1 {a[2]:.3f} | held-out people/indoor vs teacher (n={len(web)}): "
          f"RMSE {b[0]:.3f} absrel {b[1]:.3f} d1 {b[2]:.3f}", flush=True)


def tile(im, label, vmax=6.0):
    if im.ndim == 3:
        t = cv2.cvtColor(np.asarray(im, np.uint8), cv2.COLOR_RGB2BGR)
    else:
        t = cv2.applyColorMap(np.clip(255 * (1 - im / vmax), 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    t = cv2.resize(t, (256, 192), interpolation=cv2.INTER_NEAREST)
    cv2.putText(t, label[:28], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return t


grid = []
for f in web[:10]:
    z = np.load(f)
    x = cv2.resize(z["rgb"], (W, H), interpolation=cv2.INTER_AREA)
    r = [tile(x, os.path.basename(f)[4:-4]), tile(z["teacher"].astype(np.float32), "teacher (DA-V2 S)")]
    r += [tile(C.float_forward(m, x), n) for n, m in models]
    grid.append(np.hstack(r))
cv2.imwrite(os.path.join(HERE, "export", "compare_models.png"), np.vstack(grid))
with open(os.path.join(HERE, "export", "compare_models.md"), "w") as fo:
    fo.write("| checkpoint | NYU RMSE | NYU absrel | NYU d1 | people/indoor vs teacher RMSE | absrel | d1 |\n|---|---|---|---|---|---|---|\n")
    for n, a, b in rows:
        fo.write(f"| {n} | {a[0]:.3f} | {a[1]:.3f} | {a[2]:.3f} | {b[0]:.3f} | {b[1]:.3f} | {b[2]:.3f} |\n")
