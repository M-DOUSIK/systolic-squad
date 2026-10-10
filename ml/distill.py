"""Distil Depth Anything V2 Small (metric indoor teacher, laptop only) into our FastDepth student at 128 x 96. Same architecture as before,
so hardware, firmware and export path are unchanged: only the weights (model_blob.bin) change.
Data: ml/data/teacher/*.npz from ml/make_teacher_labels.py (NYU train frames with ground truth + Wikimedia indoor/people photos without).
Loss: NYU -> L1(gt) + 0.5 L1(teacher rescaled to the frame's ground-truth median); web -> L1(teacher). Augment: zoom 1..1.8 (depth / zoom,
i.e. the camera moves closer), flip, brightness/contrast. 10 % of the web images are held out (index % 10 == 0).
  python ml/distill.py [epochs] [init_ckpt] [out_name]"""
import glob, os, sys, time
import numpy as np, torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fd_common as C

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
INIT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "export", "fastdepth_ft_128x96.pth")
OUTN = sys.argv[3] if len(sys.argv) > 3 else "fastdepth_distill_128x96.pth"
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5e-4
GRAD = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0      # weight of the depth-gradient (edge) loss against the teacher
W, H, BS = 128, 96, 16
torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(10)

files = sorted(glob.glob(os.path.join(HERE, "data", "teacher", "*.npz")))
web = [f for f in files if os.path.basename(f).startswith("web_")]
held = set(web[::10])
train = [f for f in files if f not in held]
data = []
for f in train:
    z = np.load(f)
    data.append((z["rgb"], z["teacher"].astype(np.float32), z["gt"].astype(np.float32) if z["gt"].size else None))
n_nyu = sum(d[2] is not None for d in data)
print(f"train: {len(data)} images ({n_nyu} NYU with ground truth, {len(data) - n_nyu} web), held-out web {len(held)}", flush=True)


def sample(i):
    rgb, td, gt = data[i]
    h0, w0 = td.shape
    s = np.random.uniform(1.0, 1.8)
    ch, cw = int(h0 / s), int(w0 / s)
    y0 = np.random.randint(0, h0 - ch + 1); x0 = np.random.randint(0, w0 - cw + 1)
    sl = (slice(y0, y0 + ch), slice(x0, x0 + cw))
    x = torch.from_numpy(rgb[sl].astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)[0]
    t = F.interpolate(torch.from_numpy(td[sl])[None, None], size=(H, W), mode="bilinear", align_corners=False)[0] / s
    if gt is not None:
        g = F.interpolate(torch.from_numpy(gt[sl])[None, None], size=(H, W), mode="nearest")[0] / s
        m = g > 0.01
        if m.sum() > 50:
            t = t * (g[m].median() / t[m].median().clamp(min=1e-3))
    else:
        g = torch.zeros_like(t)
    if np.random.rand() < 0.5:
        x, t, g = x.flip(-1), t.flip(-1), g.flip(-1)
    x = ((x - 0.5) * np.random.uniform(0.75, 1.25) + 0.5 + np.random.uniform(-0.08, 0.08)).clamp(0, 1)
    return x, t, g


model = C.load_model()
model.load_state_dict(torch.load(INIT, map_location="cpu", weights_only=False)["model_state_dict"])
model.train()
steps = EPOCHS * (len(data) // BS)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
t0 = time.time(); step = 0
for ep in range(EPOCHS):
    perm = np.random.permutation(len(data)); tot = 0; n = 0
    for b in range(len(data) // BS):
        xs, ts, gs = zip(*[sample(i) for i in perm[b * BS:(b + 1) * BS]])
        x, t, g = torch.stack(xs), torch.stack(ts), torch.stack(gs)
        p = model(x)
        mg = g > 0.01
        has_gt = mg.flatten(1).any(1)
        lt = (p - t).abs().mean(dim=(1, 2, 3))
        if GRAD > 0:                                          # match the teacher's depth edges (sharper boundaries)
            gx = lambda a: a[..., :, 1:] - a[..., :, :-1]
            gy = lambda a: a[..., 1:, :] - a[..., :-1, :]
            lt = lt + GRAD * ((gx(p) - gx(t)).abs().mean(dim=(1, 2, 3)) + (gy(p) - gy(t)).abs().mean(dim=(1, 2, 3)))
        lg = torch.stack([(p[k] - g[k]).abs()[mg[k]].mean() if has_gt[k] else p.new_zeros(()) for k in range(len(p))])
        loss = torch.where(has_gt, lg + 0.5 * lt, lt).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
        tot += loss.item(); n += 1
        if step % 20 == 0:
            print(f"ep{ep} step {step}/{steps} loss {tot / n:.3f}  {time.time() - t0:.0f}s", flush=True)
    torch.save({"model_state_dict": model.state_dict()}, os.path.join(HERE, "export", OUTN))
    print(f"epoch {ep} done, mean loss {tot / n:.3f}", flush=True)
