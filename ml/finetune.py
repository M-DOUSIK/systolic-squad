"""Fine-tune the community FastDepth checkpoint at a small input size (it was trained at 224x224 and collapses below ~192 px).
Data: NYU Depth v2 TRAIN frames fetched by ml/data/fetch_train.sh (disjoint scenes from the val images used for evaluation).
Run: python ml/finetune.py [W H epochs]   -> ml/export/fastdepth_ft_<W>x<H>.pth"""
import glob, os, sys, time
import numpy as np, torch
import fd_common as C

W = int(sys.argv[1]) if len(sys.argv) > 1 else 128
H = int(sys.argv[2]) if len(sys.argv) > 2 else 96
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
INIT = sys.argv[4] if len(sys.argv) > 4 else None          # continue from a fine-tuned state dict
LR = float(sys.argv[5]) if len(sys.argv) > 5 else 3e-4
OUTNAME = sys.argv[6] if len(sys.argv) > 6 else f"fastdepth_ft_{W}x{H}.pth"
torch.manual_seed(0); np.random.seed(0)
torch.set_num_threads(10)

def crop_pair(rgb, depth):
    from PIL import Image
    ri = Image.fromarray(rgb).resize((333, 250), Image.BILINEAR)
    di = Image.fromarray(depth.astype(np.float32), mode="F").resize((333, 250), Image.BILINEAR)
    box = ((333 - 304) // 2, (250 - 228) // 2, (333 - 304) // 2 + 304, (250 - 228) // 2 + 228)
    return np.asarray(ri.crop(box)), np.asarray(di.crop(box))

files = sorted(glob.glob(os.path.join(C.ROOT, "ml", "data", "train_raw", "s*", "**", "*.h5"), recursive=True))
rgbs, deps = [], []
for f in files:
    try:
        r, d = C.load_raw(f)
        r, d = crop_pair(r, d)
        rgbs.append(r); deps.append(d)
    except Exception:
        pass
print("train frames", len(rgbs), flush=True)
rgbs = np.stack(rgbs); deps = np.stack(deps)

def batch(idx, aug):
    from PIL import Image
    xs, ys = [], []
    for i in idx:
        r, d = rgbs[i], deps[i]
        if aug:
            if np.random.rand() < 0.5: r, d = r[:, ::-1], d[:, ::-1]
            s = np.random.uniform(1.0, 1.3); d = d / s            # same depth/scale augmentation as the original recipe
            r = np.clip(r.astype(np.float32) * np.random.uniform(0.75, 1.25) + np.random.uniform(-20, 20), 0, 255).astype(np.uint8)
        ri = np.asarray(Image.fromarray(np.ascontiguousarray(r)).resize((W, H), Image.BILINEAR), dtype=np.float32) / 255.0
        di = np.asarray(Image.fromarray(np.ascontiguousarray(d).astype(np.float32), mode="F").resize((W, H), Image.BILINEAR))
        xs.append(ri.transpose(2, 0, 1)); ys.append(di[None])
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))

model = C.load_model()
if INIT:
    model.load_state_dict(torch.load(INIT, map_location="cpu", weights_only=False)["model_state_dict"])
model.train()
BS = 16
steps = EPOCHS * (len(rgbs) // BS)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)
t0 = time.time(); step = 0
for ep in range(EPOCHS):
    perm = np.random.permutation(len(rgbs)); tot = 0; n = 0
    for b in range(len(rgbs) // BS):
        x, y = batch(perm[b * BS:(b + 1) * BS], True)
        p = model(x)
        m = y > 0.001
        loss = (p - y).abs()[m].mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
        tot += loss.item(); n += 1
        if step % 20 == 0: print(f"ep{ep} step {step}/{steps} loss {tot/n:.3f}  {time.time()-t0:.0f}s", flush=True)
    torch.save({"model_state_dict": model.state_dict()}, os.path.join(C.ROOT, "ml", "export", OUTNAME))
    print(f"epoch {ep} done, mean L1 {tot/n:.3f}", flush=True)
