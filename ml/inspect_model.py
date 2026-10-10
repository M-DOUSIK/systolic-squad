"""Step 3: layers.csv + parameter count + forward checks. Run: python ml/inspect_model.py"""
import csv, os, torch.nn as nn
import fd_common as C

m = C.load_model()
rows = []
for name, mod in m.named_modules():
    if isinstance(mod, nn.Conv2d):
        t = "dwconv" if mod.groups > 1 else ("pwconv" if mod.kernel_size == (1, 1) else "conv")
        rows.append([name, t, mod.in_channels, mod.out_channels, mod.kernel_size[0], mod.stride[0], mod.padding[0]])
with open(os.path.join(C.ROOT, "ml", "export", "layers.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["name", "type", "c_in", "c_out", "kernel", "stride", "pad"]); w.writerows(rows)
nconv = sum(p.numel() for n, p in m.named_parameters() if p.dim() == 4)
nall = sum(p.numel() for p in m.parameters())
print("conv layers", len(rows), "conv weights", nconv, "all params", nall)
