"""Fetch NYU train frames from many places inside the HF tar shards (range requests), parse tar headers, keep the .h5 members.
Run: python ml/data/fetch_spread.py   -> ml/data/train_raw/spread/<scene>_<file>.h5"""
import os, subprocess, sys
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_raw", "spread")
os.makedirs(OUT, exist_ok=True)
URL = "https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/main/data/train-{:06d}.tar"
SIZES = {i: 3003000000 for i in range(11)}; SIZES[11] = 1098700800
CHUNK = 12 * 1024 * 1024
PER_SHARD = int(sys.argv[1]) if len(sys.argv) > 1 else 8
PHASE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
for s in range(12):
    for j in range(PER_SHARD):
        start = (SIZES[s] - CHUNK - 1) * (j + PHASE) / PER_SHARD
        start = int(start) // 512 * 512
        r = subprocess.run(["curl", "-m", "300", "-L", "-s", "-r", f"{start}-{start + CHUNK - 1}", URL.format(s)], capture_output=True)
        buf = r.stdout
        n = 0; p = 0
        while p + 512 <= len(buf):
            h = buf[p:p + 512]
            if h[257:262] == b"ustar":
                name = h[0:100].split(b"\0")[0].decode(errors="ignore")
                try:
                    size = int(h[124:136].split(b"\0")[0].strip() or b"0", 8)
                except ValueError:
                    p += 512; continue
                data = buf[p + 512:p + 512 + size]
                if name.endswith(".h5") and len(data) == size:
                    fn = name.replace("/", "_")
                    with open(os.path.join(OUT, fn), "wb") as f:
                        f.write(data)
                    n += 1
                p += 512 + (size + 511) // 512 * 512
            else:
                p += 512
        print(f"shard {s} part {j}: {n} files", flush=True)
open(os.path.join(os.path.dirname(OUT), "..", f"train_fetch_done_{PER_SHARD}_{PHASE}"), "w").write("done")
