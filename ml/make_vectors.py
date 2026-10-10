"""Golden vectors for the firmware and the host emulator.
  python ml/make_vectors.py [n_nyu] [export dir]   -> <export dir, default ml/export>/vectors/frame_###.bin, _edge.bin, _depth.bin, _zones.bin, _L<i>.bin + index.json
Frames: n NYU test images (val_transform geometry) followed by the photos in ml/data/photos/ (host preprocessing: centre crop + resize).
_zones: 3 x u8 zone nearness (L, C, R) + u8 stable guidance, from reference/depth_ref.py run over the frames in this order."""
import glob, json, os, sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "reference"), os.path.join(ROOT, "host")]
import fd_common as C
from qmodel import QModel
from canny_ref import canny_ref
from depth_ref import rgb_to_gray, zone_nearness, ObstacleAgent
from preprocess import resize_rgb

EXP = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "export")
OUT = os.path.join(EXP, "vectors")


def frames(n_nyu, W, H):
    out = []
    files = C.nyu_files()
    for f in files[:: max(1, len(files) // max(n_nyu, 1))][:n_nyu]:
        out.append((os.path.basename(f), C.prep(*C.load_raw(f), W, H)[0]))
    for p in sorted(glob.glob(os.path.join(HERE, "data", "photos", "*.jpg")) + glob.glob(os.path.join(HERE, "data", "photos", "*.png"))):
        out.append((os.path.basename(p), resize_rgb(Image.open(p), W, H)))
    return out


def main():
    n_nyu = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    qm = QModel(os.path.join(EXP, "model_q.json"))
    W, H = qm.W, qm.H
    gate = qm.mq["gate"]
    kind = qm.mq["output"]["type"]
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "*.bin")):
        os.remove(f)
    agent = ObstacleAgent()
    index = []
    for i, (src, rgb) in enumerate(frames(n_nyu, W, H)):
        b = os.path.join(OUT, f"frame_{i:03d}")
        rgb.astype(np.uint8).tofile(b + ".bin")
        edge = canny_ref(rgb_to_gray(rgb).astype(np.uint8), gate["canny_low"], gate["canny_high"])
        np.asarray(edge, dtype=np.int8).tofile(b + "_edge.bin")
        outs = qm.run(rgb)
        for li, o in enumerate(outs):
            o.astype(np.int8).tofile(b + f"_L{li}.bin")
        db = qm.depth_bytes(outs)
        db.tofile(b + "_depth.bin")
        n = zone_nearness(db.astype(np.int64), kind)
        r = agent.update(n)
        np.array(list(n) + [r["stable"]], dtype=np.uint8).tofile(b + "_zones.bin")
        index.append({"frame": i, "source": src, "zones": list(map(int, n)), "stable": int(r["stable"]), "leds": r["leds"]})
    json.dump({"width": W, "height": H, "layers": len(qm.layers), "frames": index}, open(os.path.join(OUT, "index.json"), "w"), indent=1)
    print(f"{len(index)} frames -> {OUT}")


if __name__ == "__main__":
    main()
