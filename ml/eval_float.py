"""Step 4: float FastDepth on the NYU test images at 224x224 and 128x96. Run: python ml/eval_float.py [n_images]"""
import sys, numpy as np
import fd_common as C

n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
m = C.load_model()
files = C.nyu_files()[-n:]               # last n (the first images are used for calibration)
for (W, H) in [(224, 224), (128, 96)]:
    r = []
    for f in files:
        rgb, d = C.load_raw(f)
        x, g = C.prep(rgb, d, W, H)
        r.append(C.metrics(C.float_forward(m, x), g))
    r = np.array(r)
    print(f"float {W}x{H}: n={len(r)} RMSE={r[:,0].mean():.3f} m  absrel={r[:,1].mean():.3f}  delta1={r[:,2].mean():.3f}")
