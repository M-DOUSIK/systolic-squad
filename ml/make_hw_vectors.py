"""Hardware test vectors + APB scripts for hw/tb/shell/tb_npu_top.sv.
  python ml/make_hw_vectors.py [N]   -> hw/tb/vectors/<case>_{W,X,Y,Yq,cfg}.hex and <case>.cmd ; ml/export/macs.csv
Cases: rand (random tile), canny_blur (K_G, RQ M=1 S=4), canny_sobel (K_X, K_Y, RAW), enc0 (3x3 conv K=27), enc1pw (K=32), enc3pw (K=128 > N).
The depth-model cases take their input from the golden layer outputs of frame_000 and must reproduce the next golden layer bit for bit
(the generator asserts it, the RTL testbench checks it)."""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "reference"), os.path.join(ROOT, "hw", "tb", "shell")]
from qmath_ref import rq, im2col
from canny_ref import canny_ref
from depth_ref import rgb_to_gray
from qmodel import QModel, im2col_vec
import gen_shell_tests as G

OUT = os.path.join(ROOT, "hw", "tb", "vectors")
VEC = os.path.join(HERE, "export", "vectors")
K_G = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]); K_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]); K_Y = K_X.T


def h(v, digits):
    return format(int(v) & ((1 << (4 * digits)) - 1), f"0{digits}x")


def write_case(name, N, X, Wm, rq_en, relu=0, zp=0, M=None, S=None, bias=None, expect_q=None):
    """X [V][K], Wm [K][Co]. Hex files hold the K-padded matrices and the first C_out tile (N columns); the .cmd script runs every tile."""
    V, K = X.shape; Co = Wm.shape[1]
    Kp = -(-K // N) * N
    Xp = np.zeros((V, Kp), np.int64); Xp[:, :K] = X
    Wp = np.zeros((Kp, N), np.int64); Wp[:K, :min(N, Co)] = Wm[:, :N]
    acc = X @ Wm
    if rq_en:
        q = np.array([[rq(acc[v, c], bias[c], M[c], S[c], zp, relu) for c in range(Co)] for v in range(V)])
        if expect_q is not None:
            assert (q == expect_q).all(), f"{name}: export maths != golden layer"
    b = os.path.join(OUT, name)
    open(b + "_W.hex", "w").write("\n".join(h(Wp[r, c], 2) for r in range(Kp) for c in range(N)) + "\n")
    open(b + "_X.hex", "w").write("\n".join(h(Xp[m, r], 2) for m in range(V) for r in range(Kp)) + "\n")
    open(b + "_Y.hex", "w").write("\n".join(h(acc[m, c] if c < Co else 0, 8) for m in range(V) for c in range(N)) + "\n")
    if rq_en:
        open(b + "_Yq.hex", "w").write("\n".join(h(q[m, c] if c < Co else rq(0, 0, 1, 0, zp, relu), 2) for m in range(V) for c in range(N)) + "\n")
        cfg = [h(relu, 2), h(zp, 2)] + [f"{h(M[c] if c < Co else 1, 4)} {h(S[c] if c < Co else 0, 2)} {h(bias[c] if c < Co else 0, 8)}" for c in range(N)]
        open(b + "_cfg.hex", "w").write("\n".join(cfg) + "\n")
    s = G.Script()
    s.c(name)
    G.matmul_ops(s, N, 1024, X, Wm, rq_en, relu, zp, M, S, bias)
    s.lines.append("e")
    open(b + ".cmd", "w").write("\n".join(s.lines) + "\n")
    print(f"{name}: V={V} K={K} C_out={Co} ({'RQ' if rq_en else 'RAW'}), {s.apb} APB accesses")


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(5)
    qm = QModel()
    W, H = qm.W, qm.H
    write_case("rand", N, rng.integers(-128, 128, (40, N)), rng.integers(-127, 128, (N, N)), 1, 1, -128,
               rng.integers(1, 32768, N).tolist(), rng.integers(10, 20, N).tolist(), rng.integers(-50000, 50000, N).tolist())
    rgb = np.fromfile(os.path.join(VEC, "frame_000.bin"), np.uint8).reshape(H, W, 3)
    gray = rgb_to_gray(rgb)
    x = gray >> 1
    xp = np.pad(x, 1, mode="edge")[:, :, None]                                    # replicate padding
    win = im2col(xp, 3, 3)[:64]
    blur_ref = canny_ref(gray.astype(np.uint8), return_stages=True)
    write_case("canny_blur", N, win, K_G.reshape(9, 1), 1, 0, 0, [1], [4], [0])
    b_ = ((im2col(xp, 3, 3) @ K_G.reshape(9, 1) + 8) >> 4).reshape(H, W)
    bp = np.pad(b_, 1, mode="edge")[:, :, None]
    write_case("canny_sobel", N, im2col(bp, 3, 3)[:64], np.stack([K_X.reshape(9), K_Y.reshape(9)], 1), 0)
    macs = []
    for name in ("enc0", "enc1pw", "enc3pw"):
        i = [l["name"] for l in qm.layers].index(name)
        l = qm.layers[i]
        j = l["inputs"][0]
        src = (rgb.astype(np.int64) - 128) if j == -2 else np.fromfile(os.path.join(VEC, f"frame_000_L{j}.bin"), np.int8).reshape(l["in_H"], l["in_W"], l["C_in"]).astype(np.int64)
        X, Ho, Wo = im2col_vec(src, l["KH"], l["KW"], l["stride"], l["pad"], l["in_zp"])
        gold = np.fromfile(os.path.join(VEC, f"frame_000_L{i}.bin"), np.int8).reshape(Ho * Wo, l["C_out"]).astype(np.int64)
        sel = slice(0, 64)
        write_case(name, N, X[sel], l["_W"], 1, l["relu"], l["out_zp"], l["M"], l["S"], l["bias"], gold[sel])
    with open(os.path.join(HERE, "export", "macs.csv"), "w") as f:
        f.write("index,name,type,runs_on,K,C_out,out_pixels,MACs\n")
        tot = 0
        for i, l in enumerate(qm.layers):
            px = l["out_H"] * l["out_W"]
            if l["type"] in ("conv", "pwconv"):
                K = l["KH"] * l["KW"] * l["C_in"]; m = px * K * l["C_out"]; on = "NPU"
            elif l["type"] == "dwconv":
                K = l["KH"] * l["KW"]; m = px * K * l["C_out"]; on = "CPU"
            else:
                K = 0; m = 0; on = "CPU"
            tot += m
            f.write(f"{i},{l['name']},{l['type']},{on},{K},{l['C_out']},{px},{m}\n")
        f.write(f",total,,,,,,{tot}\n")
    print(f"macs.csv: {tot / 1e6:.1f} M MACs per frame")


if __name__ == "__main__":
    main()
