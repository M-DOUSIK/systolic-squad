"""Integer golden model of the depth network. Reads ml/export/model_q.json and runs it with integer maths
only. The maths is reference/qmath_ref.py; the functions below are vectorised copies of it (the slow reference functions are the definition
and `python ml/qmodel.py --selftest` proves the vectorised ones bit-identical to them, including a full-frame check of selected layers).

  from qmodel import QModel; m = QModel(); outs = m.run(rgb_u8)      # outs[i] = INT8 HWC output of layer i; outs[-1] = final depth q
  python ml/qmodel.py --selftest
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "reference"))
import qmath_ref as R

MODEL_Q = os.path.join(HERE, "export", "model_q.json")


def rq_vec(acc, bias, M, S, zp, relu):
    """vectorised qmath_ref.rq over the last axis (per column / channel)"""
    M = np.asarray(M, dtype=np.int64); S = np.asarray(S, dtype=np.int64)
    v = (np.asarray(acc, dtype=np.int64) + np.asarray(bias, dtype=np.int64)) * M
    v = (v + ((np.int64(1) << S) >> 1)) >> S          # S = 0: adds 0, shifts 0;  >> on int64 = floor, as Python's >>
    v = v + int(zp)
    return np.clip(v, int(zp) if relu else -128, 127)


def im2col_vec(fm, KH, KW, stride, pad, pad_value):
    """same as qmath_ref.im2col (k = (ky*KW + kx)*C + ci), vectorised"""
    t, l, b, r = pad
    x = np.pad(np.asarray(fm, dtype=np.int64), ((t, b), (l, r), (0, 0)), constant_values=pad_value)
    H, W, Cc = x.shape
    Ho, Wo = (H - KH) // stride + 1, (W - KW) // stride + 1
    cols = [x[ky:ky + stride * (Ho - 1) + 1:stride, kx:kx + stride * (Wo - 1) + 1:stride, :] for ky in range(KH) for kx in range(KW)]
    return np.concatenate(cols, axis=2).reshape(Ho * Wo, KH * KW * Cc), Ho, Wo


def matmul_exact(X, Wm):
    """INT8 x INT8 -> exact INT64 sums (float64 BLAS is exact here: |sum| < 2^53)"""
    return np.rint(X.astype(np.float64) @ Wm.astype(np.float64)).astype(np.int64)


def dwconv_vec(x, Wdw, bias, M, S, zp_out, relu, stride, pad, pad_value):
    """same as qmath_ref.dwconv_q, vectorised"""
    KH, KW, Cc = Wdw.shape
    t, l, b, r = pad
    xp = np.pad(np.asarray(x, dtype=np.int64), ((t, b), (l, r), (0, 0)), constant_values=pad_value)
    Ho, Wo = (xp.shape[0] - KH) // stride + 1, (xp.shape[1] - KW) // stride + 1
    acc = np.zeros((Ho, Wo, Cc), dtype=np.int64)
    for ky in range(KH):
        for kx in range(KW):
            acc += xp[ky:ky + stride * (Ho - 1) + 1:stride, kx:kx + stride * (Wo - 1) + 1:stride, :] * Wdw[ky, kx]
    return rq_vec(acc, bias, M, S, zp_out, relu)


class QModel:
    def __init__(self, path=MODEL_Q):
        with open(path) as f:
            self.mq = json.load(f)
        self.layers = self.mq["layers"]
        for l in self.layers:
            if "W" in l:
                W = np.asarray(l["W"], dtype=np.int64)
                l["_W"] = W.reshape(l["KH"], l["KW"], l["C_out"]) if l["type"] == "dwconv" else W.reshape(-1, l["C_out"])
                l["_b"] = np.asarray(l["bias"], dtype=np.int64)
        self.W, self.H = self.mq["preproc"]["width"], self.mq["preproc"]["height"]

    def run_layer(self, l, srcs):
        t = l["type"]
        if t in ("conv", "pwconv"):
            X, Ho, Wo = im2col_vec(srcs[0], l["KH"], l["KW"], l["stride"], l["pad"], l.get("pad_value", l["in_zp"]))
            acc = matmul_exact(X, l["_W"])
            return rq_vec(acc, l["_b"], l["M"], l["S"], l["out_zp"], l["relu"]).reshape(Ho, Wo, l["C_out"])
        if t == "dwconv":
            return dwconv_vec(srcs[0], l["_W"], l["_b"], l["M"], l["S"], l["out_zp"], l["relu"], l["stride"], l["pad"], l.get("pad_value", l["in_zp"]))
        if t == "upsample":
            return R.upsample_nn(srcs[0], l["factor"])
        if t == "relu":
            return R.relu_q(srcs[0], l["out_zp"])
        if t == "bilinear":
            return R.bilinear_q(srcs[0], l["factor"], bool(l["align"]))
        if t == "add":
            return R.add_q(srcs[0], srcs[1], l["za"], l["zb"], l["Ma"], l["Mb"], l["S_add"], l["out_zp"], l["relu"])
        raise ValueError(t)

    def run(self, rgb_u8):
        """rgb_u8: uint8 [H][W][3] at the model size. Returns every layer's INT8 output (int64 arrays holding int8 values)."""
        x0 = np.asarray(rgb_u8, dtype=np.int64) - 128
        assert x0.shape == (self.H, self.W, 3), x0.shape
        outs = []
        for l in self.layers:
            srcs = [x0 if j == -2 else outs[j] for j in l["inputs"]]
            outs.append(self.run_layer(l, srcs))
        return outs

    def depth_bytes(self, outs):
        return (outs[-1][:, :, 0] + 128).astype(np.uint8)

    def depth_m(self, outs):
        o = self.mq["output"]
        return o["scale"] * (outs[-1][:, :, 0].astype(np.float64) - o["zero_point"])


def selftest():
    rng = np.random.default_rng(7)
    # rq: vectorised == reference, including S = 0, saturation, relu, negative values, .5 ties
    acc = rng.integers(-2**31, 2**31, 4000); bias = rng.integers(-2**28, 2**28, 4000)
    M = rng.integers(1, 32768, 4000); S = rng.integers(0, 32, 4000)
    acc[:50] = 3; bias[:50] = 0; M[:50] = 1; S[:50] = 1
    for zp in (-128, -5, 0, 127):
        for relu in (0, 1):
            v = rq_vec(acc, bias, M, S, zp, relu)
            ref = [R.rq(acc[i], bias[i], M[i], S[i], zp, relu) for i in range(len(acc))]
            assert (v == ref).all(), (zp, relu)
    # im2col
    for (KH, st, p) in [(3, 1, (1, 1, 1, 1)), (3, 2, (1, 1, 1, 1)), (1, 1, (0, 0, 0, 0)), (5, 1, (2, 2, 2, 2))]:
        fm = rng.integers(-128, 128, (8, 10, 3))
        a, _, _ = im2col_vec(fm, KH, KH, st, p, -9)
        assert (a == R.im2col(fm, KH, KH, st, p, -9)).all()
    # dwconv
    for (K, st) in [(3, 1), (3, 2), (5, 1)]:
        x = rng.integers(-128, 128, (8, 6, 5)); Wd = rng.integers(-127, 128, (K, K, 5))
        b = rng.integers(-5000, 5000, 5); Mm = rng.integers(1, 32768, 5); Ss = rng.integers(5, 25, 5)
        p = (K // 2,) * 4
        assert (dwconv_vec(x, Wd, b, Mm, Ss, -128, 1, st, p, -128) == R.dwconv_q(x, Wd, b, Mm, Ss, -128, 1, st, p, -128)).all()
    # matmul exactness at the extreme (K = 1024, all -128 x -127)
    X = np.full((2, 1024), -128); Wm = np.full((1024, 3), -127)
    assert (matmul_exact(X, Wm) == R.matmul_tiled(X, Wm, 16)).all()
    print("vectorised maths == qmath_ref: PASS")
    if os.path.exists(MODEL_Q):                          # full-model check: selected layers recomputed with the slow reference functions
        m = QModel()
        frame = rng.integers(0, 256, (m.H, m.W, 3))
        outs = m.run(frame)
        x0 = frame - 128
        for i, l in enumerate(m.layers):
            if l["name"] not in ("enc0", "enc1dw", "enc13pw", "dec2add", "dec5dw", "dec6pw", "out_up"):
                continue
            srcs = [x0 if j == -2 else outs[j] for j in l["inputs"]]
            t = l["type"]
            if t in ("conv", "pwconv"):
                X = R.im2col(srcs[0], l["KH"], l["KW"], l["stride"], tuple(l["pad"]), l["in_zp"])
                accr = R.matmul_tiled(X, l["_W"], 16)
                Ho, Wo = l["out_H"], l["out_W"]
                ref = np.array([[R.rq(accr[v, c], l["_b"][c], l["M"][c], l["S"][c], l["out_zp"], l["relu"]) for c in range(l["C_out"])]
                                for v in range(accr.shape[0])]).reshape(Ho, Wo, l["C_out"])
            elif t == "dwconv":
                ref = R.dwconv_q(srcs[0], l["_W"], l["_b"], l["M"], l["S"], l["out_zp"], l["relu"], l["stride"], tuple(l["pad"]), l["in_zp"])
            elif t == "add":
                ref = R.add_q(srcs[0], srcs[1], l["za"], l["zb"], l["Ma"], l["Mb"], l["S_add"], l["out_zp"], l["relu"])
            else:
                ref = R.upsample_nn(srcs[0], l["factor"])
            assert (ref == outs[i]).all(), l["name"]
            print(f"  layer {i:2d} {l['name']:8s}: slow reference == golden model")
        print("full-frame layer check: PASS")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
