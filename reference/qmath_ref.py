"""Reference integer maths fromM4's ml/qmodel.py must agree with this.
Run: python3 reference/qmath_ref.py  (self-test)"""
import numpy as np

def rq(acc, bias, M, S, zp, relu):
    """per-column requantisation. zp = output zero point (int8)."""
    v = (int(acc) + int(bias)) * int(M)
    if S > 0:
        v = (v + (1 << (S - 1))) >> S          # Python >> on int = floor (arithmetic)
    v = v + int(zp)
    lo = int(zp) if relu else -128              # ReLU in the quantised domain = clamp at zero point
    return max(lo, min(127, v))

def fold_input_zp(bias, Wm, z_in):
    """bias'[c] = bias[c] - z_in * sum_k Wm[k][c]  (exact when im2col pads with z_in)"""
    return (np.asarray(bias, dtype=np.int64) - int(z_in) * np.asarray(Wm, dtype=np.int64).sum(axis=0)).astype(np.int64)

def choose_M_S(m):
    """real multiplier m (0 < m < 1 typically) -> (M, S): largest S<=31 with round(m*2^S) <= 32767"""
    for S in range(31, -1, -1):
        M = int(round(m * (1 << S)))
        if 1 <= M <= 32767:
            return M, S
    raise ValueError(m)

def im2col(fm, KH, KW, stride=1, pad=(0, 0, 0, 0), pad_value=0):
    """fm: int array [H][W][C] (HWC). pad = (top, left, bottom, right), padded with pad_value (= input zero point).
    returns [Ho*Wo][KH*KW*C] with k = (ky*KW+kx)*C + ci"""
    t, l, b, r = pad
    fm = np.pad(fm, ((t, b), (l, r), (0, 0)), constant_values=pad_value)
    H, W, C = fm.shape
    out = []
    for y in range(0, H - KH + 1, stride):
        for x in range(0, W - KW + 1, stride):
            out.append(fm[y:y+KH, x:x+KW, :].reshape(-1))   # numpy order = (ky, kx, ci) -> matches k formula
    return np.array(out, dtype=np.int64)

def matmul_tiled(X, Wm, N):
    """X [V][K] int8, Wm [K][Co] int8 -> INT32 [V][Co] computed tile by tile exactly as the hardware+firmware do"""
    V, K = X.shape; Co = Wm.shape[1]
    acc = np.zeros((V, Co), dtype=np.int64)
    for t in range(0, K, N):
        acc += X[:, t:t+N].astype(np.int64) @ Wm[t:t+N].astype(np.int64)
    return acc

# ---------------------------------------------------------------------------------------------------------------------
# Layer maths that runs on the CPU. Everything integer; floats appear only where an
# EXPORTER builds a table or picks M/S (build_lut, choose_add_params) - the resulting integers are the contract.
# ---------------------------------------------------------------------------------------------------------------------
def fold_input_zp_dw(bias, Wdw, z_in):
    """depthwise: bias'[c] = bias[c] - z_in * sum_{ky,kx} Wdw[ky][kx][c]   (exact when padding uses z_in)"""
    return (np.asarray(bias, dtype=np.int64) - int(z_in) * np.asarray(Wdw, dtype=np.int64).sum(axis=(0, 1))).astype(np.int64)

def dwconv_q(x, Wdw, bias, M, S, zp_out, relu, stride=1, pad=(0, 0, 0, 0), pad_value=0):
    """Depthwise KxK conv, per-channel requantisation. x [H][W][C] int, Wdw [KH][KW][C] int8, bias/M/S per channel
    (bias already zero-point folded with fold_input_zp_dw). Returns int64 array [Ho][Wo][C] holding int8 values."""
    x = np.asarray(x, dtype=np.int64); Wdw = np.asarray(Wdw, dtype=np.int64)
    KH, KW, C = Wdw.shape
    t, l, b, r = pad
    xp = np.pad(x, ((t, b), (l, r), (0, 0)), constant_values=pad_value)
    Ho = (xp.shape[0] - KH) // stride + 1; Wo = (xp.shape[1] - KW) // stride + 1
    out = np.zeros((Ho, Wo, C), dtype=np.int64)
    for oy in range(Ho):
        for ox in range(Wo):
            win = xp[oy * stride:oy * stride + KH, ox * stride:ox * stride + KW, :]
            acc = (win * Wdw).sum(axis=(0, 1))                       # per channel INT32 accumulator
            for c in range(C):
                out[oy, ox, c] = rq(acc[c], bias[c], M[c], S[c], zp_out, relu)
    return out

def choose_add_params(sa, sb, so):
    """exporter helper: real scales of the two inputs and the output -> (Ma, Mb, S) with Ma, Mb in 1..32767, S <= 30"""
    for S in range(30, -1, -1):
        Ma, Mb = int(round(sa / so * (1 << S))), int(round(sb / so * (1 << S)))
        if 1 <= Ma <= 32767 and 1 <= Mb <= 32767:
            return Ma, Mb, S
    raise ValueError((sa, sb, so))

def add_q(a, b, za, zb, Ma, Mb, S, zo, relu):
    """skip connection (residual add) of two INT8 tensors with different scales / zero points:
         v = (a - za)*Ma + (b - zb)*Mb;  if S > 0: v = (v + (1 << (S-1))) >> S;  v += zo;  clamp [relu ? zo : -128, 127]"""
    v = (np.asarray(a, dtype=np.int64) - int(za)) * int(Ma) + (np.asarray(b, dtype=np.int64) - int(zb)) * int(Mb)
    if S > 0:
        v = (v + (1 << (S - 1))) >> S
    v = v + int(zo)
    lo = int(zo) if relu else -128
    return np.clip(v, lo, 127)

def upsample_nn(x, f=2):
    """nearest-neighbour upsample of [H][W][C] by integer factor f (no requantisation: scale and zero point are unchanged)"""
    return np.repeat(np.repeat(np.asarray(x), f, axis=0), f, axis=1)

def relu_q(x, zp):
    """standalone ReLU on an INT8 tensor with zero point zp (real 0 = zp): y = max(x, zp); scale and zero point unchanged"""
    return np.maximum(np.asarray(x, dtype=np.int64), int(zp))

def bilinear_taps(n_in, n_out, align_corners):
    """per output index: (i0, i1, w1) with weights in 1/256 units, exact integer arithmetic (same as the C firmware).
       align_corners=1: src = o*(n_in-1)/(n_out-1);  align_corners=0: src = (o+0.5)*n_in/n_out - 0.5, clamped at 0."""
    taps = []
    for o in range(n_out):
        if align_corners:
            num, den = o * (n_in - 1), max(n_out - 1, 1)
        else:
            num, den = max((2 * o + 1) * n_in - n_out, 0), 2 * n_out
        i0, rem = num // den, num % den
        w1 = (rem * 512 + den) // (2 * den)                # round(rem * 256 / den)
        i1 = min(i0 + 1, n_in - 1)
        taps.append((min(i0, n_in - 1), i1, w1))
    return taps

def bilinear_q(x, f, align_corners):
    """bilinear upsample of [H][W][C] INT8 by integer factor f; scale and zero point unchanged:
       y = (sum_{a,b} wy[a]*wx[b]*x[iy[a]][ix[b]] + 32768) >> 16, weights in 1/256 (wy0 = 256 - wy1)"""
    x = np.asarray(x, dtype=np.int64)
    H, W, _ = x.shape
    ty, tx = bilinear_taps(H, H * f, align_corners), bilinear_taps(W, W * f, align_corners)
    out = np.zeros((H * f, W * f, x.shape[2]), dtype=np.int64)
    for oy, (y0, y1, wy1) in enumerate(ty):
        wy0 = 256 - wy1
        for ox, (x0, x1, wx1) in enumerate(tx):
            wx0 = 256 - wx1
            v = wy0 * (wx0 * x[y0, x0] + wx1 * x[y0, x1]) + wy1 * (wx0 * x[y1, x0] + wx1 * x[y1, x1])
            out[oy, ox] = (v + 32768) >> 16
    return out

def build_lut(fn, s_in, z_in, s_out, z_out):
    """exporter helper for activations other than ReLU: 256-entry INT8 table indexed by (x + 128):
         T[i] = clamp(round(fn((i - 128 - z_in) * s_in) / s_out) + z_out, -128, 127)"""
    import math
    t = []
    for i in range(256):
        real = fn((i - 128 - int(z_in)) * s_in)
        t.append(max(-128, min(127, int(math.floor(real / s_out + 0.5)) + int(z_out))))
    return np.array(t, dtype=np.int64)

def lut_q(x, table):
    """apply a 256-entry table to an INT8 tensor: y = T[x + 128]"""
    return np.asarray(table, dtype=np.int64)[np.asarray(x, dtype=np.int64) + 128]

if __name__ == "__main__":
    assert rq(1000, 0, 16384, 15, 0, False) == 127          # 500 saturates
    assert rq(-300, 0, 16384, 15, 0, False) == -128            # -150 saturates
    assert rq(-3, 0, 1, 1, 0, False) == -1                  # (-3+1)>>1 = -1
    assert rq(-3, 0, 1, 1, 0, True) == 0
    assert rq(-3, 0, 1, 1, -128, False) == -128 and rq(10, 0, 1, 0, -128, True) == -118
    assert rq(-50, 0, 1, 0, -128, True) == -128        # relu clamps at zero point
    # zero-point folding is exact: sum_k (x-z)*w + b == sum_k x*w + fold(b)
    rng = np.random.default_rng(1); X = rng.integers(-128, 128, (5, 9)); Wm = rng.integers(-127, 128, (9, 4)); b = rng.integers(-1000, 1000, 4)
    assert ((X - 7) @ Wm + b == X @ Wm + fold_input_zp(b, Wm, 7)).all()
    cols = im2col(np.ones((4, 4, 1), dtype=int) * 3, 3, 3, stride=2, pad=(1, 1, 1, 1), pad_value=-5)
    assert cols.shape == (4, 9) and cols[0][0] == -5 and cols[0][4] == 3
    assert rq(5, 0, 1, 1, 0, False) == 3                    # (5+1)>>1 = 3
    M, S = choose_M_S(0.0123); assert abs(M / 2**S - 0.0123) < 1e-6
    fm = np.arange(5*5*2).reshape(5, 5, 2); cols = im2col(fm, 3, 3)
    assert cols.shape == (9, 18) and cols[0][2] == fm[0, 1, 0]   # k=2 -> ky=0,kx=1,ci=0
    A = np.array([[18, 8, -6, -13], [0, 18, -2, 2], [-10, -10, 3, 15], [19, 3, -18, 1]])
    W = np.array([[-19, 3, 9, 17], [-19, 0, 12, -9], [1, 4, 6, 7], [-5, -6, -18, 16]])
    assert (matmul_tiled(A, W, 2) == A @ W).all() and (A @ W)[0, 0] == -435
    # --- contract v1.1 layer maths (CPU side) ---
    Wd = np.array([[[1, -2], [0, 3]], [[2, 1], [-1, 0]]], dtype=np.int64)            # 2x2 depthwise, C=2
    xd = np.arange(3 * 3 * 2).reshape(3, 3, 2) - 9
    bd = np.array([5, -7]); Md = [3, 5]; Sd = [2, 3]
    od = dwconv_q(xd, Wd, bd, Md, Sd, 1, False)
    manual = rq(sum(int(xd[ky, kx, 0]) * int(Wd[ky, kx, 0]) for ky in range(2) for kx in range(2)), 5, 3, 2, 1, False)
    assert od.shape == (2, 2, 2) and od[0, 0, 0] == manual
    zi = -3                                                                              # zero-point folding is exact for dw too
    xz = np.random.default_rng(2).integers(-128, 128, (4, 4, 3)); Wz = np.random.default_rng(3).integers(-127, 128, (3, 3, 3))
    bz = np.array([100, -50, 7]); Mz = [100, 200, 300]; Sz = [9, 9, 9]
    # folded bias + padding with z_in  ==  unfolded bias on the zero-point-subtracted tensor padded with 0
    f1 = dwconv_q(xz, Wz, fold_input_zp_dw(bz, Wz, zi), Mz, Sz, 0, False, pad=(1, 1, 1, 1), pad_value=zi)
    f2 = dwconv_q(xz - zi, Wz, bz, Mz, Sz, 0, False, pad=(1, 1, 1, 1), pad_value=0)
    assert (f1 == f2).all()
    Ma, Mb, Sa = choose_add_params(0.05, 0.1, 0.08)
    assert abs(Ma / 2**Sa - 0.05 / 0.08) < 1e-3 and abs(Mb / 2**Sa - 0.1 / 0.08) < 1e-3
    assert add_q(10, 20, 0, 0, 1, 1, 0, 0, False) == 30 and add_q(100, 100, 0, 0, 1, 1, 0, 0, False) == 127
    assert add_q(-5, -5, 0, 0, 1, 1, 0, 0, True) == 0
    assert (upsample_nn(np.array([[[1], [2]], [[3], [4]]]), 2)[:, :, 0] == [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]).all()
    assert relu_q(np.array([-128, -60, -50, 10]), -50).tolist() == [-50, -50, -50, 10]
    # bilinear: constant stays constant, and it matches PyTorch within 1 step (both align_corners modes)
    xc = np.full((3, 4, 2), 37); assert (bilinear_q(xc, 2, 1) == 37).all() and (bilinear_q(xc, 2, 0) == 37).all()
    try:
        import torch, torch.nn.functional as F
        xr = np.random.default_rng(4).integers(-128, 128, (6, 8, 3))
        for ac in (0, 1):
            t = F.interpolate(torch.tensor(xr, dtype=torch.float64).permute(2, 0, 1)[None], scale_factor=2, mode="bilinear", align_corners=bool(ac))[0].permute(1, 2, 0).numpy()
            assert np.abs(bilinear_q(xr, 2, ac) - t).max() <= 1.0, ac
    except ImportError:
        pass
    T = build_lut(lambda v: max(v, 0.0), 0.1, 0, 0.1, 0)                               # ReLU as a table
    assert lut_q(np.array([-128, -1, 0, 5, 127]), T).tolist() == [0, 0, 0, 5, 127]
    print("qmath_ref self-test PASS")
