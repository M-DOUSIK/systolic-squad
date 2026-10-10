"""Generates APB command scripts for hw/tb/shell/tb_npu_top.sv (npu_top, contract v1.2) and the expected results (reference/qmath_ref.py).
The driver sequence written here is the one the firmware uses (fw/src/npu.c npu_matmul): weights per (K tile, C_out tile), ACC_PASS flags,
vectors pushed with IN FIFO polling, results popped while pushing in the LAST pass.

  python hw/tb/shell/gen_shell_tests.py N ACC_DEPTH out_file [seed] [XR_DEPTH]

Script lines:  w ADDR DATA | r ADDR EXPECT MASK | p ADDR VALUE MASK (poll until (rd & MASK) == VALUE) | l ADDR VALUE MASK (poll until (rd & MASK) <= VALUE)
               c TEXT (comment printed by the TB) | e (end)
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "reference"))
from qmath_ref import rq

ID, CFG, CTRL, STATUS, W_DATA, W_COMMIT, X_DATA, Y_DATA = 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C
RQ_MULT, RQ_SHIFT, COL_IDX, BIAS, PERF_CTRL, ERR_CLR, SCRATCH, OUT_ZP, ACC_PASS, ACC_INFO = 0x20, 0x24, 0x28, 0x2C, 0x3C, 0x40, 0x44, 0x4C, 0x58, 0x5C
XR_WPTR, XR_REPLAY, XR_INFO = 0x60, 0x64, 0x68
RP_ACTIVE = 1 << 20
IN_DEPTH = 32


class Script:
    def __init__(self):
        self.lines = []
        self.apb = 0

    def w(self, a, d):
        self.lines.append(f"w {a:03x} {d & 0xFFFFFFFF:08x} 0"); self.apb += 1

    def r(self, a, exp, mask=0xFFFFFFFF):
        self.lines.append(f"r {a:03x} {exp & 0xFFFFFFFF:08x} {mask:08x}"); self.apb += 1

    def p(self, a, v, mask):
        self.lines.append(f"p {a:03x} {v:08x} {mask:08x}"); self.apb += 1

    def l(self, a, v, mask):
        self.lines.append(f"l {a:03x} {v:08x} {mask:08x}"); self.apb += 1

    def c(self, t):
        self.lines.append("c " + t.replace(" ", "_"))


def pack(vals):
    """4 int8 values -> one little-endian word"""
    return sum((int(v) & 0xFF) << (8 * b) for b, v in enumerate(vals))


def matmul_ops(s, N, ACC_DEPTH, X, Wm, rq_en, relu=0, zp=0, M=None, S=None, bias=None, use_acc=True, newline_every=None, xr=0):
    """X [V][K] int8, Wm [K][Co] int8. Emits the driver sequence and the expected reads. Returns nothing (expectations are in the script).
    xr > 0: X replay buffer of xr vectors - the first C_out tile pushes (and records) the chunk, the others replay it."""
    V, K = X.shape
    Co = Wm.shape[1]
    kt_n = -(-K // N); ct_n = -(-Co // N)
    acc = X.astype(np.int64) @ Wm.astype(np.int64)
    acc_mode = use_acc and kt_n > 1
    s.p(STATUS, 0, 1 << 16)
    s.w(CTRL, (rq_en << 1) | (relu << 2) | (1 << 4) | (int(acc_mode) << 5) | (int(xr > 0) << 6))
    if not acc_mode and kt_n > 1:
        raise ValueError("K > N needs the accumulator in this generator")
    chunk = ACC_DEPTH if acc_mode else V
    if xr:
        chunk = min(chunk, xr // kt_n)
    for v0 in range(0, V, chunk):
        v1 = min(V, v0 + chunk)
        n = v1 - v0
        for ct in range(ct_n):
            s.p(STATUS, 0, 1 << 16)                                      # postproc parameters change only when idle
            if rq_en:
                s.w(OUT_ZP, zp & 0xFF)
                s.w(COL_IDX, 0)
                for c in range(N):
                    co = ct * N + c
                    s.w(RQ_MULT, M[co] if co < Co else 1)
                    s.w(RQ_SHIFT, S[co] if co < Co else 0)
                    s.w(BIAS, bias[co] if co < Co else 0)
            for kt in range(kt_n):
                s.p(STATUS, 1 << 17, 1 << 17)                            # W_READY
                for r in range(N - 1, -1, -1):
                    k = kt * N + r
                    row = [Wm[k, ct * N + c] if (k < K and ct * N + c < Co) else 0 for c in range(N)]
                    for wd in range(N // 4):
                        s.w(W_DATA, pack(row[4 * wd:4 * wd + 4]))
                last = kt == kt_n - 1
                if xr and ct > 0:                                        # replay the recorded K slice
                    s.p(STATUS, 0, RP_ACTIVE)
                    s.w(W_COMMIT, 1)
                    if acc_mode:
                        s.w(ACC_PASS, int(kt == 0) | (int(last) << 1))
                    s.w(XR_REPLAY, (n << 16) | (kt * n))
                    if last:
                        for v in range(v0, v1):
                            pop_one(s, N, acc, v, ct, Co, rq_en, relu, zp, M, S, bias)
                    continue
                s.w(W_COMMIT, 1)
                if acc_mode:
                    s.w(ACC_PASS, int(kt == 0) | (int(last) << 1))
                if xr:
                    s.w(XR_WPTR, kt * n)
                popped = v0
                for v in range(v0, v1):
                    if (v - v0) % 8 == 0:
                        s.l(STATUS, IN_DEPTH - 8, 0xFF)                  # room for 8 more vectors
                    xs = [X[v, kt * N + r] if kt * N + r < K else 0 for r in range(N)]
                    for wd in range(N // 4):
                        s.w(X_DATA, pack(xs[4 * wd:4 * wd + 4]))
                    if last:
                        while v + 1 - popped > 24:                       # keep the OUT FIFO draining in the last pass
                            pop_one(s, N, acc, popped, ct, Co, rq_en, relu, zp, M, S, bias)
                            popped += 1
                if last:
                    while popped < v1:
                        pop_one(s, N, acc, popped, ct, Co, rq_en, relu, zp, M, S, bias)
                        popped += 1


def pop_one(s, N, acc, v, ct, Co, rq_en, relu, zp, M, S, bias):
    s.p(STATUS, 0, 1 << 19)                                              # OUT not empty
    cols = []
    for c in range(N):
        co = ct * N + c
        a = int(acc[v, co]) if co < Co else 0
        if rq_en:
            cols.append(rq(a, bias[co], M[co], S[co], zp, relu) if co < Co else rq(0, 0, 1, 0, zp, relu))
        else:
            cols.append(((a + 2**31) % 2**32) - 2**31)
    if rq_en:
        for wd in range(N // 4):
            s.r(Y_DATA, pack(cols[4 * wd:4 * wd + 4]))
    else:
        for c in range(N):
            s.r(Y_DATA, cols[c])


def main():
    N = int(sys.argv[1]); ACC_DEPTH = int(sys.argv[2]); out = sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    XR = int(sys.argv[5]) if len(sys.argv) > 5 else 256
    rng = np.random.default_rng(seed)
    s = Script()
    s.c(f"registers N={N}")
    s.r(ID, 0x53514431)
    s.r(CFG, N | (32 << 8) | (3 << 24), 0xFF00FFFF)
    s.r(ACC_INFO, ACC_DEPTH)
    s.r(XR_INFO, XR)
    s.w(SCRATCH, 0xCAFE1234); s.r(SCRATCH, 0xCAFE1234)
    s.w(PERF_CTRL, 3)
    s.c("error: Y read while empty")
    s.r(Y_DATA, 0xDEADBEEF)
    s.r(STATUS, 1 << 26, 1 << 26)
    s.w(ERR_CLR, 1 << 26)
    s.r(STATUS, 0, 1 << 26)

    def rand_rq(Co):
        M = rng.integers(1, 32768, Co).tolist(); S = rng.integers(8, 24, Co).tolist()
        b = rng.integers(-200000, 200000, Co).tolist()
        return M, S, b

    cases = [  # (name, V, K, Co, rq_en, relu, zp, zero_frac)
        ("raw K=N", 40, N, N, 0, 0, 0, 0.0),
        ("rq K=N relu", 50, N, N, 1, 1, -128, 0.3),
        ("rq K=27 (enc0-like) acc", 70, 27, 32, 1, 1, -128, 0.0),
        ("raw K=3N acc", 33, 3 * N, N, 0, 0, 0, 0.0),
        ("rq K=100 Co=40 acc", 50, 100, 40, 1, 1, -128, 0.2),
        ("rq K=2N V=1 acc", 1, 2 * N, N, 1, 0, 5, 0.0),
        ("rq K=2N V=2 acc", 2, 2 * N, N, 1, 0, -3, 0.0),
        (f"rq K=32 V>ACC_DEPTH chunks", ACC_DEPTH + 37, 32, 2 * N, 1, 1, -128, 0.1),
    ]
    for (name, V, K, Co, rq_en, relu, zp, zf) in cases:
        s.c(name)
        X = rng.integers(-128, 128, (V, K)); X[rng.random((V, K)) < zf] = 0
        Wm = rng.integers(-127, 128, (K, Co))
        M, S, b = rand_rq(Co)
        a0 = s.apb
        matmul_ops(s, N, ACC_DEPTH, X, Wm, rq_en, relu, zp, M, S, b)
        kt = -(-K // N); ct = -(-Co // N)
        raw_cpu = V * ct * kt * (N // 4 + N) + kt * ct * N * N // 4      # v1.1 driver: RAW read-back of every K tile
        s.c(f"APB_accesses_{s.apb - a0}_(RAW+CPU-sum_would_need_about_{raw_cpu})")
        if kt <= XR:                                                  # same case with the X replay buffer (same expected results)
            s.c(name + " REPLAY")
            a1 = s.apb
            matmul_ops(s, N, ACC_DEPTH, X, Wm, rq_en, relu, zp, M, S, b, xr=XR)
            s.c(f"APB_accesses_with_replay_{s.apb - a1}")

    s.c("weight swap while vectors are queued (K=N, raw)")
    X = rng.integers(-128, 128, (20, N)); W1 = rng.integers(-127, 128, (N, N)); W2 = rng.integers(-127, 128, (N, N))
    s.p(STATUS, 0, 1 << 16)
    s.w(CTRL, 1 << 4)
    for Wt, xs in ((W1, X[:10]), (W2, X[10:])):
        s.p(STATUS, 1 << 17, 1 << 17)
        for r in range(N - 1, -1, -1):
            for wd in range(N // 4):
                s.w(W_DATA, pack(Wt[r, 4 * wd:4 * wd + 4]))
        s.w(W_COMMIT, 1)
        for x in xs:
            for wd in range(N // 4):
                s.w(X_DATA, pack(x[4 * wd:4 * wd + 4]))
    exp = np.vstack([X[:10] @ W1, X[10:] @ W2])
    for v in range(20):
        s.p(STATUS, 0, 1 << 19)
        for c in range(N):
            s.r(Y_DATA, int(exp[v, c]))

    s.c("accumulator read-after-write: two passes on slot 0 issued on consecutive clocks (SLEEP holds them in the IN FIFO)")
    Wt = rng.integers(-127, 128, (N, N)); x1 = rng.integers(-128, 128, N); x2 = rng.integers(-128, 128, N); x3 = rng.integers(-128, 128, N)
    s.p(STATUS, 0, 1 << 16)
    s.w(CTRL, (1 << 4) | (1 << 5) | (1 << 3))                    # RAW, ACC_EN, SLEEP
    s.p(STATUS, 1 << 17, 1 << 17)
    for r in range(N - 1, -1, -1):
        for wd in range(N // 4):
            s.w(W_DATA, pack(Wt[r, 4 * wd:4 * wd + 4]))
    s.w(W_COMMIT, 1)
    for flags, x in ((1, x1), (0, x2), (2, x3)):                 # FIRST, middle, LAST - all slot 0
        s.w(ACC_PASS, flags)
        for wd in range(N // 4):
            s.w(X_DATA, pack(x[4 * wd:4 * wd + 4]))
    s.w(CTRL, (1 << 4) | (1 << 5))                               # release: the three vectors issue back to back
    exp = x1 @ Wt + x2 @ Wt + x3 @ Wt
    s.p(STATUS, 0, 1 << 19)
    for c in range(N):
        s.r(Y_DATA, int(exp[c]))
    s.p(STATUS, 0, 1 << 16)
    s.w(CTRL, 1 << 4)

    s.c("error: W_DATA while W_READY=0 is dropped")
    s.w(W_COMMIT, 1)                               # commit pending -> W_READY = 0
    s.r(STATUS, 0, 1 << 17)
    s.w(W_DATA, 0x01010101)
    s.r(STATUS, 1 << 24, 1 << 24)
    s.w(ERR_CLR, 1 << 24)
    s.r(STATUS, 0, 1 << 24)
    s.c("soft reset clears the pending commit")
    s.w(CTRL, 1 | (1 << 4))
    s.p(STATUS, 1 << 17, 1 << 17)
    s.r(0x30, 0, 0)                                # PERF_CYCLES readable (value not checked)
    s.lines.append("e")
    open(out, "w").write("\n".join(s.lines) + "\n")
    print(f"{out}: {len(s.lines)} lines")


if __name__ == "__main__":
    main()
