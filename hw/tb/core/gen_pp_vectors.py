"""Vectors for tb_postproc, expected values from reference/qmath_ref.py rq().
Usage: python hw/tb/core/gen_pp_vectors.py N NB K prefix   -> prefix_{acc,exp,m,s,bias,zp,relu}.hex
Layout: params indexed b*N+c (zp, relu indexed b); acc/exp indexed (b*K+v)*N+c. One batch = static params, K vectors."""
import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "reference"))
from qmath_ref import rq

N, NB, K, pfx = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
rnd = random.Random(12345 + N)
I32 = (-(1 << 31), (1 << 31) - 1)


def r32():
    return rnd.randint(*I32)


def pick_bias():
    return rnd.choice([0, 0, r32(), I32[0], I32[1], rnd.randint(-5000, 5000), -1, 1])


def pick_acc(M, S, bias):
    k = rnd.randrange(9)
    if k == 0:
        return 0
    if k == 1:
        return rnd.choice(I32)
    if k == 2:
        return r32()
    if k == 3:
        return rnd.randint(-3000, 3000)
    if k == 4:  # exact .5 tie: (acc+bias)*M == 2^(S-1) mod 2^S
        if S >= 1 and M % 2 == 1:
            m = 1 << S
            base = ((1 << (S - 1)) * pow(M, -1, m)) % m
            for _ in range(8):
                s = base + rnd.randint(-(1 << 20), 1 << 20) * m if S < 31 else base - (m if rnd.random() < .5 else 0)
                a = s - bias
                if I32[0] <= a <= I32[1]:
                    return a
        return rnd.randint(-100, 100)
    if k in (5, 6):  # near the saturation edges for this M,S
        target = rnd.choice([127, 128, -128, -129, 0, 255, -256]) + rnd.randint(-2, 2)
        s = (target << S) // M
        a = s - bias
        return a if I32[0] <= a <= I32[1] else r32()
    return rnd.randint(-(1 << 20), 1 << 20)


H = {k: [] for k in "acc exp m s bias zp relu".split()}
for b in range(NB):
    zp = rnd.choice([-128, 0, 127, rnd.randint(-128, 127), rnd.randint(-128, 127)])
    relu = rnd.randint(0, 1)
    Ms, Ss, Bs = [], [], []
    for c in range(N):
        Ms.append(rnd.choice([1, 32767, rnd.randint(1, 32767), rnd.randint(1, 32767), rnd.randint(1, 64)]))
        Ss.append(rnd.choice([0, 31, rnd.randint(0, 31), rnd.randint(0, 31), rnd.randint(10, 20)]))
        Bs.append(pick_bias())
    H["zp"].append(zp & 0xFF)
    H["relu"].append(relu)
    for c in range(N):
        H["m"].append(Ms[c])
        H["s"].append(Ss[c])
        H["bias"].append(Bs[c] & 0xFFFFFFFF)
    for v in range(K):
        for c in range(N):
            a = pick_acc(Ms[c], Ss[c], Bs[c])
            H["acc"].append(a & 0xFFFFFFFF)
            H["exp"].append(rq(a, Bs[c], Ms[c], Ss[c], zp, relu) & 0xFF)
os.makedirs(os.path.dirname(pfx) or ".", exist_ok=True)
w = {"acc": 8, "exp": 2, "m": 4, "s": 2, "bias": 8, "zp": 2, "relu": 1}
for k, vals in H.items():
    with open(f"{pfx}_{k}.hex", "w") as f:
        f.write("\n".join(f"{x:0{w[k]}x}" for x in vals) + "\n")
print(f"{pfx}: {NB*K} vectors x {N} columns = {NB*K*N} cases")
