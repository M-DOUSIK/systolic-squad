"""Independent check of the depth-model exports Does NOT import ml/convert.py, ml/qmodel.py or
ml/export_c.py: everything is re-derived from model_q.json with reference/*.py only.

  python scripts/check_exports.py [n_frames] [export dir name]   (default: 2 frames recomputed with the slow reference functions, ml/export)

Checks: 1 model_q.json structure, shape chain, value ranges, preproc/output/gate blocks
        2 model_blob.bin re-packed from model_q.json byte for byte (layout), size + CRC in the model table
        3 every row of fw/include/model_<name>.h against model_q.json + the re-derived offsets
        4 golden vectors: every layer of n frames recomputed with qmath_ref (im2col + matmul_tiled + rq, dwconv_q, add_q, upsample_nn),
          _depth = q + 128, _edge = canny_ref, _zones = depth_ref over the whole sequence
        5 hw/tb/vectors: Y = X W and Yq = rq(...) re-derived from the hex files
Prints PASS/FAIL per check and exits 1 on any failure."""
import json, os, re, struct, sys, zlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "reference"))
import qmath_ref as R
from canny_ref import canny_ref
from depth_ref import rgb_to_gray, zone_nearness, ObstacleAgent

EXPN = sys.argv[2] if len(sys.argv) > 2 else "export"
EXP = os.path.join(ROOT, "ml", EXPN)
VEC = os.path.join(EXP, "vectors")
HDR = os.path.join(ROOT, "fw", "include", "model_" + ("fastdepth" if EXPN == "export" else EXPN.replace("export_", "")) + ".h")
TYPES = {"conv": 0, "pwconv": 1, "dwconv": 2, "upsample": 3, "add": 4, "lut": 5, "fc": 6, "maxpool": 7, "avgpool": 8, "flatten": 9,
         "relu": 10, "bilinear": 11}
fails = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        fails.append(name)


def main():
    nfr = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    mq = json.load(open(os.path.join(EXP, "model_q.json")))
    L = mq["layers"]
    W, H = mq["preproc"]["width"], mq["preproc"]["height"]

    # ---------------------------------------------------------------- 1 structure
    errs = []
    shp = {-2: (H, W, 3)}
    for i, l in enumerate(L):
        for j in l["inputs"]:
            if not (j == -2 or 0 <= j < i):
                errs.append(f"{i}: input {j}")
        h, w, c = shp[l["inputs"][0]]
        if (l["in_H"], l["in_W"], l["C_in"]) != (h, w, c):
            errs.append(f"{i} {l['name']}: in shape {(l['in_H'], l['in_W'], l['C_in'])} != producer {(h, w, c)}")
        t = l["type"]
        if t in ("conv", "pwconv", "dwconv"):
            p = l["pad"]
            ho = (h + p[0] + p[2] - l["KH"]) // l["stride"] + 1
            wo = (w + p[1] + p[3] - l["KW"]) // l["stride"] + 1
            co = l["C_out"]
            nW = l["KH"] * l["KW"] * (1 if t == "dwconv" else c) * co
            if t == "dwconv" and co != c: errs.append(f"{i}: dw C_out != C_in")
            if t == "pwconv" and (l["KH"], l["KW"]) != (1, 1): errs.append(f"{i}: pwconv kernel")
            if len(l["W"]) != nW: errs.append(f"{i}: |W| {len(l['W'])} != {nW}")
            if not all(-127 <= v <= 127 for v in l["W"]): errs.append(f"{i}: W out of [-127,127]")
            if len(l["bias"]) != co or len(l["M"]) != co or len(l["S"]) != co: errs.append(f"{i}: per-channel vector length")
            if not all(-2**31 <= v < 2**31 for v in l["bias"]): errs.append(f"{i}: bias not int32")
            if not all(1 <= v <= 32767 for v in l["M"]): errs.append(f"{i}: M range")
            if not all(0 <= v <= 31 for v in l["S"]): errs.append(f"{i}: S range")
            # bias' really is the zero-point folded bias of the stored integers: recover b = bias' + in_zp * sum(W) and check it is consistent
            # with the documented scales (|b * s_in * s_w - real bias| is unknown here, so only the integer identity is checked)
            sh = (ho, wo, co)
        elif t in ("upsample", "bilinear"):
            f = l["factor"]
            if l["in_zp"] != l["out_zp"]: errs.append(f"{i}: {t} changes zero point")
            sh = (h * f, w * f, c)
        elif t == "relu":
            if l["in_zp"] != l["out_zp"]: errs.append(f"{i}: relu changes zero point")
            sh = (h, w, c)
        elif t == "add":
            if shp[l["inputs"][1]] != (h, w, c): errs.append(f"{i}: add shapes differ")
            if not (1 <= l["Ma"] <= 32767 and 1 <= l["Mb"] <= 32767 and 0 <= l["S_add"] <= 30): errs.append(f"{i}: add params")
            if l["za"] != (L[l["inputs"][0]]["out_zp"] if l["inputs"][0] >= 0 else -128) or l["zb"] != (L[l["inputs"][1]]["out_zp"] if l["inputs"][1] >= 0 else -128):
                errs.append(f"{i}: add zero points != producers' out_zp")
            sh = (h, w, c)
        else:
            errs.append(f"{i}: unexpected type {t}"); sh = (h, w, c)
        if (l["out_H"], l["out_W"]) != sh[:2]: errs.append(f"{i}: out shape")
        if not (-128 <= l["in_zp"] <= 127 and -128 <= l["out_zp"] <= 127): errs.append(f"{i}: zero point range")
        # the zero point a layer reads must be the one its producer wrote
        pz = -128 if l["inputs"][0] == -2 else L[l["inputs"][0]]["out_zp"]
        if l["in_zp"] != pz: errs.append(f"{i}: in_zp {l['in_zp']} != producer out_zp {pz}")
        shp[i] = sh
    if shp[len(L) - 1] != (H, W, 1): errs.append(f"final shape {shp[len(L) - 1]}")
    o = mq["output"]
    if o["zero_point"] != L[-1]["out_zp"] or abs(o["scale"] - L[-1]["scales"]["s_out"]) > 1e-12: errs.append("output block != last layer")
    if mq["preproc"]["input_transform"] != "u8_minus_128" or W % 32 or H % 32: errs.append("preproc")
    if not (0 <= mq["gate"]["canny_low"] <= mq["gate"]["canny_high"] <= 1016): errs.append("gate")
    check("1 model_q.json structure / shapes / ranges", not errs, "; ".join(errs[:5]) or f"{len(L)} layers, {W}x{H}")

    # ---------------------------------------------------------------- 2 blob
    blob = bytearray(); offs = []
    def put(b):
        while len(blob) % 4: blob.append(0)
        o_ = len(blob); blob.extend(b); return o_
    for l in L:
        d = dict(w=0, b=0, m=0, s=0, lut=0)
        if "W" in l:
            d["w"] = put(struct.pack(f"<{len(l['W'])}b", *l["W"]))
            d["b"] = put(struct.pack(f"<{len(l['bias'])}i", *l["bias"]))
            d["m"] = put(struct.pack(f"<{len(l['M'])}H", *l["M"]))
            d["s"] = put(struct.pack(f"<{len(l['S'])}B", *l["S"]))
        offs.append(d)
    while len(blob) % 4: blob.append(0)
    disk = open(os.path.join(EXP, "model_blob.bin"), "rb").read()
    check("2 model_blob.bin == re-packed model_q.json", bytes(blob) == disk, f"{len(disk)} B, CRC {zlib.crc32(disk) & 0xFFFFFFFF:08X}")

    # ---------------------------------------------------------------- 3 header
    hdr = open(HDR).read()
    common = open(os.path.join(ROOT, "fw", "include", "model.h")).read()
    errs = []
    gi = lambda k: int(re.search(rf"#define {k}\s+\(?(-?[0-9xA-Fa-f]+)u?\)?", common).group(1), 0)
    desc = re.search(r"model_desc_t MODEL_DESC_\w+ = \{\s*\"[^\"]*\",\s*\w+,\s*(\d+),\s*(\d+)u,\s*(0x[0-9A-F]+)u,\s*(\d),\s*(-?\d+),\s*(-?\d+)", hdr)
    if not desc: errs.append("model_desc_t not found")
    else:
        n_l, nb, crc_, kind, sc6, zp_ = desc.groups()
        if int(nb) != len(disk) or int(crc_, 16) != zlib.crc32(disk) & 0xFFFFFFFF: errs.append("blob bytes / CRC")
        if int(n_l) != len(L): errs.append("n_layers")
        if int(kind) != (0 if o["type"] == "depth" else 1) or int(sc6) != round(o["scale"] * 1e6) or int(zp_) != o["zero_point"]: errs.append("output")
    if gi("PREPROC_W") != W or gi("PREPROC_H") != H: errs.append("PREPROC")
    if gi("GATE_CANNY_LOW") != mq["gate"]["canny_low"] or gi("GATE_CANNY_HIGH") != mq["gate"]["canny_high"]: errs.append("GATE_*")
    rows = re.findall(r"/\*\s*(\d+)\s+\S+\s*\*/\s*\{([^}]*)\}", hdr)
    if len(rows) != len(L): errs.append(f"{len(rows)} rows")
    for (idx, body), l, d in zip(rows, L, offs):
        v = [int(x.strip().rstrip("u")) for x in body.split(",") if x.strip()]
        p = l.get("pad", [0, 0, 0, 0]); ins = l["inputs"]
        want = [TYPES[l["type"]], l["relu"], l["stride"], l["factor"], l["KH"], l["KW"], l["C_in"], l["C_out"], l["in_H"], l["in_W"],
                l["out_H"], l["out_W"], *p, l["in_zp"], l["out_zp"], l.get("za", 0), l.get("zb", 0), ins[0], ins[1] if len(ins) > 1 else 0,
                l.get("Ma", 0), l.get("Mb", 0), l.get("S_add", 0), l.get("align", 0), l.get("pad_value", l["in_zp"]),
                d["w"], d["b"], d["m"], d["s"], d["lut"]]
        if v != want: errs.append(f"row {idx}")
    check(f"3 {os.path.basename(HDR)} rows / constants", not errs, "; ".join(errs[:5]) or f"{len(rows)} rows")

    # ---------------------------------------------------------------- 4 golden vectors
    errs = []
    nframes = len([f for f in os.listdir(VEC) if re.fullmatch(r"frame_\d{3}\.bin", f)])
    agent = ObstacleAgent()
    for fr in range(nframes):
        rgb = np.fromfile(os.path.join(VEC, f"frame_{fr:03d}.bin"), np.uint8).reshape(H, W, 3)
        edge = np.fromfile(os.path.join(VEC, f"frame_{fr:03d}_edge.bin"), np.int8).reshape(H, W)
        if not (edge == np.asarray(canny_ref(rgb_to_gray(rgb).astype(np.uint8), mq["gate"]["canny_low"], mq["gate"]["canny_high"]))).all():
            errs.append(f"frame {fr} edge")
        gold = [np.fromfile(os.path.join(VEC, f"frame_{fr:03d}_L{i}.bin"), np.int8).astype(np.int64) for i in range(len(L))]
        if fr < nfr:                                            # full slow recomputation
            x0 = rgb.astype(np.int64) - 128
            for i, l in enumerate(L):
                src = [x0 if j == -2 else gold[j].reshape(L[j]["out_H"], L[j]["out_W"], L[j]["C_out"]) for j in l["inputs"]]
                t = l["type"]
                if t in ("conv", "pwconv"):
                    Wm = np.array(l["W"], np.int64).reshape(-1, l["C_out"])
                    X = R.im2col(src[0], l["KH"], l["KW"], l["stride"], tuple(l["pad"]), l.get("pad_value", l["in_zp"]))
                    acc = R.matmul_tiled(X, Wm, 16)
                    y = np.array([[R.rq(acc[v, c], l["bias"][c], l["M"][c], l["S"][c], l["out_zp"], l["relu"]) for c in range(l["C_out"])]
                                  for v in range(acc.shape[0])])
                elif t == "dwconv":
                    y = R.dwconv_q(src[0], np.array(l["W"], np.int64).reshape(l["KH"], l["KW"], l["C_out"]), l["bias"], l["M"], l["S"],
                                   l["out_zp"], l["relu"], l["stride"], tuple(l["pad"]), l.get("pad_value", l["in_zp"]))
                elif t == "add":
                    y = R.add_q(src[0], src[1], l["za"], l["zb"], l["Ma"], l["Mb"], l["S_add"], l["out_zp"], l["relu"])
                elif t == "relu":
                    y = R.relu_q(src[0], l["out_zp"])
                elif t == "bilinear":
                    y = R.bilinear_q(src[0], l["factor"], bool(l["align"]))
                else:
                    y = R.upsample_nn(src[0], l["factor"])
                if not (np.asarray(y).reshape(-1) == gold[i]).all():
                    errs.append(f"frame {fr} layer {i} {l['name']}")
        depth = np.fromfile(os.path.join(VEC, f"frame_{fr:03d}_depth.bin"), np.uint8)
        if not (depth.astype(np.int64) == gold[-1] + 128).all(): errs.append(f"frame {fr} depth != L_last + 128")
        n = zone_nearness(depth.reshape(H, W).astype(np.int64), o["type"])
        r = agent.update(n)
        z = np.fromfile(os.path.join(VEC, f"frame_{fr:03d}_zones.bin"), np.uint8)
        if list(z) != list(n) + [r["stable"]]: errs.append(f"frame {fr} zones")
    check(f"4 golden vectors ({nfr} frames x {len(L)} layers recomputed, {nframes} frames edge/depth/zones)", not errs, "; ".join(errs[:5]))

    # ---------------------------------------------------------------- 5 hw vectors
    errs = []
    hv = os.path.join(ROOT, "hw", "tb", "vectors")
    rd = lambda f, bits: [((int(x, 16) + (1 << (bits - 1))) % (1 << bits)) - (1 << (bits - 1)) for x in open(f).read().split()]
    cases = sorted({f.split("_")[0] if not f.startswith("canny") else "_".join(f.split("_")[:2]) for f in os.listdir(hv) if f.endswith("_W.hex")})
    N = 16
    for c in cases:
        Wv = np.array(rd(os.path.join(hv, f"{c}_W.hex"), 8)).reshape(-1, N)
        Xv = np.array(rd(os.path.join(hv, f"{c}_X.hex"), 8)).reshape(-1, Wv.shape[0])
        Yv = np.array(rd(os.path.join(hv, f"{c}_Y.hex"), 32)).reshape(-1, N)
        if not (Xv @ Wv == Yv).all(): errs.append(f"{c} Y")
        qf = os.path.join(hv, f"{c}_Yq.hex")
        if os.path.exists(qf):
            cfg = open(os.path.join(hv, f"{c}_cfg.hex")).read().split("\n")
            relu = int(cfg[0], 16); zp = rd_one = ((int(cfg[1], 16) + 128) % 256) - 128
            prm = [l.split() for l in cfg[2:2 + N]]
            Yq = np.array(rd(qf, 8)).reshape(-1, N)
            for v in range(Yq.shape[0]):
                for k in range(N):
                    M_, S_, b_ = int(prm[k][0], 16), int(prm[k][1], 16), ((int(prm[k][2], 16) + 2**31) % 2**32) - 2**31
                    if R.rq(int(Yv[v, k]), b_, M_, S_, zp, relu) != Yq[v, k]:
                        errs.append(f"{c} Yq[{v},{k}]"); break
    check(f"5 hw/tb/vectors ({', '.join(cases)})", not errs, "; ".join(errs[:5]))
    print("check_exports:", "ALL PASS" if not fails else f"FAILED: {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
