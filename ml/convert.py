"""FastDepth (float, PyTorch) -> integer layer list `ml/export/model_q.json`.

  python ml/convert.py [--ckpt PATH] [--width 128] [--height 96] [--calib 64]

What it does:
  1. folds every BatchNorm into the conv before it and writes the network as a flat layer list in the order of
     (enc0, 13 x (dw, pw), decoder 5 x (dw, pw, upsample [, add]), dec6pw at half resolution, final upsample);
  2. checks that this float layer list reproduces the PyTorch model (max abs difference printed, must be < 1e-3 m);
  3. calibrates one activation range per tensor on `--calib` NYU TRAIN frames (99.99th percentile; encoder layers are ReLU6, so their
     range is capped at 6.0 and the clamp at 127 IS the ReLU6);
  4. quantises: weights INT8 symmetric per output channel, bias INT32 at scale s_in*s_w[c], input zero point folded into the bias,
     (M, S) from reference/qmath_ref.choose_M_S, skip adds via choose_add_params; network input x = p - 128 with RGB/255 folded into layer 0.
All integer maths used later (ml/qmodel.py, firmware, RTL) comes from reference/qmath_ref.py; floats are only used here to pick integers.
"""
import argparse, glob, json, os, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "reference"))
import fd_common as C
from qmath_ref import choose_M_S, choose_add_params, fold_input_zp, fold_input_zp_dw

EXPORT = os.path.join(HERE, "export")


# ----------------------------------------------------------------------------------------------------------------------------
# 1. float layer list with BN folded
# ----------------------------------------------------------------------------------------------------------------------------
def fold(seq):
    """nn.Sequential(conv, bn, act) -> (w [Co,Ci/g,KH,KW], b [Co], conv, act_name)"""
    conv, bn, act = seq[0], seq[1], seq[2]
    g = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
    w = conv.weight.detach() * g[:, None, None, None]
    b = (bn.bias - bn.running_mean * g).detach()
    if conv.bias is not None:
        b = b + conv.bias.detach() * g
    a = "relu6" if isinstance(act, nn.ReLU6) else "relu"
    return w.double().numpy(), b.double().numpy(), conv, a


def conv_layer(name, seq):
    w, b, conv, act = fold(seq)
    p = conv.padding[0]
    kind = "dwconv" if conv.groups > 1 else ("pwconv" if conv.kernel_size == (1, 1) else "conv")
    return dict(name=name, type=kind, KH=conv.kernel_size[0], KW=conv.kernel_size[1], stride=conv.stride[0], pad=[p, p, p, p],
                C_in=conv.in_channels, C_out=conv.out_channels, act=act, w=w, b=b, inputs=[-1])


def build_float_layers(model):
    e, d = model.encoder, model.decoder
    L = [conv_layer("enc0", e.enc_layer1)]
    for i in range(13):
        blk = getattr(e, f"enc_layer{i + 2}")
        L.append(conv_layer(f"enc{i + 1}dw", blk[0]))
        L.append(conv_layer(f"enc{i + 1}pw", blk[1]))
    idx = {l["name"]: i for i, l in enumerate(L)}
    skips = {2: "enc5pw", 3: "enc3pw", 4: "enc1pw"}              # decoder stage -> encoder output added after its upsample
    for s in range(1, 6):
        blk = getattr(d, f"conv{s}")
        L.append(conv_layer(f"dec{s}dw", blk[0]))
        L.append(conv_layer(f"dec{s}pw", blk[1]))
        if s < 5:                                                 # dec5: the upsample moves behind dec6pw
            L.append(dict(name=f"dec{s}up", type="upsample", factor=2, inputs=[-1]))
        if s in skips:
            L.append(dict(name=f"dec{s}add", type="add", inputs=[len(L) - 1, idx[skips[s]]]))
    L.append(conv_layer("dec6pw", d.output))
    L.append(dict(name="out_up", type="upsample", factor=2, inputs=[-1]))
    for i, l in enumerate(L):                                     # resolve -1 = previous layer into absolute indices (-2 = network input)
        l["inputs"] = [(i - 1 if i > 0 else -2) if j == -1 else j for j in l["inputs"]]
    return L


def float_exec(L, x):
    """x: torch [B,3,H,W] float (RGB/255). Returns list of every layer's output (torch NCHW)."""
    outs = []
    for l in L:
        src = [x if j == -2 else outs[j] for j in l["inputs"]]
        t = l["type"]
        if t in ("conv", "pwconv", "dwconv"):
            y = F.conv2d(src[0], torch.from_numpy(l["w"]).to(src[0].dtype), torch.from_numpy(l["b"]).to(src[0].dtype),
                         stride=l["stride"], padding=l["pad"][0], groups=l["C_in"] if t == "dwconv" else 1)
            y = torch.clamp(y, 0, 6) if l["act"] == "relu6" else torch.relu(y)
        elif t == "upsample":
            y = F.interpolate(src[0], scale_factor=l["factor"], mode="nearest")
        elif t == "add":
            y = src[0] + src[1]
        outs.append(y)
    return outs


# ----------------------------------------------------------------------------------------------------------------------------
# 2. quantisation
# ----------------------------------------------------------------------------------------------------------------------------
def act_qparams(hi):
    """non-negative tensor with range [0, hi] -> (scale, zero_point): zp = -128 so real 0 = q -128 and real hi = q 127"""
    return float(hi) / 255.0, -128


def quantise(L, ranges, W, H):
    s_in, z_in = 1.0 / 255.0, -128                     # x = p - 128, real = p/255 = (x + 128)/255  -> scale 1/255, zero point -128
    sc, zp, shp = {-2: s_in}, {-2: z_in}, {-2: (H, W, 3)}
    Q = []
    for i, l in enumerate(L):
        a = l["inputs"][0]
        h, w_, cin = shp[a]
        q = dict(name=l["name"], type=l["type"], inputs=l["inputs"], in_H=h, in_W=w_, C_in=cin)
        t = l["type"]
        if t in ("conv", "pwconv", "dwconv"):
            hi = min(ranges[i], 6.0) if l["act"] == "relu6" else ranges[i]
            so, zo = act_qparams(hi)
            si, zi = sc[a], zp[a]
            wf, bf = l["w"], l["b"]
            Co = l["C_out"]
            if t == "dwconv":
                wk = wf[:, 0].transpose(1, 2, 0)                                     # [KH][KW][C]
                amax = np.abs(wk).reshape(-1, Co).max(axis=0)
            else:
                wk = wf.transpose(2, 3, 1, 0).reshape(-1, Co)                       # Wm[k][co], k = (ky*KW + kx)*C_in + ci
                amax = np.abs(wk).max(axis=0)
            sw = np.where(amax > 0, amax / 127.0, so / si)                           # dead channel: weights 0, multiplier 1
            sw = np.maximum(sw, np.abs(bf) / (si * 2**29))                           # near-dead channel: keep the bias inside int32
            sw = np.maximum(sw, so / (si * 2**29))                                   # ... and the multiplier representable by (M, S <= 31)
            Wq = np.clip(np.round(wk / sw), -127, 127).astype(np.int64)
            bq = np.round(bf / (si * sw))
            assert np.all(np.abs(bq) < 2**31 - 2**24), f"{l['name']}: bias overflows int32"
            bq = bq.astype(np.int64)
            bias_f = fold_input_zp_dw(bq, Wq, zi) if t == "dwconv" else fold_input_zp(bq, Wq, zi)
            MS = [choose_M_S(si * sw[c] / so) for c in range(Co)]
            KH, KW, st, p = l["KH"], l["KW"], l["stride"], l["pad"]
            ho, wo = (h + p[0] + p[2] - KH) // st + 1, (w_ + p[1] + p[3] - KW) // st + 1
            q.update(KH=KH, KW=KW, stride=st, pad=p, factor=1, C_out=Co, in_zp=zi, out_zp=zo, relu=1,
                     M=[m for m, _ in MS], S=[s for _, s in MS], W=Wq.reshape(-1).tolist(), bias=bias_f.tolist(),
                     scales=dict(s_in=si, s_out=so, s_w=sw.tolist()))
            shp[i], sc[i], zp[i] = (ho, wo, Co), so, zo
        elif t == "upsample":
            f = l["factor"]
            q.update(KH=1, KW=1, stride=1, pad=[0, 0, 0, 0], factor=f, C_out=cin, in_zp=zp[a], out_zp=zp[a], relu=0,
                     scales=dict(s_in=sc[a], s_out=sc[a]))
            shp[i], sc[i], zp[i] = (h * f, w_ * f, cin), sc[a], zp[a]
        elif t == "add":
            b = l["inputs"][1]
            assert shp[a] == shp[b], (l["name"], shp[a], shp[b])
            so, zo = act_qparams(ranges[i])
            Ma, Mb, S = choose_add_params(sc[a], sc[b], so)
            q.update(KH=1, KW=1, stride=1, pad=[0, 0, 0, 0], factor=1, C_out=cin, in_zp=zp[a], out_zp=zo, relu=1,
                     Ma=Ma, Mb=Mb, S_add=S, za=zp[a], zb=zp[b], scales=dict(s_a=sc[a], s_b=sc[b], s_out=so))
            shp[i], sc[i], zp[i] = shp[a], so, zo
        q["out_H"], q["out_W"] = shp[i][0], shp[i][1]
        Q.append(q)
    return Q, sc[len(L) - 1], zp[len(L) - 1]


# ----------------------------------------------------------------------------------------------------------------------------
def calib_frames(n, W, H):
    tr = sorted(glob.glob(os.path.join(C.ROOT, "ml", "data", "train_raw", "**", "*.h5"), recursive=True))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(tr), size=min(n, len(tr)), replace=False) if tr else []
    out = []
    for k in pick:
        try:
            r, d = C.load_raw(tr[k])
            out.append(C.prep(r, d, W, H)[0])
        except Exception:
            pass
    if not out:                                                 # no train frames yet: fall back to the first val frames
        out = [C.prep(*C.load_raw(f), W, H)[0] for f in C.nyu_files()[:n]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="fine-tuned state dict (default: the vendor checkpoint)")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--calib", type=int, default=64)
    ap.add_argument("--pct", type=float, default=99.99)
    a = ap.parse_args()
    W, H = a.width, a.height
    assert W % 32 == 0 and H % 32 == 0
    model = C.load_model()
    ck_name = os.path.basename(C.CKPT)
    if a.ckpt:
        model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["model_state_dict"])
        ck_name = os.path.basename(a.ckpt)
    L = build_float_layers(model)

    frames = calib_frames(a.calib, W, H)
    xb = torch.from_numpy(np.stack(frames).astype(np.float64) / 255.0).permute(0, 3, 1, 2)
    with torch.no_grad():
        ref = model.double()(xb)
        outs = float_exec(L, xb)
    err = float((outs[-1] - ref).abs().max())
    print(f"float layer list vs PyTorch model: max |diff| = {err:.2e} m over {len(frames)} frames")
    assert err < 1e-3
    ranges = []
    for o in outs:
        v = o.reshape(-1).numpy()
        ranges.append(float(max(np.percentile(v, a.pct), 1e-3)))

    Q, s_last, z_last = quantise(L, ranges, W, H)
    hi = s_last * 255
    gate = {"canny_low": 40, "canny_high": 100}
    mq = {
        "model": "FastDepth MobileNet-NNConv5(depthwise) + additive skips (unpruned), weights " + ck_name,
        "input_size": [H, W],
        "preproc": {"width": W, "height": H, "channels": 3, "color": "rgb", "resize": {"method": "bilinear", "done_by": "host"},
                    "input_transform": "u8_minus_128",
                    "normalise": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0], "pixel_scale": "1/255", "folded_into_layer0": True}},
        "output": {"type": "depth", "unit": "m", "scale": s_last, "zero_point": z_last, "byte_offset": 128, "min": 0.0, "max": hi},
        "gate": gate,
        "calibration": {"frames": len(frames), "percentile": a.pct, "ranges": ranges},
        "layers": Q,
    }
    os.makedirs(EXPORT, exist_ok=True)
    with open(os.path.join(EXPORT, "model_q.json"), "w") as f:
        json.dump(mq, f, separators=(",", ":"))
    nW = sum(len(q.get("W", [])) for q in Q)
    print(f"model_q.json: {len(Q)} layers, {nW} INT8 weights, input {W}x{H}, output depth scale {s_last:.5f} m/step (max {hi:.2f} m)")


if __name__ == "__main__":
    main()
