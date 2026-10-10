"""MiDaS v2.1 small -> integer layer list ml/export/model_q.json (same format as the FastDepth export, plus two layer types:
"relu" (standalone ReLU, y = max(x, zero point)) and "bilinear" (x2 bilinear upsample, see reference/qmath_ref.py)).

  python ml/convert_midas.py [--width 128] [--height 96] [--calib 48]

1. traces the PyTorch model (torch.fx), folds every BatchNorm into its conv, fuses ReLU/ReLU6 that directly follow a conv, keeps the
   other ReLUs as standalone layers, turns adds and bilinear upsamples into layers; TF-"same" padding is computed for the input size;
2. folds the input normalisation (mean/std) into the first conv, so the board input stays x = p - 128;
3. checks the float layer list against the PyTorch model;
4. calibrates one range per tensor (signed tensors get an asymmetric zero point), quantises like convert.py.
The output is relative inverse depth ("disparity": large = near)."""
import argparse, glob, json, os, sys
import numpy as np
import torch, torch.fx as fx
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(os.path.dirname(HERE), "reference"), os.path.join(os.path.dirname(HERE), "host")]
import fd_common as C
import midas_common as MC
from qmath_ref import choose_M_S, choose_add_params, fold_input_zp, fold_input_zp_dw
from preprocess import resize_rgb

EXPORT = os.path.join(HERE, "export_midas")          # keeps the FastDepth export in ml/export untouched
LEAF = ("Conv2d", "Conv2dSame", "Conv2dSameExport", "BatchNorm2d", "ReLU", "ReLU6", "Interpolate", "Identity", "Dropout")


class Tracer(fx.Tracer):
    def is_leaf_module(self, mod, name):
        return type(mod).__name__ in LEAF or super().is_leaf_module(mod, name)


def same_pad(i, k, s, d=1):
    p = max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)
    return p // 2, p - p // 2


# ---------------------------------------------------------------------------------------------------------------- graph -> layers
def build_float_layers(model, H, W):
    g = Tracer().trace(model)
    gm = fx.GraphModule(model, g)
    ShapeProp(gm).propagate(torch.zeros(1, 3, H, W))
    mods = dict(model.named_modules())
    L, of = [], {}                                   # of[node] = index of the layer that produces it (-2 = network input)
    def src(a):
        return of[a]
    for n in g.nodes:
        if n.op == "placeholder":
            of[n] = -2
        elif n.op == "call_module":
            m = mods[n.target]; t = type(m).__name__
            if t in ("Conv2d", "Conv2dSame", "Conv2dSameExport"):
                ih, iw = n.args[0].meta["tensor_meta"].shape[2:]
                k, s = m.kernel_size[0], m.stride[0]
                if t == "Conv2d":
                    p = m.padding[0]; pad = [p, p, p, p]
                else:
                    pt, pb = same_pad(ih, k, s); pl, pr = same_pad(iw, k, s); pad = [pt, pl, pb, pr]
                w = m.weight.detach().double().numpy()
                b = m.bias.detach().double().numpy() if m.bias is not None else np.zeros(m.out_channels)
                kind = "dwconv" if m.groups > 1 else ("pwconv" if k == 1 else "conv")
                assert m.groups in (1, m.in_channels), "grouped conv not supported"
                L.append(dict(name=n.target, type=kind, KH=k, KW=k, stride=s, pad=pad, C_in=m.in_channels, C_out=m.out_channels,
                              w=w, b=b, act=None, inputs=[src(n.args[0])], node_users=len(n.users)))
                of[n] = len(L) - 1
            elif t == "BatchNorm2d":
                j = src(n.args[0]); l = L[j]
                assert l["type"] in ("conv", "pwconv", "dwconv") and l["act"] is None and l["node_users"] == 1, n.target
                gam = (m.weight / torch.sqrt(m.running_var + m.eps)).detach().double().numpy()
                l["w"] = l["w"] * gam[:, None, None, None]
                l["b"] = (l["b"] - m.running_mean.detach().double().numpy()) * gam + m.bias.detach().double().numpy()
                l["node_users"] = len(n.users)
                of[n] = j
            elif t in ("ReLU", "ReLU6"):
                j = src(n.args[0]); l = L[j] if j >= 0 else None
                if l is not None and l["type"] in ("conv", "pwconv", "dwconv") and l["act"] is None and l["node_users"] == 1:
                    l["act"] = "relu6" if t == "ReLU6" else "relu"
                    l["node_users"] = len(n.users)
                    of[n] = j
                else:
                    assert t == "ReLU", "standalone ReLU6 not supported"
                    L.append(dict(name=n.target + f"#{len(L)}", type="relu", inputs=[j])); of[n] = len(L) - 1
            elif t in ("Identity", "Dropout"):
                of[n] = src(n.args[0])
            elif t == "Interpolate":
                assert m.mode == "bilinear" and m.scale_factor == 2
                L.append(dict(name=n.target, type="bilinear", factor=2, align=int(bool(m.align_corners)), inputs=[src(n.args[0])])); of[n] = len(L) - 1
            else:
                raise ValueError(f"module {t} not supported")
        elif n.op == "call_function":
            nm = getattr(n.target, "__name__", str(n.target))
            if nm == "add":
                L.append(dict(name=f"add{len(L)}", type="add", inputs=[src(n.args[0]), src(n.args[1])])); of[n] = len(L) - 1
            elif nm == "interpolate":
                assert n.kwargs.get("mode") == "bilinear" and n.kwargs.get("scale_factor") == 2
                L.append(dict(name=f"up{len(L)}", type="bilinear", factor=2, align=int(bool(n.kwargs.get("align_corners"))), inputs=[src(n.args[0])]))
                of[n] = len(L) - 1
            elif nm == "squeeze":
                of[n] = src(n.args[0])
            else:
                raise ValueError(f"function {nm} not supported")
        elif n.op == "output":
            out = src(n.args[0])
            assert out == len(L) - 1, "the output must be the last layer"
        else:
            raise ValueError(n.op)
    for l in L:
        l.pop("node_users", None)
    return L


def fold_normalisation(L):
    """first conv sees r = p/255 instead of (r - mean)/std: w' = w/std[c], b' = b - sum w*mean/std (border pixels differ slightly:
       the original pads with 0 in normalised space, the board pads with r = 0)"""
    l = L[0]
    assert l["inputs"] == [-2] and l["type"] == "conv"
    mean, std = np.array(MC.MEAN), np.array(MC.STD)
    w = l["w"] / std[None, :, None, None]
    l["b"] = l["b"] - (l["w"] * (mean / std)[None, :, None, None]).sum(axis=(1, 2, 3))
    l["w"] = w
    l["pad_real"] = round(float(mean.mean()) * 255) / 255.0     # pad with grey (~ the mean colour = 0 after normalisation), not black


def float_exec(L, x):
    outs = []
    for l in L:
        s = [x if j == -2 else outs[j] for j in l["inputs"]]
        t = l["type"]
        if t in ("conv", "pwconv", "dwconv"):
            p = l["pad"]
            xi = F.pad(s[0], (p[1], p[3], p[0], p[2]), value=l.get("pad_real", 0.0))
            y = F.conv2d(xi, torch.from_numpy(l["w"]).to(xi.dtype), torch.from_numpy(l["b"]).to(xi.dtype), stride=l["stride"],
                         groups=l["C_in"] if t == "dwconv" else 1)
            if l["act"] == "relu": y = torch.relu(y)
            elif l["act"] == "relu6": y = torch.clamp(y, 0, 6)
        elif t == "relu":
            y = torch.relu(s[0])
        elif t == "add":
            y = s[0] + s[1]
        elif t == "bilinear":
            y = F.interpolate(s[0], scale_factor=2, mode="bilinear", align_corners=bool(l["align"]))
        outs.append(y)
    return outs


# ---------------------------------------------------------------------------------------------------------------- quantisation
def qparams(lo, hi):
    lo, hi = min(lo, 0.0), max(hi, 1e-6)
    s = (hi - lo) / 255.0
    zp = int(round(-128 - lo / s))
    return s, max(-128, min(127, zp))


def quantise(L, ranges, W, H):
    sc, zp, shp = {-2: 1 / 255.0}, {-2: -128}, {-2: (H, W, 3)}
    Q = []
    for i, l in enumerate(L):
        a = l["inputs"][0]; h, w_, cin = shp[a]
        q = dict(name=l["name"], type=l["type"], inputs=l["inputs"], in_H=h, in_W=w_, C_in=cin)
        t = l["type"]
        if t in ("conv", "pwconv", "dwconv"):
            lo, hi = ranges[i]
            if l["act"] in ("relu", "relu6"):
                lo = 0.0
            if l["act"] == "relu6":
                hi = min(hi, 6.0)
            so, zo = qparams(lo, hi)
            si, zi = sc[a], zp[a]
            Co = l["C_out"]
            if t == "dwconv":
                wk = l["w"][:, 0].transpose(1, 2, 0); amax = np.abs(wk).reshape(-1, Co).max(axis=0)
            else:
                wk = l["w"].transpose(2, 3, 1, 0).reshape(-1, Co); amax = np.abs(wk).max(axis=0)
            sw = np.where(amax > 0, amax / 127.0, so / si)
            sw = np.maximum(sw, np.abs(l["b"]) / (si * 2**29))
            sw = np.maximum(sw, so / (si * 2**29))
            Wq = np.clip(np.round(wk / sw), -127, 127).astype(np.int64)
            bq = np.round(l["b"] / (si * sw)).astype(np.int64)
            bias_f = fold_input_zp_dw(bq, Wq, zi) if t == "dwconv" else fold_input_zp(bq, Wq, zi)
            MS = [choose_M_S(si * sw[c] / so) for c in range(Co)]
            KH, st, p = l["KH"], l["stride"], l["pad"]
            ho, wo = (h + p[0] + p[2] - KH) // st + 1, (w_ + p[1] + p[3] - KH) // st + 1
            pv = int(round(l["pad_real"] / si + zi)) if "pad_real" in l else zi
            q.update(KH=KH, KW=KH, stride=st, pad=p, pad_value=pv, factor=1, C_out=Co, in_zp=zi, out_zp=zo, relu=int(l["act"] is not None),
                     M=[m for m, _ in MS], S=[s for _, s in MS], W=Wq.reshape(-1).tolist(), bias=bias_f.tolist(),
                     scales=dict(s_in=si, s_out=so, s_w=sw.tolist()))
            shp[i], sc[i], zp[i] = (ho, wo, Co), so, zo
        elif t == "relu":
            q.update(KH=1, KW=1, stride=1, pad=[0, 0, 0, 0], factor=1, C_out=cin, in_zp=zp[a], out_zp=zp[a], relu=1, scales=dict(s_in=sc[a], s_out=sc[a]))
            shp[i], sc[i], zp[i] = (h, w_, cin), sc[a], zp[a]
        elif t == "bilinear":
            q.update(KH=1, KW=1, stride=1, pad=[0, 0, 0, 0], factor=2, align=l["align"], C_out=cin, in_zp=zp[a], out_zp=zp[a], relu=0,
                     scales=dict(s_in=sc[a], s_out=sc[a]))
            shp[i], sc[i], zp[i] = (h * 2, w_ * 2, cin), sc[a], zp[a]
        elif t == "add":
            b = l["inputs"][1]
            assert shp[a] == shp[b], (l["name"], shp[a], shp[b])
            so, zo = qparams(*ranges[i])
            Ma, Mb, S = choose_add_params(sc[a], sc[b], so)
            q.update(KH=1, KW=1, stride=1, pad=[0, 0, 0, 0], factor=1, C_out=cin, in_zp=zp[a], out_zp=zo, relu=0,
                     Ma=Ma, Mb=Mb, S_add=S, za=zp[a], zb=zp[b], scales=dict(s_a=sc[a], s_b=sc[b], s_out=so))
            shp[i], sc[i], zp[i] = shp[a], so, zo
        q["out_H"], q["out_W"] = shp[i][0], shp[i][1]
        Q.append(q)
    return Q, sc[len(L) - 1], zp[len(L) - 1]


def calib_frames(n, W, H):
    fs = C.nyu_files()[:n // 2]                              # first NYU test images (the accuracy check uses the others)
    out = [C.prep(*C.load_raw(f), W, H)[0] for f in fs]
    for p in sorted(glob.glob(os.path.join(HERE, "data", "photos", "*.jpg"))):
        from PIL import Image
        out.append(resize_rgb(Image.open(p), W, H))
    for p in sorted(glob.glob(os.path.join(HERE, "data", "unlabeled", "*.jpg")))[: n // 2]:
        from PIL import Image
        out.append(resize_rgb(Image.open(p), W, H))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--calib", type=int, default=48)
    ap.add_argument("--pct", type=float, default=99.99)
    a = ap.parse_args()
    W, H = a.width, a.height
    model = MC.load_midas_small()
    L = build_float_layers(model, H, W)
    xin = torch.from_numpy(np.stack(calib_frames(a.calib, W, H)).astype(np.float64) / 255.0).permute(0, 3, 1, 2)
    with torch.no_grad():
        ref = model.double()((xin - torch.tensor(MC.MEAN, dtype=torch.float64).view(1, 3, 1, 1)) / torch.tensor(MC.STD, dtype=torch.float64).view(1, 3, 1, 1))
        exact = float_exec(L, (xin - torch.tensor(MC.MEAN, dtype=torch.float64).view(1, 3, 1, 1)) / torch.tensor(MC.STD, dtype=torch.float64).view(1, 3, 1, 1))[-1][:, 0]
        print(f"layer list vs PyTorch (same input): max |diff| = {float((exact - ref).abs().max()):.2e} (relative to output max {float(ref.abs().max()):.1f})")
        fold_normalisation(L)
        outs = float_exec(L, xin)
        d = (outs[-1][:, 0] - ref).abs()
        print(f"after folding the normalisation into conv 0: mean |diff| = {float(d.mean()) / float(ref.abs().mean()) * 100:.2f} % of the mean output")
    ranges = []
    for o in outs:
        v = o.reshape(-1).numpy()
        ranges.append((float(np.percentile(v, 100 - a.pct)), float(np.percentile(v, a.pct))))
    Q, s_last, z_last = quantise(L, ranges, W, H)
    mq = {
        "model": "MiDaS v2.1 small (EfficientNet-Lite3 encoder), Intel ISL, MIT licence - relative inverse depth",
        "short_name": "MiDaS v2.1 small (EfficientNet-Lite3)",
        "n_weights": sum(len(q.get("W", [])) for q in Q),
        "input_size": [H, W],
        "preproc": {"width": W, "height": H, "channels": 3, "color": "rgb", "resize": {"method": "bilinear", "done_by": "host"},
                    "input_transform": "u8_minus_128", "normalise": {"mean": MC.MEAN, "std": MC.STD, "folded_into_layer0": True}},
        "output": {"type": "disparity", "unit": "relative", "scale": s_last, "zero_point": z_last, "byte_offset": 128, "min": 0.0, "max": s_last * 255},
        "gate": {"canny_low": 40, "canny_high": 100},
        "calibration": {"frames": int(xin.shape[0]), "percentile": a.pct},
        "layers": Q,
    }
    os.makedirs(EXPORT, exist_ok=True)
    with open(os.path.join(EXPORT, "model_q.json"), "w") as f:
        json.dump(mq, f, separators=(",", ":"))
    with open(os.path.join(EXPORT, "macs.csv"), "w", newline="") as f:          # same columns as ml/export/macs.csv
        f.write("index,name,type,runs_on,K,C_out,out_pixels,MACs\n")
        tot = 0
        for i, q in enumerate(Q):
            px = q["out_H"] * q["out_W"]
            K = q["KH"] * q["KW"] * q["C_in"] if q["type"] in ("conv", "pwconv") else (q["KH"] * q["KW"] if q["type"] == "dwconv" else 0)
            mac = px * q["C_out"] * K
            tot += mac
            f.write(f"{i},{q['name']},{q['type']},{'NPU' if q['type'] in ('conv', 'pwconv') else 'CPU'},{K},{q['C_out']},{px},{mac}\n")
        f.write(f",total,,,,,,{tot}\n")
    types = {}
    for q in Q:
        types[q["type"]] = types.get(q["type"], 0) + 1
    print(f"model_q.json: {len(Q)} layers {types}, {sum(len(q.get('W', [])) for q in Q)} INT8 weights, input {W}x{H}, output disparity")


if __name__ == "__main__":
    main()
