"""Demo web app: the laptop serves a dashboard (host/web/index.html, the team's UI style) to its own browser and to any phone nearby.
Camera frames go phone/laptop -> this server -> PolarFire (depth network on the systolic array) -> back, with live timings, graphs and
measured reports. The laptop page shows a QR code for the phones and can "watch" what the phones send.

  python host/phone_server.py --emulate          # no board: integer golden model on the laptop (same bytes as the board)
  python host/phone_server.py --port COM8        # board
  python host/phone_server.py --port COM8 --model midas   # start with MiDaS (the MODEL button on the page switches any time)

Laptop:  http://localhost:8080/   (plain http is allowed camera access on localhost; opened automatically)
Phones:  https://<laptop-ip>:8443/ (QR code on the laptop page). Phone browsers need https for the camera, so the server uses a
         self-signed certificate (made once with openssl in WSL): tap "Advanced -> proceed" once. Same Wi-Fi as the laptop, or the
         laptop's Mobile Hotspot (https://192.168.137.1:8443/). Allow python.exe in the Windows firewall prompt.
The board serves one frame at a time (requests queue); every number on the page comes from the board/emulator or host/demo_facts.json."""
import argparse, base64, csv, io, json, os, re, socket, ssl, struct, subprocess, sys, threading, time, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import app as A                       # Board, Emulator, load_meta, BLOB
import proto as P
from preprocess import resize_rgb

CERT_DIR = os.path.join(HERE, "certs")
PAGE = os.path.join(HERE, "web", "index.html")


def local_ips():
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); ips.add(s.getsockname()[0]); s.close()
    except Exception:
        pass
    ips.add("192.168.137.1")                  # Windows Mobile Hotspot address
    return sorted(i for i in ips if not i.startswith("127."))


def ensure_cert(ips):
    cert, key = os.path.join(CERT_DIR, "cert.pem"), os.path.join(CERT_DIR, "key.pem")
    if os.path.exists(cert) and os.path.exists(key):
        return cert, key
    os.makedirs(CERT_DIR, exist_ok=True)
    san = ",".join([f"IP:{i}" for i in ips] + ["DNS:localhost"])
    wsl_dir = "/mnt/" + CERT_DIR[0].lower() + CERT_DIR[2:].replace("\\", "/")
    cmd = (f"openssl req -x509 -newkey rsa:2048 -nodes -days 60 -subj /CN=systolic-squad -addext subjectAltName={san} "
           f"-keyout '{wsl_dir}/key.pem' -out '{wsl_dir}/cert.pem'")
    subprocess.run(["wsl", "-e", "bash", "-lc", cmd], check=True, capture_output=True)
    return cert, key


def qr_data_url(url):
    import qrcode
    img = np.asarray(qrcode.make(url, border=2).convert("L"))
    ok, png = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()


def static_info(meta, mode, url, npu_present, export_dir, model):
    facts = json.load(open(os.path.join(HERE, "demo_facts.json"), encoding="utf-8"))
    layers = []
    with open(os.path.join(export_dir, "macs.csv")) as f:
        for r in csv.DictReader(f):
            if r["name"] != "total":
                layers.append({"name": r["name"], "type": r["type"], "on": r["runs_on"], "macs": int(r["MACs"])})
    tot = sum(l["macs"] for l in layers); npu = sum(l["macs"] for l in layers if l["on"] == "NPU")
    acc = []
    rep = os.path.join(export_dir, "accuracy_report.md")
    if os.path.exists(rep):
        for line in open(rep, encoding="utf-8"):
            c = [x.strip().strip("*") for x in line.strip().strip("|").split("|")]
            if len(c) == 5 and re.match(r"^[0-9.]+$", c[2]):
                acc.append([c[0].replace("community FastDepth checkpoint", "original checkpoint").replace("shipped model", "ours"), c[1], c[2] + " m", c[4]])
    W, H = meta["preproc"]["width"], meta["preproc"]["height"]
    nW = meta.get("n_weights", 3933088)
    o = meta["output"]
    disp = o["type"] == "disparity"
    return {
        "mode": mode, "npu_present": npu_present, "url": url, "qr": qr_data_url(url), "facts": facts,
        "display_max_m": 255 * 0.6 * o["scale"], "output_max_m": o["max"],
        "system": facts["system"] + [["Mode", "board" if mode == "board" else "emulator on the laptop"]],
        "kind": o["type"], "model_id": model, "models": [{"id": k, "name": A.MODEL_NAMES[k]} for k in A.EXPORTS], "model": {"name": meta.get("short_name", "FastDepth (MobileNet + NNConv5, skip-add)"), "mac_summary": f"{tot / 1e6:.1f} M MACs per frame, {100 * npu / tot:.1f} % on the array ({len(layers)} layers)",
                  "facts": [["Input", f"{W}x{H} RGB"], ["Weights", f"{nW / 1e6:.2f} M INT8"], ["MACs / frame", f"{tot / 1e6:.0f} M"],
                            ["On the array", f"{100 * npu / tot:.0f} %"], ["Output", "relative inverse depth" if disp else f"depth 0-{o['max']:.1f} m"]]},
        "layers": layers,
        "accuracy": acc, "accuracy_note": "Float vs INT8 on the NYU Depth v2 test split (109 images). INT8 = the integer golden model = the board's bytes.",
        "hardware": facts["hardware"], "hardware_note": facts["hardware_note"],
        "verification": facts["verification"] + facts["board_measured"], "not_measured": facts["not_measured"], "flow_note": facts["flow_note"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM8")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--emulate", action="store_true")
    ap.add_argument("--https-port", type=int, default=8443)
    ap.add_argument("--http-port", type=int, default=8080)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--model", choices=sorted(A.EXPORTS), default="fastdepth")
    a = ap.parse_args()
    npu_present = False
    if a.emulate:
        dev, where, short = None, "laptop emulator (golden model)", "EMULATOR"
    else:
        dev = A.Board(a.port, a.baud)
        pong = dev.ping()
        if not pong:
            print(f"no PONG on {a.port}"); return 1
        npu_present = pong[1] == 0x53514431
        where, short = "PolarFire SoC", "BOARD"
    ips = local_ips()
    hot = [i for i in ips if i == "192.168.137.1"]
    wifi = [i for i in ips if i != "192.168.137.1" and not i.startswith("172.")]
    url = f"https://{(wifi or hot or ips)[0]}:{a.https_port}/"
    lock = threading.Lock()
    clients = {}
    latest = {}
    S = {"model": None, "meta": None, "info": None, "loading": None, "emu": None}     # the active model

    def activate(name):
        """switch the board (or the emulator) to model `name`; stream its weights only if the board does not hold them yet"""
        nonlocal dev
        A.use_model(name)
        meta = A.load_meta()
        nice = A.MODEL_NAMES[name]
        S["loading"] = f"switching to {nice} ..."
        try:
            if a.emulate:
                S["emu"] = A.Emulator()
            else:
                with lock:
                    dev.select_model(A.MODEL_INDEX[name])
                    def prog(pct):
                        S["loading"] = f"loading {nice} weights into the board: {pct} % (once per power-up)"
                    if not dev.load_weights(open(A.BLOB, "rb").read(), progress=prog):
                        S["loading"] = None
                        return f"{nice}: weights not accepted"
            S.update(model=name, meta=meta,
                     info=static_info(meta, "emulator" if a.emulate else "board", url, npu_present, os.path.dirname(A.MODEL_Q), name))
            return None
        finally:
            S["loading"] = None

    err = activate(a.model)
    if err:
        print(err); return 1

    def run_frame(jpeg, client):
        if S["loading"]:
            return {"error": S["loading"], "loading": True}
        meta = S["meta"]
        W, H = meta["preproc"]["width"], meta["preproc"]["height"]
        scale, zp = meta["output"]["scale"], meta["output"]["zero_point"]
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "bad image"}
        rgb = resize_rgb(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), W, H)
        t0 = time.time()
        with lock:
            res, db, tel, edge = (S["emu"] if a.emulate else dev).run(rgb)
        out = {"where": where, "where_short": short, "w": W, "h": H, "scale": scale, "zp": zp, "kind": meta["output"]["type"], "model": S["model"]}
        if res:
            out.update(frame=res["frame_id"], cnn_ms=res["t_cnn_us"] / 1000, gated=bool(res["flags"] & 1))
            if res["flags"] & 2:
                out.update(npu_ms=res["logits"][4] / 1000, cpu_ms=res["logits"][5] / 1000, npu_cycles=res["npu_cycles"], npu_active=res["npu_active"])
        if db is not None:
            b = np.frombuffer(db, np.uint8)
            m = scale * (b.astype(np.float64) - 128 - zp)
            out.update(depth=base64.b64encode(bytes(db)).decode(), dmin=float(np.percentile(m, 2)), dmed=float(np.median(m)), dmax=float(np.percentile(m, 98)))
        elif res and res["flags"] & 1:
            out["error"] = "scene static: depth skipped by the Canny gate"
        c = clients.setdefault(client, {"frames": 0, "t": 0}); c["frames"] += 1; c["t"] = time.time()
        latest.clear(); latest.update(out); latest.update(rgb="data:image/jpeg;base64," + base64.b64encode(jpeg).decode(), client=client, t=time.time(),
                                                        rtt=(time.time() - t0) * 1000)
        return out

    def command(body):
        c = body.get("cmd")
        if S["loading"]:
            return {"error": S["loading"]}
        if c == "model":
            name = body.get("name")
            if name not in A.EXPORTS:
                return {"error": "unknown model"}
            if name == S["model"]:
                return {"text": A.MODEL_NAMES[name] + " is already running"}
            S["loading"] = f"switching to {A.MODEL_NAMES[name]} ..."
            threading.Thread(target=activate, args=(name,), daemon=True).start()
            return {"text": S["loading"], "switching": True}
        if a.emulate:
            return {"error": "board only (emulator mode)"}
        with lock:
            if c == "npu":
                dev.send(P.cmd(P.CMD_USE_NPU, 1 if body.get("on") else 0)); return {"text": "conv layers on the " + ("NPU" if body.get("on") else "CPU")}
            if c == "gate":
                dev.send(P.cmd(P.CMD_SET_GATE, 1 if body.get("on") else 0)); return {"text": "Canny gate " + ("on" if body.get("on") else "off")}
            if c == "selftest":
                dev.send(P.cmd(P.CMD_SELFTEST)); got, _ = dev.wait([P.T_SELFTEST], 20.0)
                if P.T_SELFTEST not in got:
                    return {"error": "no SELFTEST answer"}
                ok, n, nf = struct.unpack("<BHH", got[P.T_SELFTEST])
                return {"text": f"SELFTEST {'PASS' if ok else 'FAIL'}: {n - nf}/{n} NPU results equal the CPU"}
            if c == "bench":
                dev.send(P.cmd(P.CMD_BENCH)); got, _ = dev.wait([P.T_BENCH], 180.0)
                if P.T_BENCH not in got:
                    return {"error": "no BENCH answer (send at least one frame first)"}
                cpu, npu, macs, match = struct.unpack("<IIIB", got[P.T_BENCH])
                return {"cpu_ms": cpu / 1000, "npu_ms": npu / 1000, "macs": macs, "match": bool(match),
                        "text": f"BENCH: CPU {cpu / 1000:.0f} ms, NPU {npu / 1000:.0f} ms, outputs match = {bool(match)}"}
        return {"error": "unknown command"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *x):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj):
            self._send(200, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            p = self.path.split("?")[0]
            if p in ("/", "/index.html"):
                self._send(200, open(PAGE, "rb").read(), "text/html; charset=utf-8")
            elif p == "/info":
                self._json(dict(S["info"], loading=S["loading"]))
            elif p == "/latest":
                d = dict(latest)
                if d:
                    d["age"] = time.time() - d["t"]
                self._json(d)
            elif p == "/clients":
                now = time.time()
                self._json({"clients": [{"ip": k, "frames": v["frames"]} for k, v in clients.items() if now - v["t"] < 15]})
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            data = self.rfile.read(n)
            if self.path == "/frame":
                try:
                    out = run_frame(data, self.client_address[0])
                except Exception as e:
                    out = {"error": f"board error: {type(e).__name__}: {e}"}
                print(f"{self.client_address[0]}: frame -> {out.get('cnn_ms', '?')} ms", flush=True)
                self._json(out)
            elif self.path == "/cmd":
                try:
                    self._json(command(json.loads(data or b"{}")))
                except Exception as e:
                    self._json({"error": f"{type(e).__name__}: {e}"})
            else:
                self._send(404, b"not found", "text/plain")

    cert, key = ensure_cert(ips)
    https = ThreadingHTTPServer(("0.0.0.0", a.https_port), Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    https.socket = ctx.wrap_socket(https.socket, server_side=True)
    http = ThreadingHTTPServer(("127.0.0.1", a.http_port), Handler)          # laptop only: http://localhost is a secure context
    threading.Thread(target=https.serve_forever, daemon=True).start()
    threading.Thread(target=http.serve_forever, daemon=True).start()
    print(f"demo running ({where}).", flush=True)
    print(f"  laptop: http://localhost:{a.http_port}/", flush=True)
    for ip in ips:
        print(f"  phones: https://{ip}:{a.https_port}/", flush=True)
    if not a.no_browser:
        webbrowser.open(f"http://localhost:{a.http_port}/")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
