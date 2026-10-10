"""Systolic Squad live depth demo (laptop side).

  python host/app.py --port COM8                 # board: laptop camera -> UART -> FastDepth on the systolic array -> depth map back
  python host/app.py --emulate                   # no board: the integer golden model (ml/qmodel.py) gives the same bytes
  python host/app.py --port COM8 --image a.jpg   # a still image instead of the camera (repeats it)
  python host/app.py --port COM8 --model midas    # MiDaS v2.1 small (ml/export_midas) instead of FastDepth (ml/export)

Keys in the window:  q quit | n toggle NPU / CPU-only on the board | g toggle the Canny gate | b BENCH (CPU vs NPU, same frame)
                     s SELFTEST (NPU matmul vs CPU) | e show the next Canny edge map | + / - near threshold
At start the app streams ml/export/model_blob.bin to the board (WEIGHTS packets, ~45 s at 921600 baud) unless the board already
holds a blob with the right CRC (it answers WEIGHTS_ACK ok to an empty WEIGHTS_DONE)."""
import argparse, json, os, queue, struct, sys, threading, time, zlib
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, os.path.join(ROOT, "ml"), os.path.join(ROOT, "reference")]
import proto as P
from preprocess import resize_rgb

MODEL_Q = os.path.join(ROOT, "ml", "export", "model_q.json")
BLOB = os.path.join(ROOT, "ml", "export", "model_blob.bin")
SCALE = 4                                    # display magnification
EXPORTS = {"fastdepth": "export", "midas": "export_midas"}           # order = firmware model index (fw/include/model.h)
MODEL_INDEX = {k: i for i, k in enumerate(EXPORTS)}
MODEL_NAMES = {"fastdepth": "FastDepth", "midas": "MiDaS small"}


def use_model(name):
    """point MODEL_Q / BLOB at another export (the board firmware must be built for the same model)"""
    global MODEL_Q, BLOB
    MODEL_Q = os.path.join(ROOT, "ml", EXPORTS[name], "model_q.json")
    BLOB = os.path.join(ROOT, "ml", EXPORTS[name], "model_blob.bin")


def load_meta():
    with open(MODEL_Q) as f:
        head = f.read(4000)                  # preproc / output / gate are at the start; avoid parsing 12 MB for the board path
    i = head.find('"layers"')
    meta = json.loads(head[:i].rstrip(",") + "}") if i > 0 else json.load(open(MODEL_Q))
    return meta


class Board:
    def __init__(self, port, baud):
        import serial
        self.s = serial.Serial(port, baud, timeout=0.05)
        self.p = P.Parser()
        self.logs = []

    def send(self, data):
        self.s.write(data)

    def wait(self, types, timeout=5.0):
        """collect packets until one of `types` arrives; returns (dict type -> last payload, list of all packets)"""
        got, allp = {}, []
        t0 = time.time()
        while time.time() - t0 < timeout:
            for t, pl in self.p.feed(self.s.read(65536)):
                allp.append((t, pl))
                got[t] = pl
                if t == P.T_LOG:
                    msg = pl.decode(errors="replace")
                    self.logs.append(msg)
                    print("[board]", msg)
            if any(t in got for t in types):
                return got, allp
        return got, allp

    def ping(self):
        self.send(P.cmd(P.CMD_PING))
        got, _ = self.wait([P.T_PONG], 2.0)
        if P.T_PONG in got:
            return struct.unpack("<II", got[P.T_PONG])
        return None

    def select_model(self, idx):
        """CMD_SET_MODEL (firmware 3.0): the board keeps every model's weights in its own slot; answer = that slot's WEIGHTS_ACK"""
        self.send(P.cmd(P.CMD_SET_MODEL, idx))
        got, _ = self.wait([P.T_WEIGHTS_ACK], 10.0)
        return struct.unpack("<IIB", got[P.T_WEIGHTS_ACK]) if P.T_WEIGHTS_ACK in got else None

    def load_weights(self, blob, force=False, progress=None):
        crc = zlib.crc32(blob) & 0xFFFFFFFF
        if not force:
            self.send(P.cmd(P.CMD_WEIGHTS_DONE))
            got, _ = self.wait([P.T_WEIGHTS_ACK], 10.0)
            if P.T_WEIGHTS_ACK in got:
                n, c, ok = struct.unpack("<IIB", got[P.T_WEIGHTS_ACK])
                if ok and c == crc:
                    print(f"board already holds the model blob ({n} B, CRC {c:08X})")
                    return True
        for attempt in range(1, 4):                   # the board checks a CRC; on a mismatch the whole blob is sent again
            print(f"streaming model_blob.bin ({len(blob)} B), attempt {attempt} ...", flush=True)
            t0 = time.time()
            for off in range(0, len(blob), 4096):
                self.send(P.packet(P.T_WEIGHTS, struct.pack("<I", off) + blob[off:off + 4096]))
                self.s.flush()                        # wait until the packet has left the laptop, then leave a real gap on the wire,
                time.sleep(0.004)                     # so the board can copy it to DDR before the next one (its UART FIFO is 16 bytes)
                if (off // 4096) % 64 == 0:
                    print(f"  {off * 100 // len(blob)} %", flush=True)
                    if progress:
                        progress(off * 100 // len(blob))
            self.send(P.cmd(P.CMD_WEIGHTS_DONE))
            got, _ = self.wait([P.T_WEIGHTS_ACK], 30.0)
            if P.T_WEIGHTS_ACK not in got:
                print("no WEIGHTS_ACK"); continue
            n, c, ok = struct.unpack("<IIB", got[P.T_WEIGHTS_ACK])
            print(f"WEIGHTS_ACK: {n} B, CRC {c:08X} (host {crc:08X}), ok={ok}, {time.time() - t0:.1f} s")
            if ok and c == crc:
                return True
        return False

    def run(self, rgb):
        self.send(P.frame(rgb))
        got, allp = self.wait([P.T_RESULT], 60.0)
        res = P.decode_result(got[P.T_RESULT]) if P.T_RESULT in got else None
        dm = P.decode_map(got[P.T_DEPTH_MAP])[2] if P.T_DEPTH_MAP in got else None
        tel = P.decode_telemetry(got[P.T_TELEMETRY]) if P.T_TELEMETRY in got else None
        edge = P.decode_map(got[P.T_EDGE_MAP])[2] if P.T_EDGE_MAP in got else None
        if P.T_TELEMETRY not in got:
            g2, _ = self.wait([P.T_TELEMETRY], 0.3)
            if P.T_TELEMETRY in g2:
                tel = P.decode_telemetry(g2[P.T_TELEMETRY])
        return res, dm, tel, edge


class Emulator:
    """same bytes as the board, computed by the integer golden model on the laptop"""
    def __init__(self):
        from qmodel import QModel
        from depth_ref import zone_nearness, ObstacleAgent
        self.q = QModel(MODEL_Q)
        self.zn, self.agent = zone_nearness, ObstacleAgent()
        self.kind = self.q.mq["output"]["type"]
        self.fid = 0

    def run(self, rgb):
        t0 = time.time()
        db = self.q.depth_bytes(self.q.run(rgb))
        n = self.zn(db.astype(np.int64), self.kind)
        r = self.agent.update(n)
        self.fid += 1
        dt = int((time.time() - t0) * 1e6)
        res = {"frame_id": self.fid, "g_raw": r["raw"], "g_stable": r["stable"], "leds": r["leds"], "flags": 4,
               "logits": list(n) + [0, 0, dt, 0, 0], "t_total_us": dt, "t_cnn_us": dt, "npu_cycles": 0, "npu_active": 0}
        return res, db.tobytes(), None, None


def colour_depth(db, W, H, vmax=None, disparity=False):
    """depth bytes -> BGR image. Colours follow the scene (2nd..98th percentile: red = near, blue = far, never black), like the dashboard.
    disparity = True: the bytes are inverse depth (large = near)."""
    d = np.frombuffer(db, dtype=np.uint8).reshape(H, W).astype(np.float32)
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    f = np.clip((d - lo) / max(hi - lo, 1.0), 0, 1)
    if disparity:
        f = 1 - f
    return cv2.applyColorMap(((0.95 - 0.83 * f) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

def draw(cam_bgr, depth_bgr, res, tel, meta, mode_txt, extra):
    H, W = cam_bgr.shape[:2]
    big = lambda im: cv2.resize(im, (W * SCALE, H * SCALE), interpolation=cv2.INTER_NEAREST)
    left = big(cam_bgr)
    right = big(depth_bgr) if depth_bgr is not None else np.zeros_like(left)
    panel = np.zeros((H * SCALE, 360, 3), np.uint8)
    y = 30
    def put(t, c=(230, 230, 230), s=0.6):
        nonlocal y
        cv2.putText(panel, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, s, c, 1, cv2.LINE_AA); y += int(30 * s / 0.6)
    put("Systolic Squad: depth on a 16x16 systolic array", (255, 255, 255), 0.5)
    put(mode_txt, (120, 220, 255))
    if res:                                   # (obstacle guidance is computed on the board but not shown)
        gated = res["flags"] & 1
        put(f"frame {res['frame_id']}  {'GATED (scene static)' if gated else ('NPU' if res['flags'] & 2 else 'CPU only')}")
        put(f"depth net  {res['t_cnn_us'] / 1000:.0f} ms" + ("" if gated else ""))
        if res["flags"] & 2:
            put(f"  conv on NPU {res['logits'][4] / 1000:.0f} ms, CPU layers {res['logits'][5] / 1000:.0f} ms", s=0.5)
            put(f"  array: {res['npu_cycles']} clk, busy {res['npu_active']} clk", s=0.5)
        put(f"frame total {res['t_total_us'] / 1000:.0f} ms", s=0.5)
    if tel:
        put(f"motion {tel['motion_score']} / edges {tel['edge_score']} permille", s=0.5)
        put(f"runs {tel['cnn_runs']}  gated {tel['frames_gated']}", s=0.5)
    for t in extra[-4:]:
        put(t, (200, 200, 120), 0.45)
    put("q quit  n NPU/CPU  g gate  b bench  s selftest", (150, 150, 150), 0.45)
    return np.hstack([left, right, panel])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM8")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--image", default=None)
    ap.add_argument("--emulate", action="store_true")
    ap.add_argument("--model", choices=sorted(EXPORTS), default="fastdepth")
    ap.add_argument("--reload", action="store_true", help="always stream the weights")
    ap.add_argument("--frames", type=int, default=0, help="stop after n frames (0 = run until q)")
    ap.add_argument("--headless", action="store_true", help="no window, print one line per frame")
    ap.add_argument("--near-m", type=float, default=1.0, help="a zone is NEAR when its nearest 10 %% is closer than this (metres)")
    ap.add_argument("--warn-m", type=float, default=1.8, help="a zone is WARN when its nearest 10 %% is closer than this (metres)")
    a = ap.parse_args()
    use_model(a.model)
    meta = load_meta()
    W, H = meta["preproc"]["width"], meta["preproc"]["height"]
    disp = meta["output"]["type"] == "disparity"
    vmax = 255 * 0.6                          # colour range: 0 .. 60 % of the output range (indoor scenes)
    scale = meta["output"]["scale"]           # metres per depth byte step (byte = q + 128, real = scale * byte since zero_point = -128)
    to_near = lambda m: int(max(0, min(255, round(255 - (m / scale - meta["output"]["zero_point"] - 128)))))
    t_near, t_warn = to_near(a.near_m), to_near(a.warn_m)
    print(f"obstacle thresholds: warn < {a.warn_m} m (nearness {t_warn}), near < {a.near_m} m (nearness {t_near})")
    extra = []
    if a.emulate:
        dev = Emulator(); mode = "EMULATOR (golden model on the laptop)"
        dev.agent.t_warn, dev.agent.t_near = t_warn, t_near
    else:
        dev = Board(a.port, a.baud)
        pong = dev.ping()
        if not pong:
            print(f"no PONG on {a.port} at {a.baud} baud - is the depth app programmed and the board power-cycled?"); return 1
        print(f"board firmware {pong[0]:08X}, NPU id {pong[1]:08X}")
        dev.select_model(MODEL_INDEX[a.model])
        if not dev.load_weights(open(BLOB, "rb").read(), a.reload):
            print("weights not accepted"); return 1
        mode = "BOARD: PolarFire SoC + our 16x16 array"
        dev.send(P.cmd(P.CMD_SET_THRESH, 0, t_warn)); dev.send(P.cmd(P.CMD_SET_THRESH, 1, t_near))
    use_npu, gate = 1, 1
    cap, still = None, None
    if a.image:
        still = resize_rgb(cv2.cvtColor(cv2.imread(a.image), cv2.COLOR_BGR2RGB), W, H)
    else:
        cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Three loops so nothing waits on anything else: the camera thread always holds the NEWEST frame (no stale buffered frames),
    # the worker thread sends that frame to the board (or emulator) and stores the answer, the main loop draws at camera speed.
    stop = threading.Event()
    latest = {"rgb": None}
    st = {"res": None, "depth": None, "tel": None, "edge": None, "rtt": 0.0, "t_res": 0.0, "n": 0, "err": ""}
    cmds = queue.Queue()

    def camera():
        while not stop.is_set():
            if still is not None:
                latest["rgb"] = still; time.sleep(0.03); continue
            ok, bgr = cap.read()
            if ok:
                latest["rgb"] = resize_rgb(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), W, H)
            else:
                time.sleep(0.01)

    def command(pkt, want, fmt, label):
        dev.send(pkt)
        if want is None:
            return
        got, _ = dev.wait([want], 120.0)
        if want in got:
            v = struct.unpack(fmt, got[want])
            if want == P.T_BENCH:
                extra.append(f"BENCH cpu {v[0] / 1000:.0f} ms, npu {v[1] / 1000:.0f} ms, x{v[0] / max(v[1], 1):.2f}, outputs match={v[3]}")
            elif want == P.T_SELFTEST:
                extra.append(f"SELFTEST {'PASS' if v[0] else 'FAIL'} {v[1] - v[2]}/{v[1]}")
            print(extra[-1], flush=True)
        else:
            extra.append(f"{label}: no answer")

    def worker():
        while not stop.is_set():
            try:
                while not cmds.empty():
                    command(*cmds.get())
                rgb = latest["rgb"]
                if rgb is None:
                    time.sleep(0.01); continue
                t0 = time.time()
                res, db, tel, edge = dev.run(rgb)
                st["rtt"] = time.time() - t0
                if res is None:
                    st["err"] = "no RESULT from the board"; continue
                st["err"] = ""
                st["res"], st["tel"], st["t_res"] = res, tel or st["tel"], time.time()
                if db is not None:
                    st["depth"] = colour_depth(db, W, H, vmax, disp)
                if edge is not None:
                    st["edge"] = edge
                st["n"] += 1
                if a.headless or a.frames:
                    print(f"frame {res['frame_id']}: cnn {res['t_cnn_us'] / 1000:.0f} ms "
                          f"(npu {res['logits'][4] / 1000:.0f} / cpu {res['logits'][5] / 1000:.0f}) flags {res['flags']} round trip {st['rtt'] * 1000:.0f} ms", flush=True)
            except Exception as e:                                    # board reset / cable: reconnect, re-send weights if needed
                st["err"] = f"board link lost ({type(e).__name__}), reconnecting"
                print(st["err"], flush=True)
                if a.emulate:
                    time.sleep(1); continue
                try:
                    dev.s.close()
                except Exception:
                    pass
                while not stop.is_set():
                    time.sleep(2)
                    try:
                        dev.__init__(a.port, a.baud)
                        if dev.ping() and dev.select_model(MODEL_INDEX[a.model]) and dev.load_weights(open(BLOB, "rb").read()):
                            dev.send(P.cmd(P.CMD_SET_THRESH, 0, t_warn)); dev.send(P.cmd(P.CMD_SET_THRESH, 1, t_near))
                            dev.send(P.cmd(P.CMD_USE_NPU, use_npu)); dev.send(P.cmd(P.CMD_SET_GATE, gate))
                            st["err"] = ""; break
                    except Exception:
                        pass

    threads = [threading.Thread(target=camera, daemon=True), threading.Thread(target=worker, daemon=True)]
    for t in threads:
        t.start()
    try:
        while True:
            if a.headless or a.frames:
                time.sleep(0.05)
                if a.frames and st["n"] >= a.frames:
                    break
                continue
            rgb = latest["rgb"]
            if rgb is None:
                cv2.waitKey(10); continue
            res = st["res"]
            age = time.time() - st["t_res"] if res else 0
            label = mode
            if res and not a.emulate:
                label += " | conv on NPU" if res["flags"] & 2 else (" | CPU only" if res["flags"] & 4 else " | gated")
            label += f" | depth {1 / max(st['rtt'], 1e-3):.1f} fps, age {age:.1f} s"
            img = draw(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), st["depth"], res, st["tel"], meta, label, extra + ([st["err"]] if st["err"] else []))
            cv2.imshow("Systolic Squad depth", img)
            if st["edge"] is not None:
                cv2.imshow("Canny gate (board)", cv2.resize(np.frombuffer(st["edge"], np.uint8).reshape(H, W) * 2, (W * SCALE, H * SCALE),
                                                            interpolation=cv2.INTER_NEAREST))
                st["edge"] = None
            k = cv2.waitKey(15) & 0xFF
            if k == ord("q") or cv2.getWindowProperty("Systolic Squad depth", cv2.WND_PROP_VISIBLE) < 1:
                break
            if not a.emulate:
                if k == ord("n"):
                    use_npu ^= 1; cmds.put((P.cmd(P.CMD_USE_NPU, use_npu), None, "", "npu"))
                elif k == ord("g"):
                    gate ^= 1; cmds.put((P.cmd(P.CMD_SET_GATE, gate), None, "", "gate")); extra.append(f"gate {'on' if gate else 'off'}")
                elif k == ord("e"):
                    cmds.put((P.cmd(P.CMD_EDGE_NEXT), None, "", "edge"))
                elif k == ord("b"):
                    extra.append("BENCH running (CPU and NPU on the same frame) ...")
                    cmds.put((P.cmd(P.CMD_BENCH), P.T_BENCH, "<IIIB", "BENCH"))
                elif k == ord("s"):
                    cmds.put((P.cmd(P.CMD_SELFTEST), P.T_SELFTEST, "<BHH", "SELFTEST"))
    finally:
        stop.set()
        time.sleep(0.2)
        if cap:
            cap.release()
        cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    sys.exit(main())
