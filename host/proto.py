"""UART protocol of(host side): packet builder + streaming parser + payload decoders."""
import struct

T_FRAME, T_CMD, T_WEIGHTS = 0x01, 0x02, 0x03
T_RESULT, T_TELEMETRY, T_LOG, T_EDGE_MAP, T_BENCH, T_SELFTEST, T_DEPTH_MAP, T_WEIGHTS_ACK, T_PONG = 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x8F
CMD_PING, CMD_SET_MODE, CMD_SELFTEST, CMD_BENCH, CMD_SET_GATE, CMD_EDGE_NEXT, CMD_SEND_DEPTH, CMD_SET_THRESH, CMD_WEIGHTS_DONE, CMD_USE_NPU, CMD_SET_MODEL, CMD_PROFILE = range(1, 13)
GUIDANCE = {0: "CLEAR", 1: "GO LEFT", 2: "GO RIGHT", 3: "STOP"}
RESULT_FMT, TELEMETRY_FMT = "<IBBBBB8iIIII", "<BBHIIIIHHBBHI"
assert struct.calcsize(RESULT_FMT) == 57 and struct.calcsize(TELEMETRY_FMT) == 32


def packet(t, payload=b""):
    n = len(payload)
    assert n <= 65535
    chk = (t + (n & 0xFF) + (n >> 8) + sum(payload)) & 0xFF
    return bytes([0xA5, 0x5A, t, n & 0xFF, n >> 8]) + bytes(payload) + bytes([chk])


def cmd(c, *args):
    return packet(T_CMD, bytes([c] + list(args)))


def frame(rgb):
    H, W, C = rgb.shape
    return packet(T_FRAME, struct.pack("<HHB", W, H, C) + rgb.astype("uint8").tobytes())


class Parser:
    """feed(bytes) -> list of (type, payload) for every complete packet with a good checksum; resyncs on 0xA5 0x5A."""
    def __init__(self):
        self.buf = bytearray()
        self.bad = 0

    def feed(self, data):
        self.buf += data
        out = []
        while True:
            i = self.buf.find(b"\xA5\x5A")
            if i < 0:
                del self.buf[:-1]
                return out
            del self.buf[:i]
            if len(self.buf) < 5:
                return out
            t, n = self.buf[2], self.buf[3] | (self.buf[4] << 8)
            if len(self.buf) < 6 + n:
                return out
            p = bytes(self.buf[5:5 + n])
            chk = self.buf[5 + n]
            if (t + (n & 0xFF) + (n >> 8) + sum(p)) & 0xFF == chk:
                out.append((t, p))
                del self.buf[:6 + n]
            else:
                self.bad += 1
                del self.buf[:2]


def decode_result(p):
    f = struct.unpack(RESULT_FMT, p)
    return {"frame_id": f[0], "n_classes": f[1], "g_raw": f[2], "g_stable": f[3], "leds": f[4], "flags": f[5], "logits": list(f[6:14]),
            "t_total_us": f[14], "t_cnn_us": f[15], "npu_cycles": f[16], "npu_active": f[17]}


def decode_telemetry(p):
    k = ["mode", "reason", "fps_x10", "frames_seen", "frames_gated", "cnn_runs", "uptime_ms", "motion_score", "edge_score",
         "engines_awake", "gate_engine", "zero_skip_permille", "big_sleep_ms"]
    return dict(zip(k, struct.unpack(TELEMETRY_FMT, p)))


def decode_map(p):
    W, H = struct.unpack("<HH", p[:4])
    return W, H, p[4:]
