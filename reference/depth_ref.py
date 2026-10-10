"""Integer reference for the depth application. Firmware (C), the laptop emulator
and the exporter must reproduce these functions exactly. Run `python reference/depth_ref.py` for the self-test.

1. rgb_to_gray    : gray = (77 R + 150 G + 29 B + 128) >> 8    (feeds the Canny gate, reference/canny_ref.py; same size as the frame)
2. input_i8       : the network input is x = p - 128 (uint8 pixel -> int8); the exporter folds normalisation + this offset into layer 0
3. depth byte     : the DEPTH_MAP packet carries byte = q + 128, q = the network's final INT8 output (model_q.json "output" gives
                    the real value: real = scale * (q - zero_point); kind = "depth" (metres, small = near) or "disparity" (large = near))
4. obstacle zones : nearness n = 255 - byte (kind "depth") or byte (kind "disparity"); the map is split into LEFT / CENTRE / RIGHT thirds
                    of the width, using only the rows from H/4 downwards (the top quarter is ceiling/sky); zone nearness = the value that
                    the nearest 10 % of the zone's pixels reach (robust "nearest obstacle", from a 256-bin histogram)
5. ObstacleAgent  : per-zone warn/near levels with hysteresis, guidance CLEAR / GO_LEFT / GO_RIGHT / STOP, a candidate must repeat
                    CONFIRM frames before it becomes stable, except STOP which is immediate (safety)
"""
import numpy as np

CLEAR, GO_LEFT, GO_RIGHT, STOP = 0, 1, 2, 3
GUIDANCE_NAMES = {CLEAR: "CLEAR", GO_LEFT: "GO LEFT", GO_RIGHT: "GO RIGHT", STOP: "STOP"}

T_WARN, T_NEAR = 96, 160     # nearness thresholds (0..255), tunable at the venue from the dashboard; defaults are placeholders
EXIT_MARGIN = 8              # a level is left only when nearness falls EXIT_MARGIN below its entry threshold
PERCENTILE = 10              # the nearest 10 % of a zone's pixels define its nearness
CONFIRM = 3                  # frames a new (non-STOP) guidance must repeat before it becomes stable


def rgb_to_gray(rgb):
    p = np.asarray(rgb, dtype=np.int64)
    return (77 * p[..., 0] + 150 * p[..., 1] + 29 * p[..., 2] + 128) >> 8


def input_i8(rgb):
    return np.asarray(rgb, dtype=np.int64) - 128


def depth_byte(q):
    return np.asarray(q, dtype=np.int64) + 128


def nearness(depth_bytes, kind):
    b = np.asarray(depth_bytes, dtype=np.int64)
    assert kind in ("depth", "disparity")
    return 255 - b if kind == "depth" else b


def zone_columns(W):
    return [(0, W // 3), (W // 3, 2 * (W // 3)), (2 * (W // 3), W)]


def zone_nearness(depth_bytes, kind, pct=PERCENTILE):
    """-> (nL, nC, nR), each 0..255: the nearness reached by the nearest `pct` percent of the zone's pixels (rows from H/4 down)."""
    n = nearness(depth_bytes, kind)
    H, W = n.shape
    out = []
    for (c0, c1) in zone_columns(W):
        z = n[H // 4:, c0:c1].reshape(-1)
        need = -(-z.size * pct // 100)                       # ceil(size * pct / 100)
        hist = np.bincount(z, minlength=256)
        acc = 0
        t = 0
        for v in range(255, -1, -1):                         # walk down from the nearest value until `need` pixels are covered
            acc += int(hist[v])
            if acc >= need:
                t = v
                break
        out.append(t)
    return tuple(out)


class ObstacleAgent:
    def __init__(self, t_warn=T_WARN, t_near=T_NEAR):
        self.t_warn, self.t_near = t_warn, t_near
        self.warn = [False] * 3
        self.near = [False] * 3
        self.stable = CLEAR
        self.cand = CLEAR
        self.cand_n = 0

    def _levels(self, n):
        for z in range(3):
            self.warn[z] = n[z] >= self.t_warn if not self.warn[z] else n[z] >= self.t_warn - EXIT_MARGIN
            self.near[z] = n[z] >= self.t_near if not self.near[z] else n[z] >= self.t_near - EXIT_MARGIN
        return [2 if self.near[z] else (1 if self.warn[z] else 0) for z in range(3)]

    @staticmethod
    def _raw(lv, n):
        if lv[1] == 0:
            return CLEAR
        left_free, right_free = lv[0] < 2, lv[2] < 2
        if not left_free and not right_free:
            return STOP
        if left_free and right_free:
            return GO_LEFT if n[0] <= n[2] else GO_RIGHT
        return GO_LEFT if left_free else GO_RIGHT

    def update(self, n):
        """n = (nL, nC, nR) from zone_nearness. Returns dict(levels, raw, stable, leds)."""
        lv = self._levels(n)
        raw = self._raw(lv, n)
        if raw == STOP:
            self.stable, self.cand, self.cand_n = STOP, STOP, 0
        elif raw == self.stable:
            self.cand, self.cand_n = raw, 0
        else:
            self.cand_n = self.cand_n + 1 if raw == self.cand else 1
            self.cand = raw
            if self.cand_n >= CONFIRM:
                self.stable, self.cand_n = raw, 0
        leds = (lv[0] >= 1) | ((lv[0] == 2) << 1) | ((lv[1] >= 1) << 2) | ((lv[1] == 2) << 3) | ((lv[2] >= 1) << 4) | ((lv[2] == 2) << 5)
        leds |= (int(self.stable == STOP) << 6) | (1 << 7)
        return {"levels": lv, "raw": raw, "stable": self.stable, "leds": int(leds)}


if __name__ == "__main__":
    # gray conversion: weights sum to 256, white stays 255, black 0
    assert rgb_to_gray([[[255, 255, 255]]])[0, 0] == 255 and rgb_to_gray([[[0, 0, 0]]])[0, 0] == 0
    assert rgb_to_gray([[[255, 0, 0]]])[0, 0] == (77 * 255 + 128) >> 8
    assert input_i8([[[0, 128, 255]]]).tolist() == [[[-128, 0, 127]]]
    assert depth_byte([-128, 0, 127]).tolist() == [0, 128, 255]
    # zone nearness: far everywhere (depth kind: big byte = far), a near object (small byte) only in the left third
    H, W = 48, 64
    far = np.full((H, W), 230, dtype=np.int64)
    d = far.copy(); d[20:, 0:20] = 40
    nl, nc, nr = zone_nearness(d, "depth")
    assert nl == 255 - 40 and nc == 255 - 230 and nr == 255 - 230, (nl, nc, nr)
    # the ceiling quarter is ignored
    d2 = far.copy(); d2[0:H // 4, :] = 10
    assert zone_nearness(d2, "depth") == (25, 25, 25)
    # disparity kind flips the polarity
    assert zone_nearness(np.full((H, W), 200), "disparity") == (200, 200, 200)
    # agent: clear -> object ahead (go to the free side) -> hysteresis -> STOP when everything is near -> clear again
    ag = ObstacleAgent()
    clear = (30, 30, 30)
    for _ in range(4):
        r = ag.update(clear)
    assert r["stable"] == CLEAR and r["raw"] == CLEAR
    blocked_c = (40, 200, 60)                                  # centre near, left a bit nearer than right? left 40 < right 60 -> GO_LEFT
    out = [ag.update(blocked_c) for _ in range(3)]
    assert [o["raw"] for o in out] == [GO_LEFT] * 3
    assert [o["stable"] for o in out] == [CLEAR, CLEAR, GO_LEFT]   # needs CONFIRM = 3 frames
    assert ag.update((220, 220, 220))["stable"] == STOP                 # everything near: immediate STOP
    assert ag.update((220, 220, 220))["leds"] & (1 << 6)
    # hysteresis: a value just under the warn threshold keeps the zone in warn until it falls EXIT_MARGIN lower
    ag2 = ObstacleAgent(); ag2.update((0, T_WARN, 0)); assert ag2.warn[1]
    ag2.update((0, T_WARN - 1, 0)); assert ag2.warn[1]
    ag2.update((0, T_WARN - EXIT_MARGIN - 1, 0)); assert not ag2.warn[1]
    print("depth_ref self-test PASS")
