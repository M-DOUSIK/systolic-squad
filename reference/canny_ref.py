"""Integer Canny edge detector — THE reference. Training data, firmware (C) and hardware must all
produce exactly this output, bit for bit. Run `python reference/canny_ref.py` for the self-test.

Pipeline (all integers, image H x W, row-major, gray uint8 input p):
  1. x = p >> 1                                   (0..127, fits the array's signed INT8 input)
  2. blur  b = (sum(K_G * x_window) + 8) >> 4      K_G = [[1,2,1],[2,4,2],[1,2,1]]   -> 0..127   (array: 3x3 conv)
  3. gx = sum(K_X * b_window), gy = sum(K_Y * b_window)                                           (array: 3x3 conv, 2 cols)
         K_X = [[-1,0,1],[-2,0,2],[-1,0,1]],  K_Y = [[-1,-2,-1],[0,0,0],[1,2,1]]   (y grows downwards)
  4. mag = |gx| + |gy|                            (0..1016)
  5. direction sector (ax=|gx|, ay=|gy|):
         100*ay <= 41*ax  -> horizontal gradient: neighbours (x-1,y) and (x+1,y)
         41*ay >= 100*ax  -> vertical gradient:   neighbours (x,y-1) and (x,y+1)
         gx*gy > 0        -> diagonal:            neighbours (x-1,y-1) and (x+1,y+1)
         else             -> anti-diagonal:       neighbours (x+1,y-1) and (x-1,y+1)
  6. non-maximum suppression: keep if mag > mag(first neighbour listed) and mag >= mag(second neighbour listed)
     (neighbours outside the image count as 0)
  7. hysteresis: strong = kept and mag >= high; weak = kept and low <= mag < high.
     edge = strong pixels + weak pixels 8-connected to a strong pixel through other weak pixels.
     (This set is unique, so any correct implementation — stack, queue, repeated passes — gives the same result.)
  8. output: 127 for edge, 0 otherwise (INT8, ready to be the CNN input; zero point 0)
Borders: steps 2 and 3 use replicate padding (clamp coordinates), so every stage keeps the H x W size.
"""
import numpy as np

K_G = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.int64)
K_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int64)
K_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.int64)


def conv3x3_replicate(img, k):
    """img int64 [H][W]; returns sum over the 3x3 window with clamped (replicated) borders."""
    p = np.pad(img, 1, mode="edge")
    H, W = img.shape
    out = np.zeros((H, W), dtype=np.int64)
    for dy in range(3):
        for dx in range(3):
            out += k[dy, dx] * p[dy:dy + H, dx:dx + W]
    return out


def canny_ref(gray_u8, low=40, high=100, return_stages=False):
    g = np.asarray(gray_u8, dtype=np.int64)
    assert g.ndim == 2 and g.min() >= 0 and g.max() <= 255
    x = g >> 1
    b = (conv3x3_replicate(x, K_G) + 8) >> 4
    gx = conv3x3_replicate(b, K_X)
    gy = conv3x3_replicate(b, K_Y)
    mag = np.abs(gx) + np.abs(gy)
    H, W = mag.shape

    def m(yy, xx):
        return mag[yy, xx] if 0 <= yy < H and 0 <= xx < W else 0

    keep = np.zeros((H, W), dtype=bool)
    for yy in range(H):
        for xx in range(W):
            ax, ay = abs(int(gx[yy, xx])), abs(int(gy[yy, xx]))
            if 100 * ay <= 41 * ax:
                n1, n2 = m(yy, xx - 1), m(yy, xx + 1)
            elif 41 * ay >= 100 * ax:
                n1, n2 = m(yy - 1, xx), m(yy + 1, xx)
            elif int(gx[yy, xx]) * int(gy[yy, xx]) > 0:
                n1, n2 = m(yy - 1, xx - 1), m(yy + 1, xx + 1)
            else:
                n1, n2 = m(yy - 1, xx + 1), m(yy + 1, xx - 1)
            v = mag[yy, xx]
            keep[yy, xx] = v > n1 and v >= n2

    strong = keep & (mag >= high)
    weak = keep & (mag >= low) & (mag < high)
    edge = strong.copy()
    stack = list(zip(*np.nonzero(strong)))
    while stack:
        yy, xx = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = yy + dy, xx + dx
                if 0 <= ny < H and 0 <= nx < W and weak[ny, nx] and not edge[ny, nx]:
                    edge[ny, nx] = True
                    stack.append((ny, nx))
    out = np.where(edge, 127, 0).astype(np.int8)
    if return_stages:
        return out, dict(x=x, blur=b, gx=gx, gy=gy, mag=mag, keep=keep, strong=strong, weak=weak)
    return out


def _hysteresis_by_passes(keep, mag, low, high):
    """Independent second implementation (repeated sweeps) used only to prove the result is order-independent."""
    strong = keep & (mag >= high)
    weak = keep & (mag >= low) & (mag < high)
    edge = strong.copy()
    H, W = mag.shape
    changed = True
    while changed:
        changed = False
        p = np.pad(edge, 1)
        nb = np.zeros_like(edge)
        for dy in range(3):
            for dx in range(3):
                nb |= p[dy:dy + H, dx:dx + W]
        new = edge | (weak & nb)
        if (new != edge).any():
            edge, changed = new, True
    return np.where(edge, 127, 0).astype(np.int8)


if __name__ == "__main__":
    # 1. flat image -> no edges
    assert (canny_ref(np.full((16, 16), 200)) == 0).all()
    # 2. bright square on dark background -> edges only around its border, one pixel thick-ish, none inside/outside
    img = np.zeros((32, 32), dtype=np.int64); img[8:24, 8:24] = 240
    e = canny_ref(img)
    assert e[16, 16] == 0 and e[2, 2] == 0, "no edges in flat regions"
    assert e[16, 7:10].max() == 127 and e[7:10, 16].max() == 127, "left and top borders found"
    assert (e[16, :] == 127).sum() <= 4, "thin edges (NMS works)"
    # 3. hysteresis is order-independent: compare with a different algorithm on random images
    rng = np.random.default_rng(0)
    for t in range(5):
        r = rng.integers(0, 256, (24, 24))
        r = (r + np.roll(r, 1, 0) + np.roll(r, 1, 1)) // 3          # some structure
        out, st = canny_ref(r, 30, 80, return_stages=True)
        assert (out == _hysteresis_by_passes(st["keep"], st["mag"], 30, 80)).all()
    # 4. ranges
    _, st = canny_ref(rng.integers(0, 256, (20, 20)), return_stages=True)
    assert st["blur"].min() >= 0 and st["blur"].max() <= 127 and st["mag"].max() <= 1016
    print("canny_ref self-test PASS  (square border edges:", int((e == 127).sum()), "pixels)")
