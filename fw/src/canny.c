/*
 * canny.c  –  Canny edge detector implementation for the board firmware
 *             Bit-exact with canny_ref.py
 *
 * Rules:
 *   - No malloc, no OS calls, no printf
 *   - Only static memory and fixed-width types
 *   - All arithmetic is integer-only, bit-exact with Python reference
 *   - Replicate (clamp) padding for every convolution
 */

#include "canny.h"

#include <stdint.h>
#include <stddef.h>   /* NULL */

/* =========================================================================
 * Static intermediate buffers – shared across CPU and NPU paths.
 * These are module-level statics so no stack pressure and no malloc.
 * ========================================================================= */

/* Gradient magnitude map  (step 4) */
static int32_t mag_buf [MAX_CANNY_PIXELS];

/* Non-maximum suppression result  (step 6) */
static uint8_t keep_buf[MAX_CANNY_PIXELS];  /* 1 = kept after NMS */

/* Edge classification (step 7) */
static uint8_t edge_buf[MAX_CANNY_PIXELS];  /* 1 = confirmed edge  */
static uint8_t weak_buf[MAX_CANNY_PIXELS];  /* 1 = weak candidate  */

/*
 * Hysteresis BFS stack.
 * Each entry is a linearised pixel index (row*W + col).
 * Stored as int32_t so indices up to 2 billion are representable.
 */
static int32_t hyst_stack[MAX_CANNY_PIXELS];

/* =========================================================================
 * Gaussian blur kernel weights (row-major, 3×3)
 *   K_G = [[1,2,1],[2,4,2],[1,2,1]]
 * ========================================================================= */
static const int8_t K_G[9] = {
    1, 2, 1,
    2, 4, 2,
    1, 2, 1
};

/* =========================================================================
 * Sobel kernel weights
 *   K_X = [[-1,0,1],[-2,0,2],[-1,0,1]]
 *   K_Y = [[-1,-2,-1],[0,0,0],[1,2,1]]
 * ========================================================================= */
static const int8_t K_X[9] = {
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
};

static const int8_t K_Y[9] = {
    -1, -2, -1,
     0,  0,  0,
     1,  2,  1
};

/* =========================================================================
 * Helper: clamp an index to [0, limit-1]  (replicate / border-clamp)
 * ========================================================================= */
static inline int clamp_idx(int v, int limit)
{
    if (v < 0)       return 0;
    if (v >= limit)  return limit - 1;
    return v;
}

/* =========================================================================
 * Helper: absolute value of int32_t (avoid abs() / libm dependency)
 * ========================================================================= */
static inline int32_t iabs32(int32_t v)
{
    return (v < 0) ? -v : v;
}

/* =========================================================================
 * conv3x3_replicate_u8_raw
 *
 * 3×3 convolution on a uint8_t input plane with REPLICATE border.
 * Kernel weights are int8_t; accumulator and output are int32_t.
 * Used for Gaussian blur (before requantisation) and Sobel.
 *
 * @param in    Input  [H][W] uint8_t
 * @param out   Output [H][W] int32_t (raw accumulator)
 * @param H     Height
 * @param W     Width
 * @param k     Row-major 3×3 kernel, int8_t[9]
 * ========================================================================= */
static void conv3x3_replicate_u8_raw(const uint8_t *in, int32_t *out,
                                     int H, int W, const int8_t *k)
{
    int row, col, dr, dc;
    for (row = 0; row < H; ++row) {
        for (col = 0; col < W; ++col) {
            int32_t acc = 0;
            int ki = 0;
            for (dr = -1; dr <= 1; ++dr) {
                int rr = clamp_idx(row + dr, H);
                for (dc = -1; dc <= 1; ++dc) {
                    int cc = clamp_idx(col + dc, W);
                    acc += (int32_t)k[ki] * (int32_t)in[rr * W + cc];
                    ++ki;
                }
            }
            out[row * W + col] = acc;
        }
    }
}

/* =========================================================================
 * conv3x3_replicate_i8_raw
 *
 * Same as above but input plane is int8_t (used for Sobel on blur output).
 * ========================================================================= */
static void conv3x3_replicate_i8_raw(const int8_t *in, int32_t *out,
                                     int H, int W, const int8_t *k)
{
    int row, col, dr, dc;
    for (row = 0; row < H; ++row) {
        for (col = 0; col < W; ++col) {
            int32_t acc = 0;
            int ki = 0;
            for (dr = -1; dr <= 1; ++dr) {
                int rr = clamp_idx(row + dr, H);
                for (dc = -1; dc <= 1; ++dc) {
                    int cc = clamp_idx(col + dc, W);
                    acc += (int32_t)k[ki] * (int32_t)in[rr * W + cc];
                    ++ki;
                }
            }
            out[row * W + col] = acc;
        }
    }
}

/* =========================================================================
 * cpu_blur
 *
 * Steps 1 + 2 (CPU path):
 *   x[i] = gray[i] >> 1
 *   raw  = conv3x3_replicate(x, K_G)
 *   b[i] = (int8_t)((raw + 8) >> 4)          <- exact Python formula
 *
 * The blurred image is stored as int8_t.  Values of x are 0..127 which
 * fit in int8_t, but we use a uint8_t scratch implicitly through the
 * raw 32-bit accumulator – no separate x[] buffer is needed because we
 * inline the >> 1 into the pixel fetch.
 *
 * @param gray      uint8_t [H*W] input
 * @param blur_out  int8_t  [H*W] output (blurred, values 0..127)
 * @param H, W      image dimensions
 * @param tmp       int32_t [H*W] scratch (may alias gx_buf or gy_buf)
 * ========================================================================= */
static void cpu_blur(const uint8_t *gray, int8_t *blur_out,
                     int H, int W, int32_t *tmp)
{
    int i, row, col, dr, dc;
    int n = H * W;

    /* Step 1 + inline Gaussian: accumulate into tmp using x = gray>>1 */
    for (row = 0; row < H; ++row) {
        for (col = 0; col < W; ++col) {
            int32_t acc = 0;
            int ki = 0;
            for (dr = -1; dr <= 1; ++dr) {
                int rr = clamp_idx(row + dr, H);
                for (dc = -1; dc <= 1; ++dc) {
                    int cc = clamp_idx(col + dc, W);
                    int32_t xpix = (int32_t)(gray[rr * W + cc] >> 1);
                    acc += (int32_t)K_G[ki] * xpix;
                    ++ki;
                }
            }
            /* Step 2 requantisation: (acc + 8) >> 4 */
            tmp[row * W + col] = acc;
        }
    }

    for (i = 0; i < n; ++i) {
        int32_t v = (tmp[i] + 8) >> 4;
        /* v is 0..127 for valid 0..127 inputs – fits int8_t */
        blur_out[i] = (int8_t)v;
    }
}

/* =========================================================================
 * canny_postprocess  (exported – shared by CPU and NPU paths)
 *
 * Implements steps 4–8:
 *   4. mag = |gx| + |gy|
 *   5. direction sector
 *   6. NMS
 *   7. hysteresis (iterative stack, no recursion)
 *   8. write output (127 or 0)
 * ========================================================================= */
void canny_postprocess(const int32_t *gx, const int32_t *gy,
                       int H, int W,
                       int8_t *out, uint8_t low, uint8_t high)
{
    int     i, row, col;
    int32_t sp;          /* hysteresis stack pointer */
    int     n = H * W;

    /* ------------------------------------------------------------------
     * Step 4: magnitude map
     * ------------------------------------------------------------------ */
    for (i = 0; i < n; ++i) {
        mag_buf[i] = iabs32(gx[i]) + iabs32(gy[i]);
    }

    /* ------------------------------------------------------------------
     * Steps 5 + 6: sector detection + non-maximum suppression
     *
     * Sector boundaries (bit-exact with Python reference):
     *   ax = |gx|, ay = |gy|
     *   100*ay <= 41*ax            -> horizontal  neighbours (col+-1, row)
     *   41*ay  >= 100*ax           -> vertical    neighbours (col, row+-1)
     *   gx*gy  > 0                 -> diagonal    (col-1,row-1),(col+1,row+1)
     *   else                       -> anti-diag   (col+1,row-1),(col-1,row+1)
     *
     * NMS: keep pixel if mag > n1 AND mag >= n2
     *   where n1, n2 are the magnitudes of the two sector neighbours
     *   (pixels outside image -> 0).
     * ------------------------------------------------------------------ */
    for (row = 0; row < H; ++row) {
        for (col = 0; col < W; ++col) {
            int     idx = row * W + col;
            int32_t ax  = iabs32(gx[idx]);
            int32_t ay  = iabs32(gy[idx]);
            int32_t mag = mag_buf[idx];
            int32_t n1, n2;
            int     r1c1, r2c2;   /* linearised neighbour indices */
            int     nr1, nc1, nr2, nc2;

            /* Determine sector and neighbour coordinates */
            if ((int64_t)100 * ay <= (int64_t)41 * ax) {
                /* Horizontal */
                nr1 = row;     nc1 = col - 1;
                nr2 = row;     nc2 = col + 1;
            } else if ((int64_t)41 * ay >= (int64_t)100 * ax) {
                /* Vertical */
                nr1 = row - 1; nc1 = col;
                nr2 = row + 1; nc2 = col;
            } else if (gx[idx] * gy[idx] > 0) {
                /* Diagonal (NW-SE) */
                nr1 = row - 1; nc1 = col - 1;
                nr2 = row + 1; nc2 = col + 1;
            } else {
                /* Anti-diagonal (NE-SW) */
                nr1 = row - 1; nc1 = col + 1;
                nr2 = row + 1; nc2 = col - 1;
            }

            /* Neighbours outside image -> magnitude 0 */
            if (nr1 >= 0 && nr1 < H && nc1 >= 0 && nc1 < W) {
                r1c1 = nr1 * W + nc1;
                n1 = mag_buf[r1c1];
            } else {
                n1 = 0;
            }

            if (nr2 >= 0 && nr2 < H && nc2 >= 0 && nc2 < W) {
                r2c2 = nr2 * W + nc2;
                n2 = mag_buf[r2c2];
            } else {
                n2 = 0;
            }

            /* NMS: keep if mag strictly greater than n1 AND >= n2 */
            keep_buf[idx] = (mag > n1 && mag >= n2) ? 1u : 0u;
        }
    }

    /* ------------------------------------------------------------------
     * Step 7: hysteresis thresholding
     *
     * strong: kept AND mag >= high
     * weak  : kept AND low <= mag < high
     *
     * BFS/iterative flood from strong pixels through 8-connected weak
     * neighbours.  Uses hyst_stack[] – no recursion.
     * ------------------------------------------------------------------ */

    /* Classify pixels */
    for (i = 0; i < n; ++i) {
        int32_t mag = mag_buf[i];
        if (keep_buf[i]) {
            if (mag >= (int32_t)high) {
                edge_buf[i] = 1u;
                weak_buf[i] = 0u;
            } else if (mag >= (int32_t)low) {
                edge_buf[i] = 0u;
                weak_buf[i] = 1u;
            } else {
                edge_buf[i] = 0u;
                weak_buf[i] = 0u;
            }
        } else {
            edge_buf[i] = 0u;
            weak_buf[i] = 0u;
        }
    }

    /* Push all strong pixels onto the stack */
    sp = 0;
    for (i = 0; i < n; ++i) {
        if (edge_buf[i]) {
            hyst_stack[sp++] = (int32_t)i;
        }
    }

    /* Iterative 8-connected flood through weak pixels */
    while (sp > 0) {
        int32_t idx32 = hyst_stack[--sp];
        int     r     = (int)(idx32 / W);
        int     c     = (int)(idx32 % W);
        int     dr, dc;

        for (dr = -1; dr <= 1; ++dr) {
            int nr = r + dr;
            if (nr < 0 || nr >= H) continue;
            for (dc = -1; dc <= 1; ++dc) {
                int nc   = c + dc;
                int nidx;
                if (nc < 0 || nc >= W) continue;
                if (dr == 0 && dc == 0) continue;
                nidx = nr * W + nc;
                if (weak_buf[nidx]) {
                    weak_buf[nidx] = 0u;   /* consume */
                    edge_buf[nidx] = 1u;   /* confirm */
                    hyst_stack[sp++] = (int32_t)nidx;
                }
            }
        }
    }

    /* ------------------------------------------------------------------
     * Step 8: write output
     * ------------------------------------------------------------------ */
    for (i = 0; i < n; ++i) {
        out[i] = edge_buf[i] ? (int8_t)127 : (int8_t)0;
    }
}

/* =========================================================================
 * canny_cpu  –  pure-C implementation
 * ========================================================================= */
void canny_cpu(const uint8_t *gray, int H, int W,
               int32_t *gx_buf, int32_t *gy_buf, int8_t *blur_buf,
               int8_t *out,
               uint8_t low, uint8_t high)
{
    /* Step 1 + 2: scale + Gaussian blur (uses gx_buf as scratch) */
    cpu_blur(gray, blur_buf, H, W, gx_buf);

    /* Step 3: Sobel X and Y on blurred image (replicate border, int8 input) */
    conv3x3_replicate_i8_raw(blur_buf, gx_buf, H, W, K_X);
    conv3x3_replicate_i8_raw(blur_buf, gy_buf, H, W, K_Y);

    /* Steps 4–8 */
    canny_postprocess(gx_buf, gy_buf, H, W, out, low, high);
}

/* =========================================================================
 * canny_npu  –  NPU-accelerated implementation
 *
 * NPU RQ mode for blur:
 *   M=1, S=4, bias=8, zp_in=0, zp_out=0
 *   → out = clip_i8((acc*1 + 8) >> 4)
 *   This matches the CPU formula (acc + 8) >> 4 for values in 0..127.
 *
 * NPU RAW mode for Sobel:
 *   M=1, S=0, bias=0  → out = acc  (full int32 accumulator kept)
 *
 * The NPU functions (npu_conv3x3_rq / npu_conv3x3_raw) are provided by the
 * BSP / HAL and must implement REPLICATE border padding.
 * ========================================================================= */
void canny_npu(npu_t *npu,
               const uint8_t *gray, int H, int W,
               int32_t *gx_buf, int32_t *gy_buf, int8_t *blur_buf,
               int8_t *out,
               uint8_t low, uint8_t high)
{
    int i, n;
    n = H * W;

    /* ------------------------------------------------------------------
     * Step 1: scale  x[i] = gray[i] >> 1  into blur_buf temporarily.
     * We write into blur_buf (int8_t) because values are 0..127.
     * ------------------------------------------------------------------ */
    {   /* the HAL reads windows of `in` while writing `out`, so blur must not run in place */
        int8_t *xin = (int8_t *)keep_buf;   /* keep_buf is only needed later, by canny_postprocess */
        for (i = 0; i < n; ++i) {
            xin[i] = (int8_t)(gray[i] >> 1);
        }
    }

    /* ------------------------------------------------------------------
     * Step 2: Gaussian blur via NPU in RQ mode.
     *   formula: (acc + 8) >> 4  with K_G weights.
     *   NPU RQ:  clip_i8( (acc * M + bias) >> S )
     *            → M=1, bias=8, S=4
     * Output goes back into blur_buf.
     * ------------------------------------------------------------------ */
    {
        npu_cfg_t cfg;
        int ki;
        cfg.mode   = NPU_MODE_RQ;
        cfg.M      = 1;
        cfg.S      = 4;
        cfg.bias   = 0;   /* rq() already adds 1<<(S-1) = 8; bias=8 double-counted the rounding */
        cfg.zp_in  = 0;
        cfg.zp_out = 0;
        for (ki = 0; ki < 9; ++ki) {
            cfg.kernel[ki] = K_G[ki];
        }
        npu_conv3x3_rq(npu, &cfg, (const int8_t *)keep_buf, blur_buf, H, W);
    }

    /* ------------------------------------------------------------------
     * Step 3a: Sobel X via NPU RAW mode
     * ------------------------------------------------------------------ */
    {
        npu_cfg_t cfg;
        int ki;
        cfg.mode   = NPU_MODE_RAW;
        cfg.M      = 1;
        cfg.S      = 0;
        cfg.bias   = 0;
        cfg.zp_in  = 0;
        cfg.zp_out = 0;
        for (ki = 0; ki < 9; ++ki) {
            cfg.kernel[ki] = K_X[ki];
        }
        npu_conv3x3_raw(npu, &cfg, blur_buf, gx_buf, H, W);
    }

    /* ------------------------------------------------------------------
     * Step 3b: Sobel Y via NPU RAW mode
     * ------------------------------------------------------------------ */
    {
        npu_cfg_t cfg;
        int ki;
        cfg.mode   = NPU_MODE_RAW;
        cfg.M      = 1;
        cfg.S      = 0;
        cfg.bias   = 0;
        cfg.zp_in  = 0;
        cfg.zp_out = 0;
        for (ki = 0; ki < 9; ++ki) {
            cfg.kernel[ki] = K_Y[ki];
        }
        npu_conv3x3_raw(npu, &cfg, blur_buf, gy_buf, H, W);
    }

    /* Steps 4–8: CPU post-processing (mag, NMS, hysteresis) */
    canny_postprocess(gx_buf, gy_buf, H, W, out, low, high);
}

/* =========================================================================
 * canny_edge_score
 *
 * Returns 1000 * (edge_pixels) / (H * W)  [‰, fixed-point]
 * Uses integer arithmetic only.
 * ========================================================================= */
uint16_t canny_edge_score(const int8_t *edge_map, int H, int W)
{
    int32_t count = 0;
    int32_t total = (int32_t)H * (int32_t)W;
    int i;

    if (total == 0) {
        return 0u;
    }

    for (i = 0; i < total; ++i) {
        if (edge_map[i] != (int8_t)0) {
            ++count;
        }
    }

    /* Multiply first to preserve precision: (count * 1000) / total */
    return (uint16_t)((count * (int32_t)1000) / total);
}
