/*
 * canny.h  –  Canny edge detector for the board firmware (PolarFire SoC, RISC-V)
 *
 * Pipeline (bit-exact with canny_ref.py):
 *   1. scale   : x  = p >> 1          (uint8  → 0..127)
 *   2. blur    : b  = (conv3x3_rep(x, K_G) + 8) >> 4
 *                K_G = [[1,2,1],[2,4,2],[1,2,1]]
 *   3. sobel   : gx = conv3x3_rep(b, K_X),  gy = conv3x3_rep(b, K_Y)
 *                K_X = [[-1,0,1],[-2,0,2],[-1,0,1]]
 *                K_Y = [[-1,-2,-1],[0,0,0],[1,2,1]]
 *   4. mag     : |gx| + |gy|
 *   5. sector  : horizontal / vertical / diagonal / anti-diagonal
 *   6. NMS     : keep if mag > n1 AND mag >= n2
 *   7. hyster. : strong (≥high), weak (≥low & <high), 8-connected BFS/stack
 *   8. output  : 127 = edge, 0 = none  (int8_t)
 *
 * Border policy: REPLICATE PADDING (coordinate clamped) for steps 2 and 3.
 * No malloc, no OS calls, no printf.  Static-width types throughout.
 */

#ifndef CANNY_H
#define CANNY_H

#include <stdint.h>

/* -------------------------------------------------------------------------
 * Tuneable constant – must be >= H*W of any image passed to these functions.
 * Adjust at compile time:  -DMAX_CANNY_PIXELS=<n>
 * ------------------------------------------------------------------------- */
#ifndef MAX_CANNY_PIXELS
#define MAX_CANNY_PIXELS (640 * 480)   /* 307 200 – covers VGA */
#endif

/* The NPU handle type and driver API come from npu.h (canny.h used to define its own
 * `npu_t` behind the guard name NPU_H, which collided with npu.h's guard and typedef). */
#include "npu.h"

/* NPU operating modes */
typedef enum {
    NPU_MODE_RQ  = 0,   /* requantise: out = clip8( (acc*M + bias + rnd) >> S ) */
    NPU_MODE_RAW = 1    /* raw 32-bit accumulator, no requantisation            */
} npu_mode_t;

/* NPU configuration for a single-kernel 3×3 convolution */
typedef struct {
    npu_mode_t  mode;
    int32_t     M;          /* multiplier  (RQ mode)        */
    int32_t     S;          /* right-shift (RQ mode)        */
    int32_t     bias;       /* bias added before shift      */
    int32_t     zp_in;      /* input  zero-point            */
    int32_t     zp_out;     /* output zero-point            */
    int8_t      kernel[9];  /* row-major 3×3 kernel weights */
} npu_cfg_t;


/*
 * HAL functions that must be provided by the BSP / simulator:
 *
 * npu_conv3x3_rq  – run 3×3 conv with requantisation, replicate padding.
 *   in  : int8_t  input  [H][W]  (values 0..127 fit in int8_t w/ sign)
 *   out : int8_t  output [H][W]
 *
 * npu_conv3x3_raw – run 3×3 conv, raw int32 output, replicate padding.
 *   in  : int8_t  input  [H][W]
 *   out : int32_t output [H][W]
 *
 * Both functions use REPLICATE padding.
 * Returns 0 on success, negative on error.
 */
int npu_conv3x3_rq (npu_t *npu, const npu_cfg_t *cfg,
                    const int8_t *in, int8_t  *out, int H, int W);
int npu_conv3x3_raw(npu_t *npu, const npu_cfg_t *cfg,
                    const int8_t *in, int32_t *out, int H, int W);

/* =========================================================================
 * Public API
 * ========================================================================= */

/**
 * canny_cpu – pure-C Canny pipeline.
 *
 * @param gray     Input grayscale image, uint8_t [H*W], row-major.
 * @param H        Image height in pixels.
 * @param W        Image width  in pixels.
 * @param gx_buf   Caller-supplied scratch int32_t [H*W] – Sobel X output.
 * @param gy_buf   Caller-supplied scratch int32_t [H*W] – Sobel Y output.
 * @param blur_buf Caller-supplied scratch int8_t  [H*W] – blurred image.
 * @param out      Output edge map int8_t [H*W]; 127 = edge, 0 = none.
 * @param low      Low  hysteresis threshold (0..255, applied to |gx|+|gy|).
 * @param high     High hysteresis threshold (0..255).
 *
 * Preconditions: H*W <= MAX_CANNY_PIXELS, high >= low.
 */
void canny_cpu(const uint8_t *gray, int H, int W,
               int32_t *gx_buf, int32_t *gy_buf, int8_t *blur_buf,
               int8_t *out,
               uint8_t low, uint8_t high);

/**
 * canny_npu – NPU-accelerated Canny pipeline.
 *
 * Blur step uses NPU RQ mode (M=1, S=4, bias=8, zp=0).
 * Sobel step uses NPU RAW mode (two passes, K_X then K_Y).
 * mag / NMS / hysteresis run on the CPU (canny_postprocess).
 *
 * Buffer layout is identical to canny_cpu; the NPU writes into blur_buf,
 * gx_buf, and gy_buf so the caller can inspect intermediate results.
 */
void canny_npu(npu_t *npu,
               const uint8_t *gray, int H, int W,
               int32_t *gx_buf, int32_t *gy_buf, int8_t *blur_buf,
               int8_t *out,
               uint8_t low, uint8_t high);

/**
 * canny_postprocess – shared mag / NMS / hysteresis kernel.
 *
 * Called internally by both canny_cpu and canny_npu, but exposed here so
 * callers who generate gx/gy by other means can still use the downstream
 * pipeline.
 *
 * @param gx   Sobel-X gradient map int32_t [H*W].
 * @param gy   Sobel-Y gradient map int32_t [H*W].
 * @param H    Image height.
 * @param W    Image width.
 * @param out  Output edge map int8_t [H*W].
 * @param low  Low  threshold.
 * @param high High threshold.
 */
void canny_postprocess(const int32_t *gx, const int32_t *gy, int H, int W,
                       int8_t *out, uint8_t low, uint8_t high);

/**
 * canny_edge_score – edge pixel density.
 *
 * @param edge_map  int8_t [H*W] produced by canny_cpu / canny_npu.
 * @param H         Image height.
 * @param W         Image width.
 * @return          1000 * (number of edge pixels) / (H * W)  [fixed-point ‰]
 */
uint16_t canny_edge_score(const int8_t *edge_map, int H, int W);

#endif /* CANNY_H */
