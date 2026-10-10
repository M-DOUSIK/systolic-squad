/*
 * npu_conv3x3.c - the two HAL functions canny.c needs (declared in canny.h), written on top of the NPU driver (npu.h).
 * Used by canny_npu (blur and Sobel on the NPU).
 *
 * One 3x3 convolution = a matmul with K = 9 (window element k = ky*3 + kx), one output column,
 * REPLICATE padding (clamp the coordinate), one vector per output pixel.
 *   RQ  : needs the whole window in one K-tile (N >= 9, i.e. the BIG engine): hardware RQ mode, contract rq().
 *   RAW : N >= 9 -> one tile; N < 9 (LITTLE, N = 4) -> K cut into tiles of N, INT32 sums added here.
 * If N < 9 and RQ is asked, the sum is done as RAW and qmath_rq() is applied on the CPU (same result, bit-exact).
 */
#include "canny.h"
#include "npu.h"
#include "qmath.h"

#define CONV_CHUNK 128                         /* pixels per npu_run call (static buffers, no malloc) */
#define MAXN       16

static int8_t  xbuf[CONV_CHUNK * MAXN];        /* window vectors, row stride = K */
static int32_t ybuf[CONV_CHUNK * MAXN];        /* RAW results (RQ results are INT8 packed in the same memory) */

static int clampi(int v, int hi) { return v < 0 ? 0 : (v > hi ? hi : v); }

static void load_tile(npu_t *npu, const npu_cfg_t *cfg, int k0, int hw_rq, const uint16_t *M, const uint8_t *S, const int32_t *bias)
{
    int N = npu->N;
    int8_t wtile[MAXN * MAXN];
    for (int i = 0; i < N * N; i++) wtile[i] = 0;
    for (int k = 0; k < N && k0 + k < 9; k++) wtile[k * N + 0] = cfg->kernel[k0 + k];   /* Wm[k][0] */
    npu_load_weights(npu, wtile, N, N);
    npu_set_rq(npu, hw_rq, 0, (int8_t)cfg->zp_out, M, S, bias, N);
}

static int conv3x3(npu_t *npu, const npu_cfg_t *cfg, const int8_t *in, void *out, int H, int W, int want_rq)
{
    int N = npu->N, npx = H * W;
    uint16_t M[MAXN];
    uint8_t  S[MAXN];
    int32_t  bias[MAXN];
    int32_t  acc[CONV_CHUNK];
    int hw_rq = want_rq && N >= 9;
    int ntiles = (N >= 9) ? 1 : (9 + N - 1) / N;

    if (!npu->present || N > MAXN || N < 4) return -1;
    for (int c = 0; c < N; c++) { M[c] = 1; S[c] = 0; bias[c] = 0; }
    if (hw_rq) { M[0] = (uint16_t)cfg->M; S[0] = (uint8_t)cfg->S; bias[0] = cfg->bias; }
    if (ntiles == 1) load_tile(npu, cfg, 0, hw_rq, M, S, bias);        /* weights stay for the whole frame */

    for (int px = 0; px < npx; px += CONV_CHUNK) {
        int nv = (npx - px < CONV_CHUNK) ? (npx - px) : CONV_CHUNK;
        for (int v = 0; v < nv; v++) acc[v] = 0;
        for (int kt = 0; kt < ntiles; kt++) {
            int k0 = kt * N, K = (ntiles == 1) ? 9 : N;
            if (ntiles > 1) load_tile(npu, cfg, k0, 0, M, S, bias);    /* LITTLE fallback: RAW tiles, summed here */
            for (int v = 0; v < nv; v++) {
                int y = (px + v) / W, x = (px + v) % W;
                for (int k = 0; k < K; k++) {
                    int kk = k0 + k;
                    if (kk < 9) {
                        int yy = clampi(y + kk / 3 - 1, H - 1), xx = clampi(x + kk % 3 - 1, W - 1);
                        xbuf[v * K + k] = in[yy * W + xx];
                    } else {
                        xbuf[v * K + k] = 0;
                    }
                }
            }
            npu_run(npu, xbuf, nv, K, ybuf, !hw_rq);
            for (int v = 0; v < nv; v++) {
                if (hw_rq) ((int8_t *)out)[px + v] = ((const int8_t *)ybuf)[v * N + 0];
                else       acc[v] += ybuf[v * N + 0];
            }
        }
        if (!hw_rq) {
            for (int v = 0; v < nv; v++) {
                if (want_rq) ((int8_t *)out)[px + v] = qmath_rq(acc[v], cfg->bias, (uint16_t)cfg->M, (uint8_t)cfg->S, (int8_t)cfg->zp_out, 0);
                else         ((int32_t *)out)[px + v] = acc[v];
            }
        }
    }
    return 0;
}

int npu_conv3x3_rq(npu_t *npu, const npu_cfg_t *cfg, const int8_t *in, int8_t *out, int H, int W)
{
    return conv3x3(npu, cfg, in, out, H, W, 1);
}

int npu_conv3x3_raw(npu_t *npu, const npu_cfg_t *cfg, const int8_t *in, int32_t *out, int H, int W)
{
    return conv3x3(npu, cfg, in, out, H, W, 0);
}
