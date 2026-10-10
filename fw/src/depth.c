/* depth.c - see depth.h. Maths = reference/qmath_ref.py (rq, dwconv_q, add_q, upsample_nn, relu_q, bilinear_q), golden model = ml/qmodel.py. */
#include "depth.h"
#include "qmath.h"
#include "platform.h"
#include <string.h>

#define ALIGN16(x) (((x) + 15u) & ~(size_t)15u)
#define MAX_CH 2048                     /* widest layer (MiDaS small: 1392 channels) */

static size_t layer_out_bytes(const layer_desc_t *l) { return (size_t)l->out_h * l->out_w * l->cout; }

static size_t scratch_bytes(const model_desc_t *md)
{
    size_t m = 0;
    for (int i = 0; i < md->n_layers; i++) {
        const layer_desc_t *l = &md->layers[i];
        if (l->type == L_CONV || (l->type == L_PWCONV && (l->stride != 1 || l->pad_t || l->pad_l))) {
            size_t s = (size_t)l->out_h * l->out_w * l->kh * l->kw * l->cin;
            if (s > m) m = s;
        }
    }
    return m;
}

size_t depth_mem_bytes(const model_desc_t *m)
{
    size_t n = ALIGN16((size_t)PREPROC_H * PREPROC_W * PREPROC_C) + ALIGN16(scratch_bytes(m));
    for (int i = 0; i < m->n_layers; i++) n += ALIGN16(layer_out_bytes(&m->layers[i]));
    return n;
}

uint32_t depth_macs(const model_desc_t *m)
{
    uint32_t s = 0;
    for (int i = 0; i < m->n_layers; i++) {
        const layer_desc_t *l = &m->layers[i];
        uint32_t px = (uint32_t)l->out_h * l->out_w;
        if (l->type == L_CONV || l->type == L_PWCONV) s += px * l->cout * l->kh * l->kw * l->cin;
        else if (l->type == L_DWCONV) s += px * l->cout * l->kh * l->kw;
    }
    return s;
}

void depth_init(depth_net_t *d, const model_desc_t *m, const uint8_t *blob, uint8_t *mem, npu_t *npu)
{
    memset(d, 0, sizeof(*d));
    d->m = m;
    d->blob = blob;
    d->npu = npu;
    d->use_npu = npu && npu->present;
    d->in = (int8_t *)mem;      mem += ALIGN16((size_t)PREPROC_H * PREPROC_W * PREPROC_C);
    d->scratch = (int8_t *)mem; mem += ALIGN16(scratch_bytes(m));
    for (int i = 0; i < m->n_layers; i++) { d->act[i] = (int8_t *)mem; mem += ALIGN16(layer_out_bytes(&m->layers[i])); }
}

static const int8_t *src_of(const depth_net_t *d, int idx) { return idx == -2 ? d->in : d->act[idx]; }

/* im2col of output rows oy0..oy1-1 with padding value pad_v (normally in_zp), k = (ky*KW + kx)*C + ci (qmath_ref.im2col) */
static void im2col_rows(const layer_desc_t *l, const int8_t *x, int8_t *out, int oy0, int oy1)
{
    const int C = l->cin;
    out += (size_t)oy0 * l->out_w * l->kh * l->kw * C;
    for (int oy = oy0; oy < oy1; oy++)
        for (int ox = 0; ox < l->out_w; ox++)
            for (int ky = 0; ky < l->kh; ky++) {
                int iy = oy * l->stride + ky - l->pad_t;
                for (int kx = 0; kx < l->kw; kx++) {
                    int ix = ox * l->stride + kx - l->pad_l;
                    if (iy < 0 || iy >= l->in_h || ix < 0 || ix >= l->in_w) memset(out, l->pad_v, (size_t)C);
                    else memcpy(out, x + ((size_t)iy * l->in_w + ix) * C, (size_t)C);
                    out += C;
                }
            }
}

static void matmul_cpu(const int8_t *X, int V, int K, const int8_t *Wm, int Co, const int32_t *bias, const uint16_t *M,
                       const uint8_t *S, int8_t zp, int relu, int8_t *Y)
{
    static int32_t acc[MAX_CH];
    for (int v = 0; v < V; v++) {
        const int8_t *x = X + (size_t)v * K;
        for (int c = 0; c < Co; c++) acc[c] = 0;
        for (int k = 0; k < K; k++) {
            int32_t xv = x[k];
            if (xv == 0) continue;
            const int8_t *w = Wm + (size_t)k * Co;
            for (int c = 0; c < Co; c++) acc[c] += xv * w[c];
        }
        for (int c = 0; c < Co; c++) Y[(size_t)v * Co + c] = qmath_rq(acc[c], bias[c], M[c], S[c], zp, relu);
    }
}

/* ---------------------------------------------------------------------------------------------- CPU layers, split over the cores
 * Every CPU layer is cut into nparts slices of output rows (or elements); platform_parallel runs them on the app hart and the worker
 * harts. The slices write disjoint outputs, so the result does not depend on the number of parts. */
typedef struct {
    const layer_desc_t *l;
    const int8_t *x, *x2, *W;
    const int32_t *B;
    const uint16_t *M;
    const uint8_t *S;
    int8_t *y;
} job_t;

static void part_range(size_t n, int p, int np, size_t *a, size_t *b) { *a = n * (size_t)p / (size_t)np; *b = n * (size_t)(p + 1) / (size_t)np; }

/* depthwise conv (qmath_ref.dwconv_q): 8 channels at a time with the accumulators in registers */
static void dwconv_part(void *arg, int p, int np)
{
    const job_t *j = (const job_t *)arg;
    const layer_desc_t *l = j->l;
    const int C = l->cin;
    const int32_t pv = l->pad_v;
    size_t oy0, oy1;
    part_range(l->out_h, p, np, &oy0, &oy1);
    for (int oy = (int)oy0; oy < (int)oy1; oy++)
        for (int ox = 0; ox < l->out_w; ox++) {
            int8_t *o = j->y + ((size_t)oy * l->out_w + ox) * C;
            for (int c0 = 0; c0 < C; c0 += 8) {
                int nc = (C - c0 < 8) ? C - c0 : 8;
                int32_t a[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
                for (int ky = 0; ky < l->kh; ky++) {
                    int iy = oy * l->stride + ky - l->pad_t;
                    for (int kx = 0; kx < l->kw; kx++) {
                        int ix = ox * l->stride + kx - l->pad_l;
                        const int8_t *w = j->W + ((size_t)ky * l->kw + kx) * C + c0;
                        int inside = iy >= 0 && iy < l->in_h && ix >= 0 && ix < l->in_w;
                        const int8_t *px = j->x + ((size_t)(inside ? iy : 0) * l->in_w + (inside ? ix : 0)) * C + c0;
                        if (nc == 8) {
                            if (inside) {
#pragma GCC unroll 8
                                for (int q = 0; q < 8; q++) a[q] += (int32_t)px[q] * w[q];
                            } else {
#pragma GCC unroll 8
                                for (int q = 0; q < 8; q++) a[q] += pv * w[q];
                            }
                        } else {
                            for (int q = 0; q < nc; q++) a[q] += (inside ? (int32_t)px[q] : pv) * w[q];
                        }
                    }
                }
                for (int q = 0; q < nc; q++) o[c0 + q] = qmath_rq(a[q], j->B[c0 + q], j->M[c0 + q], j->S[c0 + q], l->out_zp, l->relu);
            }
        }
}

static void add_part(void *arg, int p, int np)
{
    const job_t *j = (const job_t *)arg;
    const layer_desc_t *l = j->l;
    size_t i0, i1;
    part_range(layer_out_bytes(l), p, np, &i0, &i1);
    int lo = l->relu ? l->out_zp : -128;
    for (size_t i = i0; i < i1; i++) {
        int64_t v = (int64_t)(j->x[i] - l->za) * l->Ma + (int64_t)(j->x2[i] - l->zb) * l->Mb;
        if (l->S_add > 0) v = (v + (1LL << (l->S_add - 1))) >> l->S_add;   /* arithmetic shift on int64 (GCC RISC-V) */
        v += l->out_zp;
        j->y[i] = (int8_t)(v < lo ? lo : (v > 127 ? 127 : v));
    }
}

static void upsample(const layer_desc_t *l, const int8_t *x, int8_t *y)
{
    const int C = l->cout, f = l->factor;
    for (int oy = 0; oy < l->out_h; oy++)
        for (int ox = 0; ox < l->out_w; ox++)
            memcpy(y + ((size_t)oy * l->out_w + ox) * C, x + ((size_t)(oy / f) * l->in_w + ox / f) * C, (size_t)C);
}

/* standalone ReLU on quantised values: y = max(x, zp) (qmath_ref.relu_q) */
static void relu_part(void *arg, int p, int np)
{
    const job_t *j = (const job_t *)arg;
    size_t i0, i1;
    part_range(layer_out_bytes(j->l), p, np, &i0, &i1);
    const int8_t zp = j->l->out_zp;
    for (size_t i = i0; i < i1; i++) j->y[i] = j->x[i] < zp ? zp : j->x[i];
}

/* one source index + weight (1/256) per output index (qmath_ref.bilinear_taps) */
static void bilinear_tap(int o, int n_in, int n_out, int align, int *i0, int *i1, int *w1)
{
    int num, den;
    if (align) { num = o * (n_in - 1); den = n_out > 1 ? n_out - 1 : 1; }
    else       { num = (2 * o + 1) * n_in - n_out; if (num < 0) num = 0; den = 2 * n_out; }
    int i = num / den, rem = num % den;
    *w1 = (rem * 512 + den) / (2 * den);
    *i0 = i < n_in - 1 ? i : n_in - 1;
    *i1 = i + 1 < n_in - 1 ? i + 1 : n_in - 1;
}

/* bilinear upsample, weights in 1/256: y = (sum wy*wx*x + 32768) >> 16 (qmath_ref.bilinear_q); scale and zero point unchanged */
static void bilinear_part(void *arg, int p, int np)
{
    const job_t *j = (const job_t *)arg;
    const layer_desc_t *l = j->l;
    const int C = l->cout;
    size_t oy0, oy1;
    part_range(l->out_h, p, np, &oy0, &oy1);
    for (int oy = (int)oy0; oy < (int)oy1; oy++) {
        int y0, y1, wy1;
        bilinear_tap(oy, l->in_h, l->out_h, l->align, &y0, &y1, &wy1);
        int wy0 = 256 - wy1;
        for (int ox = 0; ox < l->out_w; ox++) {
            int x0, x1, wx1;
            bilinear_tap(ox, l->in_w, l->out_w, l->align, &x0, &x1, &wx1);
            int wx0 = 256 - wx1;
            const int8_t *x = j->x;
            const int8_t *a = x + ((size_t)y0 * l->in_w + x0) * C, *b = x + ((size_t)y0 * l->in_w + x1) * C;
            const int8_t *c = x + ((size_t)y1 * l->in_w + x0) * C, *e = x + ((size_t)y1 * l->in_w + x1) * C;
            int8_t *o = j->y + ((size_t)oy * l->out_w + ox) * C;
            for (int ch = 0; ch < C; ch++) {
                int32_t v = wy0 * (wx0 * a[ch] + wx1 * b[ch]) + wy1 * (wx0 * c[ch] + wx1 * e[ch]);
                o[ch] = (int8_t)((v + 32768) >> 16);            /* arithmetic shift = floor, as Python's >> */
            }
        }
    }
}

/* im2col of a slice of output rows (the matmul itself runs on the NPU or matmul_cpu) */
static void im2col_part(void *arg, int p, int np)
{
    const job_t *j = (const job_t *)arg;
    const layer_desc_t *l = j->l;
    size_t oy0, oy1;
    part_range(l->out_h, p, np, &oy0, &oy1);
    im2col_rows(l, j->x, j->y, (int)oy0, (int)oy1);
}
int depth_run(depth_net_t *d, const uint8_t *rgb)
{
    const size_t n_in = (size_t)PREPROC_H * PREPROC_W * PREPROC_C;
    for (size_t i = 0; i < n_in; i++) d->in[i] = (int8_t)((int)rgb[i] - 128);
    d->t_npu_us = d->t_cpu_us = 0;
    int errs = 0;
    for (int i = 0; i < d->m->n_layers; i++) {
        const layer_desc_t *l = &d->m->layers[i];
        const int8_t *x = src_of(d, l->in_a);
        int8_t *y = d->act[i];
        const int8_t   *W = (const int8_t *)(d->blob + l->w_off);
        const int32_t  *B = (const int32_t *)(const void *)(d->blob + l->b_off);
        const uint16_t *M = (const uint16_t *)(const void *)(d->blob + l->m_off);
        const uint8_t  *S = d->blob + l->s_off;
        job_t job = { l, x, l->type == L_ADD ? src_of(d, l->in_b) : NULL, W, B, M, S, y };
        const int NP = platform_cores();
        uint32_t t0 = time_us();
        int on_npu = 0;
        switch (l->type) {
        case L_CONV:
        case L_PWCONV: {
            const int8_t *X = x;
            int K = l->kh * l->kw * l->cin, V = l->out_h * l->out_w;
            if (!(l->type == L_PWCONV && l->stride == 1 && l->pad_t == 0 && l->pad_l == 0)) {
                job.y = d->scratch;
                platform_parallel(im2col_part, &job, NP);
                X = d->scratch;
            }
            if (d->use_npu) {
                on_npu = 1;
                if (npu_matmul(d->npu, X, V, K, K, W, l->cout, B, M, S, l->out_zp, l->relu, y, l->cout) != 0) errs++;
            } else {
                matmul_cpu(X, V, K, W, l->cout, B, M, S, l->out_zp, l->relu, y);
            }
            break;
        }
        case L_DWCONV:   platform_parallel(dwconv_part, &job, NP); break;
        case L_UPSAMPLE: upsample(l, x, y); break;
        case L_ADD:      platform_parallel(add_part, &job, NP); break;
        case L_RELU:     platform_parallel(relu_part, &job, NP); break;
        case L_BILINEAR: platform_parallel(bilinear_part, &job, NP); break;
        default: break;
        }
        uint32_t dt = time_us() - t0;
        d->t_layer_us[i] = dt;
        if (on_npu) d->t_npu_us += dt; else d->t_cpu_us += dt;
    }
    d->npu_errors = errs;
    return errs;
}

const int8_t *depth_output(const depth_net_t *d) { return d->act[d->m->n_layers - 1]; }
