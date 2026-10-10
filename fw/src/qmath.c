#include "qmath.h"

int8_t qmath_rq(int64_t acc, int32_t bias, uint16_t M, uint8_t S, int8_t zp, int relu) {
    int64_t v = (acc + bias) * M;
    if (S > 0) {
        v = (v + (1LL << (S - 1))) >> S;
    }
    v += zp;
    int32_t lo = relu ? zp : -128;
    if (v < lo) return (int8_t)lo;
    if (v > 127) return 127;
    return (int8_t)v;
}

void qmath_fold_zp(int32_t *bias_out, const int32_t *bias_in, const int8_t *Wm, int K, int Co, int8_t z_in) {
    for (int c = 0; c < Co; c++) {
        int64_t sum_w = 0;
        for (int k = 0; k < K; k++) {
            sum_w += Wm[k * Co + c];
        }
        bias_out[c] = bias_in[c] - (int32_t)(z_in * sum_w);
    }
}

void qmath_im2col(const int8_t *fm, int H, int W, int C, int KH, int KW, int stride, int pad_top, int pad_left, int pad_bot, int pad_right, int8_t pad_value, int8_t *out) {
    int out_idx = 0;
    int H_out = (H + pad_top + pad_bot - KH) / stride + 1;
    int W_out = (W + pad_left + pad_right - KW) / stride + 1;
    
    for (int oy = 0; oy < H_out; oy++) {
        for (int ox = 0; ox < W_out; ox++) {
            for (int ky = 0; ky < KH; ky++) {
                for (int kx = 0; kx < KW; kx++) {
                    int iy = oy * stride + ky - pad_top;
                    int ix = ox * stride + kx - pad_left;
                    if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                        for (int c = 0; c < C; c++) {
                            out[out_idx++] = fm[(iy * W + ix) * C + c];
                        }
                    } else {
                        for (int c = 0; c < C; c++) {
                            out[out_idx++] = pad_value;
                        }
                    }
                }
            }
        }
    }
}

void qmath_matmul_tiled(const int8_t *X, int V, int K, const int8_t *Wm, int Co, int N, int32_t *acc) {
    for (int v = 0; v < V; v++) {
        for (int c = 0; c < Co; c++) {
            acc[v * Co + c] = 0;
        }
    }
    
    for (int t = 0; t < K; t += N) {
        int block_k = (t + N <= K) ? N : (K - t);
        for (int v = 0; v < V; v++) {
            for (int c = 0; c < Co; c++) {
                int32_t sum = 0;
                for (int k = 0; k < block_k; k++) {
                    sum += X[v * K + t + k] * Wm[(t + k) * Co + c];
                }
                acc[v * Co + c] += sum;
            }
        }
    }
}
