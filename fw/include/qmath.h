#ifndef QMATH_H
#define QMATH_H

#include <stdint.h>

int8_t qmath_rq(int64_t acc, int32_t bias, uint16_t M, uint8_t S, int8_t zp, int relu);
void qmath_fold_zp(int32_t *bias_out, const int32_t *bias_in, const int8_t *Wm, int K, int Co, int8_t z_in);
void qmath_im2col(const int8_t *fm, int H, int W, int C, int KH, int KW, int stride, int pad_top, int pad_left, int pad_bot, int pad_right, int8_t pad_value, int8_t *out);
void qmath_matmul_tiled(const int8_t *X, int V, int K, const int8_t *Wm, int Co, int N, int32_t *acc);

#endif
