#include "qmath.h"
#include "vec/vectors.h"
#include <stdio.h>
#include <stdlib.h>

#define CHECK(c) do { if (!(c)) { printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)

int test_qmath(void) {
    printf("test_qmath: START\n");
    /* rq() must equal reference/qmath_ref.py rq() on every generated vector (edge cases + random) */
    for (int i = 0; i < RQ_N; i++) {
        const rq_vec_t *v = &RQ_VEC[i];
        int8_t got = qmath_rq(v->acc, v->bias, v->M, v->S, v->zp, v->relu);
        if (got != v->expect) { printf("  FAIL rq vec %d: got %d expect %d\n", i, got, v->expect); return 1; }
    }
    /* im2col element order k = (ky*KW + kx)*C + ci, pad with the pad value */
    int8_t fm[5 * 5 * 2], col[9 * 18];
    for (int i = 0; i < 50; i++) fm[i] = (int8_t)i;
    qmath_im2col(fm, 5, 5, 2, 3, 3, 1, 0, 0, 0, 0, 0, col);
    CHECK(col[2] == fm[(0 * 5 + 1) * 2 + 0]);                       /* k=2 -> ky=0, kx=1, ci=0 */
    int8_t colp[4 * 9];
    int8_t one[2 * 2] = {1, 2, 3, 4};
    qmath_im2col(one, 2, 2, 1, 3, 3, 1, 1, 1, 1, 1, -5, colp);
    CHECK(colp[0] == -5 && colp[4] == 1);                           /* first window: padded corner, centre = pixel (0,0) */
    /* tiled matmul equals the plain matmul */
    int8_t X[7 * 37], W[37 * 5];
    int32_t acc[7 * 5];
    srand(3);
    for (int i = 0; i < 7 * 37; i++) X[i] = (int8_t)(rand() % 256 - 128);
    for (int i = 0; i < 37 * 5; i++) W[i] = (int8_t)(rand() % 255 - 127);
    qmath_matmul_tiled(X, 7, 37, W, 5, 16, acc);
    for (int v = 0; v < 7; v++) for (int c = 0; c < 5; c++) {
        int32_t s = 0; for (int k = 0; k < 37; k++) s += X[v * 37 + k] * W[k * 5 + c];
        CHECK(acc[v * 5 + c] == s);
    }
    printf("test_qmath: PASS (%d rq vectors)\n", RQ_N);
    return 0;
}
