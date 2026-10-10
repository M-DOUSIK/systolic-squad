#ifndef NPU_H
#define NPU_H

#include <stdint.h>

typedef struct {
    uintptr_t base_addr;
    uint8_t N;
    uint8_t pe_type;
    uint8_t has_zero_skip;
    uint8_t engine_id;
    uint8_t present;
    uint16_t acc_depth;              /* K-tile accumulator slots (contract v1.2, ACC_INFO); 0 = no accumulator */
    uint16_t xr_depth;               /* X replay buffer vectors (v1.3, XR_INFO); 0 = none */
    uint8_t  use_xr;                 /* 1: npu_matmul replays X for C_out tiles after the first (default when present) */
} npu_t;

void npu_init(npu_t *npu, uintptr_t base);
void npu_load_weights(npu_t *npu, const int8_t *W, int rows, int cols);
void npu_set_rq(npu_t *npu, int rq_en, int relu, int8_t zp, const uint16_t *M, const uint8_t *S, const int32_t *bias, int ncols);
void npu_run(npu_t *npu, const int8_t *X, int nvec, int K, void *Y, int raw_mode);
void npu_perf_read(npu_t *npu, uint32_t *cycles, uint32_t *active, uint32_t *vectors, uint32_t *zero_act, uint32_t *sleep_clk);
void npu_sleep(npu_t *npu, int sleep);

/* Contract v1.2 matmul with on-chip K accumulation:
 *   Y[v*ldy + co] = rq( sum_k X[v*ldx + k] * Wm[k*Co + co], bias[co], M[co], S[co], zp, relu )   for v < V, co < Co
 * X rows are K int8 values (ldx >= K), Wm is K x Co row-major (as in model_blob.bin), Y is int8.
 * Returns 0 on success, -1 if the NPU is absent or reported an error (sticky ERR bits). */
int npu_matmul(npu_t *npu, const int8_t *X, int V, int K, int ldx, const int8_t *Wm, int Co,
               const int32_t *bias, const uint16_t *M, const uint8_t *S, int8_t zp, int relu, int8_t *Y, int ldy);

#endif
