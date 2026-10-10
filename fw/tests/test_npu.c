#include "npu.h"
#include "qmath.h"
#include "test_harness.h"
#include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)
#define NMAX 16

static int run_engine(uintptr_t base, int expect_n, int expect_engine) {
    npu_t npu;
    npu_init(&npu, base);
    CHECK(npu.present == 1 && npu.N == expect_n && npu.engine_id == expect_engine);
    CHECK(npu.pe_type == expect_engine);                           /* engine 0 = 16x16, engine 1 = 4x4 test engine */
    int N = npu.N;
    for (int round = 0; round < 6; round++) {                       /* several weight tiles back to back = weight swaps */
        int8_t W[NMAX * NMAX], X[50 * NMAX];
        uint16_t M[NMAX]; uint8_t S[NMAX]; int32_t b[NMAX];
        int rq = round & 1, relu = (round >> 1) & 1; int8_t zp = (int8_t)(round * 37 - 100);
        for (int i = 0; i < N * N; i++) W[i] = (int8_t)(rand() % 255 - 127);
        for (int i = 0; i < 50 * N; i++) X[i] = (rand() % 4 == 0) ? 0 : (int8_t)(rand() % 256 - 128);   /* zeros -> PERF_ZERO_ACT */
        for (int c = 0; c < N; c++) { M[c] = (uint16_t)(1 + rand() % 3000); S[c] = (uint8_t)(rand() % 20); b[c] = rand() % 20000 - 10000; }
        npu_load_weights(&npu, W, N, N);
        npu_set_rq(&npu, rq, relu, zp, M, S, b, N);
        static int32_t y32[50 * NMAX]; static int8_t y8[50 * NMAX];
        npu_run(&npu, X, 50, N, rq ? (void *)y8 : (void *)y32, !rq);   /* 50 > FIFO depth: exercises the push/pop flow */
        for (int v = 0; v < 50; v++) for (int c = 0; c < N; c++) {
            int32_t acc = 0;
            for (int r = 0; r < N; r++) acc += (int32_t)X[v * N + r] * W[r * N + c];
            if (rq) { int8_t e = qmath_rq(acc, b[c], M[c], S[c], zp, relu); CHECK(y8[v * N + c] == e); }
            else    { CHECK(y32[v * N + c] == acc); }
        }
    }
    uint32_t cyc, act, vec, zero, slp;
    npu_perf_read(&npu, &cyc, &act, &vec, &zero, &slp);
    CHECK(vec == 300 && zero > 0);
    return 0;
}

int test_npu(void) {
    printf("test_npu: START\n");
    npu_sim_setup();
    srand(11);
    if (run_engine(0x0000, 16, 0)) return 1;                       /* BIG at NPU_BASE */
    if (run_engine(0x1000, 4, 1)) return 1;                        /* LITTLE at NPU_BASE + 0x1000 */
    /* sticky error bits: read from an empty OUT FIFO */
    CHECK(npu_reg_read(0x01C) == 0xDEADBEEFu);
    CHECK((npu_reg_read(0x00C) >> 26) & 1);
    npu_reg_write(0x040, 1u << 26);   /* v1.2: ERR_CLR bits = STATUS bit positions */
    CHECK(((npu_reg_read(0x00C) >> 26) & 1) == 0);
    /* scratch register */
    npu_reg_write(0x044, 0xA5A5A5A5u);
    CHECK(npu_reg_read(0x044) == 0xA5A5A5A5u);
    printf("test_npu: PASS (BIG N=16 and LITTLE N=4: RAW + RQ, weight swaps, zero-skip counter, error bits)\n");
    return 0;
}
