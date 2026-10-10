#include "npu.h"
#include "platform.h"
#include <stddef.h>

#define REG_ID          0x000
#define REG_CFG         0x004
#define REG_CTRL        0x008
#define REG_STATUS      0x00C
#define REG_W_DATA      0x010
#define REG_W_COMMIT    0x014
#define REG_X_DATA      0x018
#define REG_Y_DATA      0x01C
#define REG_RQ_MULT     0x020
#define REG_RQ_SHIFT    0x024
#define REG_COL_IDX     0x028
#define REG_BIAS_DATA   0x02C
#define REG_PERF_CYCLES 0x030
#define REG_PERF_ACTIVE 0x034
#define REG_PERF_VECTORS 0x038
#define REG_PERF_CTRL   0x03C
#define REG_ERR_CLR     0x040
#define REG_OUT_ZP      0x04C
#define REG_PERF_ZERO_ACT 0x050
#define REG_PERF_SLEEP  0x054
#define REG_ACC_PASS    0x058
#define REG_ACC_INFO    0x05C
#define REG_XR_WPTR     0x060
#define REG_XR_REPLAY   0x064
#define REG_XR_INFO     0x068
#define ST_RP_ACTIVE    (1u << 20)

#define ST_BUSY      (1u << 16)
#define ST_W_READY   (1u << 17)
#define ST_OUT_EMPTY (1u << 19)
#define ST_ERRS      (0xFu << 24)
#define IN_DEPTH     32

void npu_init(npu_t *npu, uintptr_t base) {
    npu->base_addr = base;
    uint32_t id = npu_reg_read(base + REG_ID);
    uint32_t cfg0 = npu_reg_read(base + REG_CFG);
    if (id == 0x53514431 && ((cfg0 & 0xFF) == 4 || (cfg0 & 0xFF) == 8 || (cfg0 & 0xFF) == 16)) {   /* bring-up probe: ID but CFG 0 -> absent */
        npu->present = 1;
        uint32_t cfg = cfg0;
        npu->N = cfg & 0xFF;
        npu->pe_type = (cfg >> 16) & 1;
        npu->has_zero_skip = (cfg >> 17) & 1;
        npu->engine_id = (cfg >> 18) & 7;
        npu->acc_depth = ((cfg >> 24) >= 2) ? (uint16_t)npu_reg_read(base + REG_ACC_INFO) : 0;
        npu->xr_depth = ((cfg >> 24) >= 3) ? (uint16_t)npu_reg_read(base + REG_XR_INFO) : 0;
        npu->use_xr = npu->xr_depth > 0;
        npu_reg_write(base + REG_PERF_CTRL, 1);
        npu_reg_write(base + REG_PERF_CTRL, 2);       /* perf counters on */
    } else {
        npu->present = 0;
    }
}

static void wait_busy(npu_t *npu) {
    while (npu_reg_read(npu->base_addr + REG_STATUS) & (1<<16)) {}
}

static void wait_wready(npu_t *npu) {
    while (!(npu_reg_read(npu->base_addr + REG_STATUS) & (1<<17))) {}
}

void npu_load_weights(npu_t *npu, const int8_t *W, int rows, int cols) {
    if (!npu->present) return;
    int N = npu->N;
    for (int c = 0; c < cols; c += N) {
        for (int r = 0; r < rows; r += N) {
            wait_wready(npu);
            for (int rr = N - 1; rr >= 0; rr--) {
                for (int cc = 0; cc < N; cc += 4) {
                    uint32_t word = 0;
                    for (int b = 0; b < 4; b++) {
                        int r_idx = r + rr;
                        int c_idx = c + cc + b;
                        uint8_t val = (r_idx < rows && c_idx < cols) ? W[r_idx * cols + c_idx] : 0;
                        word |= (val << (b * 8));
                    }
                    npu_reg_write(npu->base_addr + REG_W_DATA, word);
                }
            }
            npu_reg_write(npu->base_addr + REG_W_COMMIT, 1);
        }
    }
}

void npu_set_rq(npu_t *npu, int rq_en, int relu, int8_t zp, const uint16_t *M, const uint8_t *S, const int32_t *bias, int ncols) {
    if (!npu->present) return;
    wait_busy(npu);
    uint32_t ctrl = npu_reg_read(npu->base_addr + REG_CTRL);
    if (rq_en) ctrl |= (1<<1); else ctrl &= ~(1<<1);
    if (relu) ctrl |= (1<<2); else ctrl &= ~(1<<2);
    ctrl &= ~(1u << 5);                               /* v1.2: single-K-tile path, accumulator off */
    npu_reg_write(npu->base_addr + REG_CTRL, ctrl);
    npu_reg_write(npu->base_addr + REG_OUT_ZP, (uint8_t)zp);
    
    npu_reg_write(npu->base_addr + REG_COL_IDX, 0);
    for (int i = 0; i < ncols; i++) {
        npu_reg_write(npu->base_addr + REG_RQ_MULT, M[i]);
        npu_reg_write(npu->base_addr + REG_RQ_SHIFT, S[i]);
        npu_reg_write(npu->base_addr + REG_BIAS_DATA, bias[i]);
    }
}

void npu_run(npu_t *npu, const int8_t *X, int nvec, int K, void *Y, int raw_mode) {
    if (!npu->present) return;
    int N = npu->N;
    int words_per_vec = N / 4;
    int pushed = 0, popped = 0;
    
    int8_t *Y8 = (int8_t*)Y;
    int32_t *Y32 = (int32_t*)Y;
    
    while (popped < nvec) {
        uint32_t status = npu_reg_read(npu->base_addr + REG_STATUS);
        int in_count = status & 0xFF;
        int out_count = (status >> 8) & 0xFF;
        
        while (in_count < 32 && pushed < nvec) {
            for (int w = 0; w < words_per_vec; w++) {
                uint32_t word = 0;
                for (int b = 0; b < 4; b++) {
                    int k_idx = w * 4 + b;
                    uint8_t val = (k_idx < K) ? X[pushed * K + k_idx] : 0;
                    word |= (val << (b * 8));
                }
                npu_reg_write(npu->base_addr + REG_X_DATA, word);
            }
            pushed++;
            in_count++;
        }
        
        while (out_count > 0 && popped < nvec) {
            if (raw_mode) {
                for (int i = 0; i < N; i++) {
                    Y32[popped * N + i] = npu_reg_read(npu->base_addr + REG_Y_DATA);
                }
            } else {
                for (int w = 0; w < words_per_vec; w++) {
                    uint32_t word = npu_reg_read(npu->base_addr + REG_Y_DATA);
                    for (int b = 0; b < 4; b++) {
                        Y8[popped * N + w * 4 + b] = (word >> (b * 8)) & 0xFF;
                    }
                }
            }
            popped++;
            out_count--;
        }
    }
}

void npu_perf_read(npu_t *npu, uint32_t *cycles, uint32_t *active, uint32_t *vectors, uint32_t *zero_act, uint32_t *sleep_clk) {
    if (!npu->present) return;
    if (cycles) *cycles = npu_reg_read(npu->base_addr + REG_PERF_CYCLES);
    if (active) *active = npu_reg_read(npu->base_addr + REG_PERF_ACTIVE);
    if (vectors) *vectors = npu_reg_read(npu->base_addr + REG_PERF_VECTORS);
    if (zero_act) *zero_act = npu_reg_read(npu->base_addr + REG_PERF_ZERO_ACT);
    if (sleep_clk) *sleep_clk = npu_reg_read(npu->base_addr + REG_PERF_SLEEP);
}

void npu_sleep(npu_t *npu, int sleep) {
    if (!npu->present) return;
    uint32_t ctrl = npu_reg_read(npu->base_addr + REG_CTRL);
    if (sleep) ctrl |= (1<<3);
    else ctrl &= ~(1<<3);
    npu_reg_write(npu->base_addr + REG_CTRL, ctrl);
}

/* ------------------------------------------------------------------------------------------------------------------------
 * Contract v1.2 matmul (K-tile accumulator). For every chunk of <= acc_depth vectors and every C_out tile of N columns:
 *   wait BUSY=0 -> per-column M/S/bias + OUT_ZP -> for every K tile: wait W_READY, W rows N-1..0, W_COMMIT, ACC_PASS(first,last),
 *   push the chunk's K slice (polling IN_COUNT every 8 vectors); in the LAST pass pop the INT8 results while pushing.
 * With the X replay buffer (v1.3) the chunk is sized so that all its K slices fit in the buffer: the first C_out tile pushes and
 * records them (slice kt at kt*n), every other C_out tile replays them with one XR_REPLAY write per K tile.
 * ------------------------------------------------------------------------------------------------------------------------ */
static inline uint32_t pack4(const int8_t *p, int n)     /* n valid bytes (0..4), rest 0 */
{
    uint32_t w = 0;
    for (int b = 0; b < n; b++) w |= (uint32_t)(uint8_t)p[b] << (8 * b);
    return w;
}

static void pop_results(npu_t *npu, int n, int8_t *Y, int ldy, int v_first, int co0, int ncol)
{
    uintptr_t b = npu->base_addr;
    int NW = npu->N / 4;
    for (int i = 0; i < n; i++) {
        int8_t *y = Y + (size_t)(v_first + i) * ldy + co0;
        for (int w = 0; w < NW; w++) {
            uint32_t word = npu_reg_read(b + REG_Y_DATA);
            for (int k = 0; k < 4; k++) {
                int c = 4 * w + k;
                if (c < ncol) y[c] = (int8_t)(word >> (8 * k));
            }
        }
    }
}

int npu_matmul(npu_t *npu, const int8_t *X, int V, int K, int ldx, const int8_t *Wm, int Co,
               const int32_t *bias, const uint16_t *M, const uint8_t *S, int8_t zp, int relu, int8_t *Y, int ldy)
{
    if (!npu->present) return -1;
    uintptr_t b = npu->base_addr;
    const int N = npu->N, NW = N / 4;
    const int kt_n = (K + N - 1) / N, ct_n = (Co + N - 1) / N;
    const int acc = kt_n > 1;
    if (acc && npu->acc_depth == 0) return -1;
    int chunk = acc ? npu->acc_depth : V;
    int xr = npu->use_xr && npu->xr_depth >= kt_n && ct_n > 1;
    if (xr && chunk > npu->xr_depth / kt_n) chunk = npu->xr_depth / kt_n;

    wait_busy(npu);
    npu_reg_write(b + REG_CTRL, (1u << 1) | ((uint32_t)(relu != 0) << 2) | (1u << 4) | ((uint32_t)acc << 5) | ((uint32_t)xr << 6));
    npu_reg_write(b + REG_OUT_ZP, (uint8_t)zp);

    for (int v0 = 0; v0 < V; v0 += chunk) {
        int v1 = (v0 + chunk < V) ? v0 + chunk : V, n = v1 - v0;
        for (int ct = 0; ct < ct_n; ct++) {
            int co0 = ct * N, ncol = (Co - co0 < N) ? Co - co0 : N;
            wait_busy(npu);                                  /* postproc parameters only change when idle */
            npu_reg_write(b + REG_COL_IDX, 0);
            for (int c = 0; c < N; c++) {
                int on = c < ncol;
                npu_reg_write(b + REG_RQ_MULT, on ? M[co0 + c] : 1);
                npu_reg_write(b + REG_RQ_SHIFT, on ? S[co0 + c] : 0);
                npu_reg_write(b + REG_BIAS_DATA, on ? (uint32_t)bias[co0 + c] : 0);
            }
            for (int kt = 0; kt < kt_n; kt++) {
                int k0 = kt * N, nk = (K - k0 < N) ? K - k0 : N;
                int last = kt == kt_n - 1;
                wait_wready(npu);
                for (int r = N - 1; r >= 0; r--) {
                    const int8_t *wrow = Wm + (size_t)(k0 + r) * Co + co0;
                    for (int w = 0; w < NW; w++) {
                        int c = 4 * w;
                        int n = (r < nk) ? ((ncol - c >= 4) ? 4 : (ncol - c > 0 ? ncol - c : 0)) : 0;
                        npu_reg_write(b + REG_W_DATA, pack4(wrow + c, n));
                    }
                }
                if (xr && ct > 0) {                              /* replay the recorded K slice */
                    while (npu_reg_read(b + REG_STATUS) & ST_RP_ACTIVE) { }
                    npu_reg_write(b + REG_W_COMMIT, 1);
                    if (acc) npu_reg_write(b + REG_ACC_PASS, (uint32_t)(kt == 0) | ((uint32_t)last << 1));
                    npu_reg_write(b + REG_XR_REPLAY, ((uint32_t)n << 16) | (uint32_t)(kt * n));
                    for (int popped = v0; last && popped < v1; ) {
                        int avail = (int)((npu_reg_read(b + REG_STATUS) >> 8) & 0xFFu);
                        if (avail > v1 - popped) avail = v1 - popped;
                        pop_results(npu, avail, Y, ldy, popped, co0, ncol);
                        popped += avail;
                    }
                    continue;
                }
                npu_reg_write(b + REG_W_COMMIT, 1);
                if (acc) npu_reg_write(b + REG_ACC_PASS, (uint32_t)(kt == 0) | ((uint32_t)last << 1));
                if (xr) npu_reg_write(b + REG_XR_WPTR, (uint32_t)(kt * n));
                int popped = v0;
                for (int v = v0; v < v1; v++) {
                    if (((v - v0) & 7) == 0) {
                        while ((npu_reg_read(b + REG_STATUS) & 0xFFu) > IN_DEPTH - 8) { }
                    }
                    const int8_t *x = X + (size_t)v * ldx + k0;
                    for (int w = 0; w < NW; w++) {
                        int c = 4 * w;
                        int n = (nk - c >= 4) ? 4 : (nk - c > 0 ? nk - c : 0);
                        npu_reg_write(b + REG_X_DATA, pack4(x + c, n));
                    }
                    if (last && v + 1 - popped > 24) {         /* keep the OUT FIFO draining */
                        uint32_t st = npu_reg_read(b + REG_STATUS);
                        int avail = (int)((st >> 8) & 0xFFu);
                        pop_results(npu, avail, Y, ldy, popped, co0, ncol);
                        popped += avail;
                    }
                }
                while (last && popped < v1) {
                    uint32_t st = npu_reg_read(b + REG_STATUS);
                    int avail = (int)((st >> 8) & 0xFFu);
                    if (avail > v1 - popped) avail = v1 - popped;
                    pop_results(npu, avail, Y, ldy, popped, co0, ncol);
                    popped += avail;
                }
            }
        }
    }
    wait_busy(npu);
    npu_reg_write(b + REG_CTRL, 1u << 4);              /* leave the accumulator off for the v1.1-style callers (canny_npu) */
    uint32_t st = npu_reg_read(b + REG_STATUS);
    if (st & ST_ERRS) { npu_reg_write(b + REG_ERR_CLR, st & ST_ERRS); return -1; }
    return 0;
}
