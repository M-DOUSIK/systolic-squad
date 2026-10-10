/*
 * npu_sim.c - software model of the NPU register map for host tests (-DHOST_TEST).
 * Behaviour follows the register map exactly:
 *   - shadow / active weights: W_DATA fills the shadow tile, the first vector pushed after W_COMMIT activates it
 *   - STATUS bits (IN_COUNT, OUT_COUNT, BUSY, W_READY, IN_FULL, OUT_EMPTY, sticky ERR_W/ERR_OVF/ERR_UNF), ERR_CLR, SCRATCH
 *   - empty Y_DATA read returns 0xDEADBEEF + ERR_UNF; IN FIFO 32 vectors, OUT FIFO 64 vectors; push on full IN FIFO is dropped
 *   - CFG word = N | ACC_W<<8 | PE_TYPE<<16 | ZERO_SKIP<<17 | engine<<18 | version<<24 (BIG: PE_TYPE 0, LITTLE: 1)
 *   - PERF_ZERO_ACT counts zero activation elements; PERF_CYCLES/ACTIVE are an approximation (not cycle-accurate)
 * The sim computes a vector the moment it is pushed (while the OUT FIFO has room), which is what the driver sees.
 * K-tile accumulator (CTRL.ACC_EN, ACC_PASS first/last + slot counter, ACC_INFO), CFG version 2,
 * W_READY = 0 while a W_COMMIT is pending; only LAST vectors produce output.
 * X replay buffer (v1.3, CFG version 3): CTRL.XR_REC records pushed X vectors at XR_WPTR++, XR_REPLAY {count<<16 | start} feeds them
 * back into the IN FIFO as room frees up, STATUS bit 20 = replay active, XR_INFO = depth.
 */
#include "npu.h"
#include "platform.h"
#include "qmath.h"
#include <string.h>

#ifdef HOST_TEST

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
#define REG_SCRATCH     0x044
#define REG_LEDS        0x048
#define REG_OUT_ZP      0x04C
#define REG_PERF_ZERO_ACT 0x050
#define REG_PERF_SLEEP  0x054
#define REG_ACC_PASS    0x058
#define REG_ACC_INFO    0x05C
#define ACC_DEPTH       1024
#define REG_XR_WPTR     0x060
#define REG_XR_REPLAY   0x064
#define REG_XR_INFO     0x068
#define XR_DEPTH        8192

#define IN_DEPTH  32
#define OUT_DEPTH 64
#define MAXN      16

typedef struct {
    uint32_t cfg;
    uint32_t ctrl, scratch, leds;
    int      N;
    int8_t   out_zp;
    uint32_t col_idx;
    uint16_t rq_mult[MAXN];
    uint8_t  rq_shift[MAXN];
    int32_t  bias[MAXN];

    int8_t   w_sh[MAXN][MAXN];       /* shadow tile, filled by W_DATA */
    int8_t   w_act[MAXN][MAXN];      /* active tile used by vectors */
    int      w_row, w_word;          /* next row to fill (counts down N-1..0), words received for that row */
    int      newW_pending;

    int8_t   in_fifo[IN_DEPTH][MAXN];
    int      in_head, in_tail, in_count, in_word;
    int8_t   x_asm[MAXN];
    int      x_newW[IN_DEPTH];
    uint8_t  x_first[IN_DEPTH], x_last[IN_DEPTH];
    uint16_t x_slot[IN_DEPTH];
    int      pass_first, pass_last, slot_ctr;
    int32_t  acc[ACC_DEPTH][MAXN];
    int8_t   xr[XR_DEPTH][MAXN];
    int      xr_wptr, rp_addr, rp_left;

    int32_t  out_raw[OUT_DEPTH][MAXN];
    int8_t   out_q[OUT_DEPTH][MAXN];
    int      out_head, out_tail, out_count, out_word;

    uint32_t errors;                 /* bits 24..26 as in STATUS */
    uint32_t perf_cycles, perf_active, perf_vectors, perf_zero_act, perf_sleep;
    int      perf_en;
} sim_t;

static sim_t sims[2];

static void sim_init(int idx, uint8_t n, uint8_t pe, uint8_t zs, uint8_t engine)
{
    memset(&sims[idx], 0, sizeof(sim_t));
    sims[idx].N = n;
    sims[idx].cfg = (uint32_t)n | (32u << 8) | ((uint32_t)pe << 16) | ((uint32_t)zs << 17) | ((uint32_t)engine << 18) | (3u << 24);
    sims[idx].w_row = n - 1;
    sims[idx].pass_first = sims[idx].pass_last = 1;
    sims[idx].ctrl = 1u << 4;        /* ZERO_SKIP_EN default 1 */
    for (int c = 0; c < MAXN; c++) sims[idx].rq_mult[c] = 1;
}

void npu_sim_setup(void)
{
    sim_init(0, 16, 0, 1, 0);        /* BIG    16x16 DSP   */
    sim_init(1, 4,  1, 1, 1);        /* small 4x4 engine (tests only) */
}

static sim_t *get_sim(uintptr_t addr) { return (addr & 0x1000) ? &sims[1] : &sims[0]; }

static void pump(sim_t *s);

/* one X vector into the IN FIFO with the current tags (APB push or replay) */
static void push_vec(sim_t *s, const int8_t *x)
{
    int N = s->N;
    if (s->in_count == IN_DEPTH) { s->errors |= (1u << 25); return; }     /* ERR_OVF, vector dropped */
    memcpy(s->in_fifo[s->in_head], x, (size_t)N);
    s->x_newW[s->in_head] = s->newW_pending;
    s->newW_pending = 0;
    int acc_en = (int)((s->ctrl >> 5) & 1u);
    s->x_first[s->in_head] = (uint8_t)(acc_en ? s->pass_first : 1);
    s->x_last[s->in_head]  = (uint8_t)(acc_en ? s->pass_last : 1);
    s->x_slot[s->in_head]  = (uint16_t)((acc_en && s->slot_ctr < ACC_DEPTH) ? s->slot_ctr : 0);
    if (acc_en) { if (s->slot_ctr < ACC_DEPTH) s->slot_ctr++; else s->errors |= (1u << 27); }
    s->in_head = (s->in_head + 1) % IN_DEPTH;
    s->in_count++;
}

/* run the array, and feed replayed vectors while the IN FIFO has room */
static void progress(sim_t *s)
{
    for (;;) {
        pump(s);
        if (s->rp_left > 0 && s->in_count < IN_DEPTH) {
            push_vec(s, s->xr[s->rp_addr % XR_DEPTH]);
            s->rp_addr++; s->rp_left--;
        } else break;
    }
}

static void pump(sim_t *s)
{
    while (s->in_count > 0 && s->out_count < OUT_DEPTH) {
        int N = s->N, t = s->in_tail;
        const int8_t *x = s->in_fifo[t];
        int relu = (s->ctrl >> 2) & 1;
        if (s->x_newW[t]) memcpy(s->w_act, s->w_sh, sizeof(s->w_act));
        for (int r = 0; r < N; r++) if (x[r] == 0) s->perf_zero_act++;
        int32_t *a = s->acc[s->x_slot[t]];
        for (int c = 0; c < N; c++) {
            int32_t y = 0;
            for (int r = 0; r < N; r++) y += (int32_t)x[r] * (int32_t)s->w_act[r][c];
            a[c] = s->x_first[t] ? y : (int32_t)((uint32_t)a[c] + (uint32_t)y);
        }
        if (s->x_last[t]) {
            for (int c = 0; c < N; c++) {
                s->out_raw[s->out_head][c] = a[c];
                s->out_q[s->out_head][c] = qmath_rq(a[c], s->bias[c], s->rq_mult[c], s->rq_shift[c], s->out_zp, relu);
            }
            s->out_head = (s->out_head + 1) % OUT_DEPTH;
            s->out_count++;
        }
        s->in_tail = (s->in_tail + 1) % IN_DEPTH;
        s->in_count--;
        s->perf_vectors++;
        s->perf_active += 1;
        s->perf_cycles += 1;
    }
}

uint32_t npu_reg_read(uintptr_t addr)
{
    sim_t *s = get_sim(addr);
    uint32_t off = addr & 0xFFF;
    s->perf_cycles++;
    switch (off) {
    case REG_ID:     return 0x53514431u;
    case REG_CFG:    return s->cfg;
    case REG_CTRL:   return s->ctrl;
    case REG_STATUS:
        return (uint32_t)(s->in_count & 0xFF) | ((uint32_t)(s->out_count & 0xFF) << 8) | ((s->in_count > 0 || s->rp_left > 0) << 16) |
               ((uint32_t)(s->rp_left > 0) << 20) |
               ((uint32_t)!s->newW_pending << 17) | ((s->in_count == IN_DEPTH) << 18) | ((s->out_count == 0) << 19) | s->errors;
    case REG_Y_DATA: {
        int N = s->N, rq = (s->ctrl >> 1) & 1;
        uint32_t val = 0;
        if (s->out_count == 0) { s->errors |= (1u << 26); return 0xDEADBEEFu; }
        if (rq) {
            for (int b = 0; b < 4; b++) val |= (uint32_t)(uint8_t)s->out_q[s->out_tail][s->out_word * 4 + b] << (8 * b);
            if (++s->out_word == N / 4) { s->out_word = 0; s->out_tail = (s->out_tail + 1) % OUT_DEPTH; s->out_count--; progress(s); }
        } else {
            val = (uint32_t)s->out_raw[s->out_tail][s->out_word];
            if (++s->out_word == N) { s->out_word = 0; s->out_tail = (s->out_tail + 1) % OUT_DEPTH; s->out_count--; progress(s); }
        }
        return val;
    }
    case REG_RQ_MULT:  return s->rq_mult[s->col_idx % MAXN];
    case REG_RQ_SHIFT: return s->rq_shift[s->col_idx % MAXN];
    case REG_COL_IDX:  return s->col_idx;
    case REG_BIAS_DATA: return (uint32_t)s->bias[s->col_idx % MAXN];
    case REG_PERF_CYCLES:   return s->perf_cycles;
    case REG_PERF_ACTIVE:   return s->perf_active;
    case REG_PERF_VECTORS:  return s->perf_vectors;
    case REG_SCRATCH:       return s->scratch;
    case REG_LEDS:          return s->leds;
    case REG_OUT_ZP:        return (uint8_t)s->out_zp;
    case REG_PERF_ZERO_ACT: return s->perf_zero_act;
    case REG_PERF_SLEEP:    return s->perf_sleep;
    case REG_ACC_PASS:      return (uint32_t)s->pass_first | ((uint32_t)s->pass_last << 1);
    case REG_ACC_INFO:      return ACC_DEPTH;
    case REG_XR_WPTR:       return (uint32_t)s->xr_wptr;
    case REG_XR_REPLAY:     return (uint32_t)s->rp_left;
    case REG_XR_INFO:       return XR_DEPTH;
    }
    return 0;
}

void npu_reg_write(uintptr_t addr, uint32_t val)
{
    sim_t *s = get_sim(addr);
    uint32_t off = addr & 0xFFF;
    int N = s->N;
    switch (off) {
    case REG_CTRL:
        if (val & 1u) {              /* SOFT_RST: flush FIFOs, counters kept; bit self-clears */
            s->in_head = s->in_tail = s->in_count = s->in_word = 0;
            s->out_head = s->out_tail = s->out_count = s->out_word = 0;
            s->w_row = N - 1; s->w_word = 0; s->newW_pending = 0; s->slot_ctr = 0; s->pass_first = s->pass_last = 1;
            val &= ~1u;
        }
        s->ctrl = val;
        break;
    case REG_W_DATA:
        if (s->newW_pending) { s->errors |= (1u << 24); break; }              /* W_READY = 0: dropped, ERR_W */
        for (int b = 0; b < 4; b++) s->w_sh[s->w_row][s->w_word * 4 + b] = (int8_t)((val >> (8 * b)) & 0xFF);
        if (++s->w_word == N / 4) {
            s->w_word = 0;
            s->w_row = (s->w_row == 0) ? N - 1 : s->w_row - 1;     /* rows arrive N-1 ... 0, then wrap for the next tile */
        }
        break;
    case REG_W_COMMIT:
        s->newW_pending = 1;
        break;
    case REG_X_DATA:
        for (int b = 0; b < 4; b++) s->x_asm[s->in_word * 4 + b] = (int8_t)((val >> (8 * b)) & 0xFF);
        if (++s->in_word == N / 4) {
            s->in_word = 0;
            if (s->rp_left > 0) { s->errors |= (1u << 25); break; }              /* X_DATA during a replay: dropped */
            if (s->in_count < IN_DEPTH && ((s->ctrl >> 6) & 1u) && s->xr_wptr < XR_DEPTH)
                memcpy(s->xr[s->xr_wptr++], s->x_asm, (size_t)N);            /* XR_REC */
            push_vec(s, s->x_asm);
            pump(s);
        }
        break;
    case REG_RQ_MULT:   s->rq_mult[s->col_idx % MAXN] = (uint16_t)(val & 0xFFFF); break;
    case REG_RQ_SHIFT:  s->rq_shift[s->col_idx % MAXN] = (uint8_t)(val & 0x1F); break;
    case REG_COL_IDX:   s->col_idx = val & 0xFF; break;
    case REG_BIAS_DATA: s->bias[s->col_idx % MAXN] = (int32_t)val; s->col_idx++; break;
    case REG_PERF_CTRL:
        if (val & 1u) { s->perf_cycles = s->perf_active = s->perf_vectors = s->perf_zero_act = s->perf_sleep = 0; }
        s->perf_en = (int)((val >> 1) & 1u);
        break;
    case REG_ERR_CLR:   s->errors &= ~(val & (0xFu << 24)); break;
    case REG_ACC_PASS:  s->pass_first = (int)(val & 1u); s->pass_last = (int)((val >> 1) & 1u); s->slot_ctr = 0; break;
    case REG_SCRATCH:   s->scratch = val; break;
    case REG_LEDS:      s->leds = val & 0xFF; break;
    case REG_OUT_ZP:    s->out_zp = (int8_t)(val & 0xFF); break;
    case REG_XR_WPTR:   s->xr_wptr = (int)(val & 0xFFFF); break;
    case REG_XR_REPLAY: s->rp_addr = (int)(val & 0xFFFF); s->rp_left = (int)(val >> 16); progress(s); break;
    }
}

#endif /* HOST_TEST */
