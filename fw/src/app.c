/* app.c - see app.h. */
#include "app.h"
#include "depth.h"
#include "obstacle.h"
#include "canny.h"
#include "proto.h"
#include "platform.h"
#include "qmath.h"
#include <string.h>

#define FW_W PREPROC_W
#define FW_H PREPROC_H
#define NPIX (FW_W * FW_H)
#define ALIGN16(x) (((x) + 15u) & ~(size_t)15u)

typedef struct {                   /* one DDR region per model: weights stay loaded when the model is switched */
    uint8_t *blob;
    uint32_t cap, bytes, crc;
    int      ok;
} slot_t;

static struct {
    npu_t   *npu;
    slot_t   slot[N_MODELS];
    int      cur;                  /* active model (index into MODELS) */
    uint8_t *work;                 /* activations of the active model */
    depth_net_t net;
    obstacle_agent_t agent;
    /* buffers in work memory */
    uint8_t *rgb_last, *gray, *depth_bytes;
    int8_t  *edge, *edge_prev, *blur;
    int32_t *gx, *gy;
    /* settings */
    int gate_on, send_depth, use_npu, edge_next, have_prev, have_frame;
    uint8_t t_m;
    /* counters */
    uint32_t frame_id, frames_seen, frames_gated, cnn_runs, t_boot;
    uint8_t last_n[3];
} A;

/* ---------------------------------------------------------------------------------------------- CRC-32/IEEE */
static uint32_t crc_table[256];
static void crc_init(void)
{
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crc_table[i] = c;
    }
}
static uint32_t crc32(const uint8_t *p, uint32_t n)
{
    uint32_t c = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < n; i++) c = crc_table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

typedef struct { char b[160]; int k; } line_t;
static void ls_(line_t *l, const char *s);

static size_t net_bytes(void)
{
    size_t m = 0;
    for (int i = 0; i < N_MODELS; i++) if (depth_mem_bytes(MODELS[i]) > m) m = depth_mem_bytes(MODELS[i]);
    return ALIGN16(m);
}

size_t app_work_bytes(void)
{
    return net_bytes() + ALIGN16(NPIX * 3) + 3 * ALIGN16(NPIX) + 3 * ALIGN16(NPIX) + 2 * ALIGN16(NPIX * 4);
}

uint32_t app_blob_bytes(void)
{
    uint32_t n = 0;
    for (int i = 0; i < N_MODELS; i++) n += (MODELS[i]->blob_bytes + 0xFFFFFu) & ~0xFFFFFu;     /* 1 MB aligned slots */
    return n;
}

static void select_model(int i)
{
    A.cur = i;
    depth_init(&A.net, MODELS[i], A.slot[i].blob, A.work, A.npu);
    A.have_prev = 0;               /* next frame is never skipped by the gate */
    line_t l = { {0}, 0 };
    ls_(&l, "model: "); ls_(&l, MODELS[i]->name); ls_(&l, A.slot[i].ok ? " (weights loaded)" : " (weights not loaded yet)");
    console_log(l.b);
}

void app_init(npu_t *npu, uint8_t *blob_buf, uint32_t blob_cap, uint8_t *mem)
{
    memset(&A, 0, sizeof(A));
    crc_init();
    A.npu = npu;
    uint32_t off = 0;
    for (int i = 0; i < N_MODELS; i++) {
        uint32_t want = (MODELS[i]->blob_bytes + 0xFFFFFu) & ~0xFFFFFu;
        A.slot[i].blob = blob_buf + off;
        A.slot[i].cap = (off >= blob_cap) ? 0 : (off + want <= blob_cap ? want : blob_cap - off);
        off += want;
    }
    A.work = mem;
    depth_init(&A.net, MODELS[0], A.slot[0].blob, mem, npu);
    mem += net_bytes();
    A.rgb_last = mem;              mem += ALIGN16(NPIX * 3);
    A.gray = mem;                  mem += ALIGN16(NPIX);
    A.depth_bytes = mem;           mem += ALIGN16(NPIX);
    A.edge = (int8_t *)mem;        mem += ALIGN16(NPIX);
    A.edge_prev = (int8_t *)mem;   mem += ALIGN16(NPIX);
    A.blur = (int8_t *)mem;        mem += ALIGN16(NPIX);
    mem += ALIGN16(NPIX);
    A.gx = (int32_t *)(void *)mem; mem += ALIGN16(NPIX * 4);
    A.gy = (int32_t *)(void *)mem;
    A.gate_on = 1;
    A.send_depth = 1;
    A.use_npu = npu && npu->present;
    A.t_m = 2;                     /* per mille of edge pixels that changed; below this the frame is "static" and skipped */
    obstacle_init(&A.agent, OBS_T_WARN_DEFAULT, OBS_T_NEAR_DEFAULT);
    A.t_boot = time_us();
}

static void log_text(const char *a, uint32_t v)
{
    char s[96];
    int k = 0;
    while (*a && k < 70) s[k++] = *a++;
    char d[12]; int n = 0;
    if (v == 0) d[n++] = '0';
    while (v && n < 11) { d[n++] = (char)('0' + v % 10); v /= 10; }
    while (n) s[k++] = d[--n];
    s[k] = 0;
    proto_build_log(s);
}

/* ---------------------------------------------------------------------------------------------- human-readable console
 * One text line per event on the board's second UART (console_log, platform layer), so a plain serial monitor shows the board working. */
static void ls_(line_t *l, const char *s) { while (*s && l->k < 157) l->b[l->k++] = *s++; l->b[l->k] = 0; }
static void lu_(line_t *l, uint32_t v)
{
    char d[12]; int n = 0;
    if (v == 0) d[n++] = '0';
    while (v && n < 11) { d[n++] = (char)('0' + v % 10); v /= 10; }
    while (n && l->k < 157) l->b[l->k++] = d[--n];
    l->b[l->k] = 0;
}

static void console_frame(const result_pkt_t *r, int gated, uint32_t c, uint32_t a)
{
    line_t l = { {0}, 0 };
    ls_(&l, "frame "); lu_(&l, r->frame_id);
    if (gated) { ls_(&l, " | scene static, depth skipped (Canny gate)"); console_log(l.b); return; }
    ls_(&l, A.use_npu ? " | NPU 16x16 | depth net " : " | CPU only | depth net "); lu_(&l, r->t_cnn_us / 1000u); ls_(&l, " ms");
    if (A.use_npu) {
        ls_(&l, " (array layers "); lu_(&l, A.net.t_npu_us / 1000u); ls_(&l, " ms, CPU layers "); lu_(&l, A.net.t_cpu_us / 1000u);
        ls_(&l, " ms) | array busy "); lu_(&l, c ? (uint32_t)((uint64_t)a * 100u / c) : 0); ls_(&l, " % of "); lu_(&l, c); ls_(&l, " clk");
    }
    ls_(&l, " | NPU errors "); lu_(&l, (uint32_t)A.net.npu_errors); ls_(&l, " | frame total "); lu_(&l, r->t_total_us / 1000u); ls_(&l, " ms");
    console_log(l.b);
}

/* ---------------------------------------------------------------------------------------------- one frame */
static void run_frame(const uint8_t *rgb)
{
    uint32_t t0 = time_us();
    A.frames_seen++;
    A.frame_id++;
    memcpy(A.rgb_last, rgb, NPIX * 3);
    A.have_frame = 1;

    /* SENSE: gray -> Canny (blur + Sobel on the NPU when present) -> change score */
    for (int i = 0; i < NPIX; i++) {
        const uint8_t *p = rgb + 3 * i;
        A.gray[i] = (uint8_t)((77 * p[0] + 150 * p[1] + 29 * p[2] + 128) >> 8);
    }
    if (A.use_npu) canny_npu(A.npu, A.gray, FW_H, FW_W, A.gx, A.gy, A.blur, A.edge, GATE_CANNY_LOW, GATE_CANNY_HIGH);
    else canny_cpu(A.gray, FW_H, FW_W, A.gx, A.gy, A.blur, A.edge, GATE_CANNY_LOW, GATE_CANNY_HIGH);
    uint32_t changed = 0;
    for (int i = 0; i < NPIX; i++) changed += (A.edge[i] != A.edge_prev[i]);
    uint16_t motion = A.have_prev ? (uint16_t)(1000u * changed / NPIX) : 1000;
    uint16_t edge_score = canny_edge_score(A.edge, FW_H, FW_W);
    memcpy(A.edge_prev, A.edge, NPIX);
    A.have_prev = 1;

    result_pkt_t r;
    memset(&r, 0, sizeof(r));
    r.frame_id = A.frame_id;
    r.n_classes = 3;
    uint32_t c0 = 0, a0 = 0, c1 = 0, a1 = 0;
    if (A.npu->present) npu_perf_read(A.npu, &c0, &a0, 0, 0, 0);

    int gated = A.gate_on && motion <= A.t_m;
    uint32_t t_cnn = 0;
    if (!gated) {
        uint32_t tc = time_us();
        A.net.use_npu = A.use_npu;
        depth_run(&A.net, rgb);
        t_cnn = time_us() - tc;
        A.cnn_runs++;
        const int8_t *q = depth_output(&A.net);
        for (int i = 0; i < NPIX; i++) A.depth_bytes[i] = (uint8_t)(q[i] + 128);
        zone_nearness(A.depth_bytes, FW_H, FW_W, A.net.m->out_kind, A.last_n);
        r.flags = A.use_npu ? 2 : 4;
    } else {
        A.frames_gated++;
        r.flags = 1;
    }
    obstacle_update(&A.agent, A.last_n);
    /* LEDs: plain status, not the obstacle guidance: b0 toggles every frame, b1 NPU used, b2 frame skipped by the gate, b7 app running */
    leds_set((uint8_t)(0x80u | (A.frame_id & 1u) | ((uint32_t)(A.use_npu && !gated) << 1) | ((uint32_t)gated << 2)));
    if (A.npu->present) npu_perf_read(A.npu, &c1, &a1, 0, 0, 0);

    r.g_raw = A.agent.raw;
    r.g_stable = A.agent.stable;
    r.command = A.agent.leds;
    for (int z = 0; z < 3; z++) r.logits[z] = A.last_n[z];
    r.logits[3] = A.agent.levels[0] | (A.agent.levels[1] << 2) | (A.agent.levels[2] << 4);
    r.logits[4] = (int32_t)A.net.t_npu_us;            /* v1.2 extra info: NPU-layer time, CPU-layer time of the depth run */
    r.logits[5] = (int32_t)A.net.t_cpu_us;
    r.logits[6] = motion;
    r.logits[7] = A.net.npu_errors;
    r.t_cnn_us = t_cnn;
    r.npu_cycles = c1 - c0;
    r.npu_active = a1 - a0;
    r.t_total_us = time_us() - t0;

    if (!gated && A.send_depth) proto_build_depth_map(FW_W, FW_H, A.depth_bytes);
    if (A.edge_next) { proto_build_edge_map(FW_W, FW_H, A.edge); A.edge_next = 0; }
    proto_build_result(&r);
    console_frame(&r, gated, c1 - c0, a1 - a0);

    telemetry_pkt_t t;
    memset(&t, 0, sizeof(t));
    t.mode = 1;
    t.reason = (uint8_t)gated;
    t.fps_x10 = r.t_total_us ? (uint16_t)(10000000u / r.t_total_us) : 0;
    t.frames_seen = A.frames_seen;
    t.frames_gated = A.frames_gated;
    t.cnn_runs = A.cnn_runs;
    t.uptime_ms = (time_us() - A.t_boot) / 1000u;
    t.motion_score = motion;
    t.edge_score = edge_score;
    t.engines_awake = (uint8_t)(A.npu->present ? 1 : 0);
    t.gate_engine = (uint8_t)(A.use_npu ? 0 : 2);
    proto_build_telemetry(&t);
}

/* ---------------------------------------------------------------------------------------------- self-test / bench */
static void selftest(void)
{
    /* random matmul with K and C_out tails, compared with the CPU (exercises the K-tile accumulator) */
    static int8_t X[50 * 70], Wm[70 * 20], Yn[50 * 20], Yc[50 * 20];
    static int32_t B[20]; static uint16_t M[20]; static uint8_t S[20];
    uint32_t s = 12345u;
    for (int i = 0; i < 50 * 70; i++) { s = s * 1103515245u + 12345u; X[i] = (int8_t)(s >> 16); }
    for (int i = 0; i < 70 * 20; i++) { s = s * 1103515245u + 12345u; Wm[i] = (int8_t)(s >> 16); }
    for (int c = 0; c < 20; c++) { s = s * 1103515245u + 12345u; B[c] = (int32_t)(s >> 12) - 500000; M[c] = (uint16_t)(1000 + c * 977); S[c] = (uint8_t)(14 + c % 6); }
    int fail = 0, n = 0;
    for (int v = 0; v < 50; v++)
        for (int c = 0; c < 20; c++) {
            int32_t acc = 0;
            for (int k = 0; k < 70; k++) acc += X[v * 70 + k] * Wm[k * 20 + c];
            Yc[v * 20 + c] = qmath_rq(acc, B[c], M[c], S[c], -3, 0);
        }
    if (A.npu->present && npu_matmul(A.npu, X, 50, 70, 70, Wm, 20, B, M, S, -3, 0, Yn, 20) == 0) {
        for (int i = 0; i < 50 * 20; i++) { n++; fail += Yn[i] != Yc[i]; }
    } else { n = 1; fail = 1; }
    proto_build_selftest((uint8_t)(fail == 0), (uint16_t)n, (uint16_t)fail);
    line_t l = { {0}, 0 };
    ls_(&l, fail ? "SELFTEST FAIL: " : "SELFTEST PASS: "); lu_(&l, (uint32_t)(n - fail)); ls_(&l, "/"); lu_(&l, (uint32_t)n);
    ls_(&l, " NPU results (K=70 -> 5 K-tiles accumulated on chip) equal the CPU");
    console_log(l.b);
}

static void bench(void)
{
    if (!A.have_frame || !A.slot[A.cur].ok) { proto_build_log("BENCH needs weights and one frame first"); return; }
    static uint8_t ref[NPIX];
    uint32_t t = time_us();
    A.net.use_npu = 0;
    depth_run(&A.net, A.rgb_last);
    uint32_t cpu_us = time_us() - t;
    const int8_t *q = depth_output(&A.net);
    for (int i = 0; i < NPIX; i++) ref[i] = (uint8_t)(q[i] + 128);
    uint32_t npu_us = 0;
    int match = 0;
    if (A.npu->present) {
        t = time_us();
        A.net.use_npu = 1;
        depth_run(&A.net, A.rgb_last);
        npu_us = time_us() - t;
        q = depth_output(&A.net);
        match = 1;
        for (int i = 0; i < NPIX; i++) if ((uint8_t)(q[i] + 128) != ref[i]) { match = 0; break; }
    }
    A.net.use_npu = A.use_npu;
    proto_build_bench(cpu_us, npu_us, depth_macs(A.net.m), (uint8_t)match);
    line_t l = { {0}, 0 };
    ls_(&l, "BENCH same frame: CPU only "); lu_(&l, cpu_us / 1000u); ls_(&l, " ms, NPU "); lu_(&l, npu_us / 1000u);
    ls_(&l, " ms, speed-up x"); lu_(&l, npu_us ? cpu_us / npu_us : 0); ls_(&l, "."); lu_(&l, npu_us ? (cpu_us * 10u / npu_us) % 10u : 0);
    ls_(&l, match ? ", outputs bit-identical" : ", OUTPUTS DIFFER");
    console_log(l.b);
}

/* ---------------------------------------------------------------------------------------------- dispatch */
void app_handle_packet(uint8_t type, const uint8_t *p, uint16_t len)
{
    if (type == PROTO_TYPE_WEIGHTS && len >= 4) {
        uint32_t off = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
        uint32_t n = len - 4u;
        slot_t *s = &A.slot[A.cur];
        if (off + n <= s->cap) {
            memcpy(s->blob + off, p + 4, n);
            if (off + n > s->bytes) s->bytes = off + n;
        }
        if (off == 0) s->ok = 0;
    } else if (type == PROTO_TYPE_FRAME && len >= 5) {
        uint16_t W = (uint16_t)(p[0] | (p[1] << 8)), H = (uint16_t)(p[2] | (p[3] << 8));
        uint8_t C = p[4];
        if (W != FW_W || H != FW_H || C != 3 || len != 5u + (uint32_t)W * H * C) log_text("FRAME refused, expected 3-channel frame of width ", FW_W);
        else if (!A.slot[A.cur].ok) proto_build_log("FRAME refused: no weights (send WEIGHTS + WEIGHTS_DONE)");
        else run_frame(p + 5);
    } else if (type == PROTO_TYPE_CMD && len >= 1) {
        uint8_t a1 = len > 1 ? p[1] : 0, a2 = len > 2 ? p[2] : 0;
        switch (p[0]) {
        case CMD_PING:       proto_build_pong(FW_VERSION, A.npu->present ? 0x53514431u : 0); break;
        case CMD_SELFTEST:   selftest(); break;
        case CMD_BENCH:      bench(); break;
        case CMD_SET_GATE:   A.gate_on = a1 != 0; break;
        case CMD_EDGE_NEXT:  A.edge_next = 1; break;
        case CMD_SEND_DEPTH: A.send_depth = a1 != 0; break;
        case CMD_USE_NPU:    A.use_npu = (a1 != 0) && A.npu->present; break;
        case CMD_SET_THRESH:
            if (a1 == 0) A.agent.t_warn = a2;
            else if (a1 == 1) A.agent.t_near = a2;
            else if (a1 == 2) A.t_m = a2;
            break;
        case CMD_WEIGHTS_DONE: {
            slot_t *s = &A.slot[A.cur];
            s->crc = crc32(s->blob, s->bytes);
            s->ok = (s->bytes == A.net.m->blob_bytes) && (s->crc == A.net.m->blob_crc32);
            proto_build_weights_ack(s->bytes, s->crc, (uint8_t)s->ok);
            break;
        }
        case CMD_PROFILE: {          /* one LOG line per layer of the last run: 'L<i> t<type> <us>' */
            for (int i = 0; i < A.net.m->n_layers; i++) {
                line_t l = { {0}, 0 };
                ls_(&l, "L"); lu_(&l, (uint32_t)i); ls_(&l, " t"); lu_(&l, A.net.m->layers[i].type); ls_(&l, " "); lu_(&l, A.net.t_layer_us[i]);
                proto_build_log(l.b);
            }
            break;
        }
        case CMD_SET_MODEL:        /* answer: WEIGHTS_ACK of that model's slot (ok = 0: the host must stream its weights) */
            if (a1 < N_MODELS) select_model(a1);
            proto_build_weights_ack(A.slot[A.cur].bytes, A.slot[A.cur].crc, (uint8_t)A.slot[A.cur].ok);
            break;
        default: break;
        }
    }
}

void app_poll(void)
{
    uint8_t b;
    while (uart_read_byte(&b, 0)) {
        if (proto_rx_feed(b) == PROTO_COMPLETE) app_handle_packet(proto_get_rx_type(), proto_get_rx_payload(), proto_get_rx_len());
    }
}
