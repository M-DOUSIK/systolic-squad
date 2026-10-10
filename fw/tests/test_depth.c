/* test_depth.c - the depth application against the golden vectors of ml/make_vectors.py:
 *   1. model_blob.bin size + CRC = MODELS[TEST_MODEL] (export and generated table agree)
 *   2. every layer of every golden frame, CPU path and NPU path (npu_sim, contract v1.2 K-tile accumulator) == ml/qmodel.py
 *   3. zone nearness + obstacle agent over the frame sequence == reference/depth_ref.py
 *   4. the packet flow through app.c: WEIGHTS + WEIGHTS_DONE -> WEIGHTS_ACK ok, FRAME -> DEPTH_MAP == golden depth bytes
 * Files are read at run time from ../ml/export (run `make test` from fw/; another export: make test EXPORT=../ml/export_midas). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "depth.h"
#include "obstacle.h"
#include "app.h"
#include "proto.h"
#include "test_harness.h"

#ifndef EXPORT_DIR
#define EXPORT_DIR "../ml/export"
#endif
#define VEC EXPORT_DIR "/vectors/"
#ifndef TEST_MODEL
#define TEST_MODEL MODEL_FASTDEPTH
#endif
#define TM (MODELS[TEST_MODEL])

static uint8_t *read_file(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *p = malloc((size_t)sz + 16);
    *n = fread(p, 1, (size_t)sz, f);
    fclose(f);
    return p;
}

static uint32_t crc32_ref(const uint8_t *p, size_t n)
{
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) { c ^= p[i]; for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1; }
    return c ^ 0xFFFFFFFFu;
}

static int check_layers(depth_net_t *d, int fr, const char *path_name)
{
    int bad = 0;
    char p[256];
    for (int i = 0; i < TM->n_layers; i++) {
        size_t n;
        snprintf(p, sizeof p, VEC "frame_%03d_L%d.bin", fr, i);
        uint8_t *g = read_file(p, &n);
        const layer_desc_t *l = &TM->layers[i];
        size_t want = (size_t)l->out_h * l->out_w * l->cout;
        if (!g || n != want || memcmp(g, d->act[i], n) != 0) {
            if (bad < 3) printf("  %s frame %d layer %d mismatch (%s)\n", path_name, fr, i, g ? "data" : "missing file");
            bad++;
        }
        free(g);
    }
    return bad;
}

/* find the first packet of `type` in the harness TX stream; returns payload pointer into buf */
static const uint8_t *find_pkt(const uint8_t *buf, uint32_t n, uint8_t type, uint16_t *len)
{
    for (uint32_t i = 0; i + 6 <= n; i++) {
        if (buf[i] == 0xA5 && buf[i + 1] == 0x5A && buf[i + 2] == type) {
            *len = (uint16_t)(buf[i + 3] | (buf[i + 4] << 8));
            if (i + 6u + *len <= n) return buf + i + 5;
        }
    }
    return NULL;
}

static uint8_t txcap[1 << 20];
static uint32_t txn;
static void drain(void)
{
    uint16_t n;
    static uint8_t tmp[65536];
    harness_drain_tx(tmp, &n);
    memcpy(txcap + txn, tmp, n);
    txn += n;
}

int test_depth(void)
{
    int fail = 0;
    size_t nb;
    uint8_t *blob = read_file(EXPORT_DIR "/model_blob.bin", &nb);
    if (!blob) { printf("test_depth: " EXPORT_DIR "/model_blob.bin missing (run ml/convert.py + ml/export_c.py)\n"); return 1; }
    uint32_t crc = crc32_ref(blob, nb);
    if (nb != TM->blob_bytes || crc != TM->blob_crc32) { printf("test_depth: blob size/CRC differ from the model table\n"); free(blob); return 1; }
    printf("test_depth: %s: blob %zu B CRC OK, %d layers\n", TM->name, nb, TM->n_layers);

    npu_t npu;
    npu_sim_setup();
    npu_init(&npu, 0x40000000u);
    if (!npu.present || npu.acc_depth == 0) { printf("test_depth: NPU sim without v1.2 accumulator\n"); return 1; }
    uint8_t *mem = malloc(app_work_bytes());
    static depth_net_t d;
    depth_init(&d, TM, blob, mem, &npu);

    obstacle_agent_t ag;
    obstacle_init(&ag, OBS_T_WARN_DEFAULT, OBS_T_NEAR_DEFAULT);
    int frames = 0, bad_cpu = 0, bad_npu = 0, bad_zone = 0;
    for (int fr = 0; fr < 100; fr++) {
        char p[256];
        size_t n;
        snprintf(p, sizeof p, VEC "frame_%03d.bin", fr);
        uint8_t *rgb = read_file(p, &n);
        if (!rgb) break;
        frames++;
        d.use_npu = 0;
        depth_run(&d, rgb);
        bad_cpu += check_layers(&d, fr, "CPU");
        d.use_npu = 1;
        if (depth_run(&d, rgb) != 0) { printf("  NPU path reported errors\n"); bad_npu++; }
        bad_npu += check_layers(&d, fr, "NPU");
        uint8_t db[PREPROC_W * PREPROC_H], nz[3];
        for (int i = 0; i < PREPROC_W * PREPROC_H; i++) db[i] = (uint8_t)(depth_output(&d)[i] + 128);
        zone_nearness(db, PREPROC_H, PREPROC_W, TM->out_kind, nz);
        obstacle_update(&ag, nz);
        snprintf(p, sizeof p, VEC "frame_%03d_zones.bin", fr);
        uint8_t *z = read_file(p, &n);
        if (!z || z[0] != nz[0] || z[1] != nz[1] || z[2] != nz[2] || z[3] != ag.stable) bad_zone++;
        free(z);
        free(rgb);
    }
    printf("test_depth: %d frames x %d layers: CPU mismatches %d, NPU(sim) mismatches %d, zone/agent mismatches %d\n",
           frames, TM->n_layers, bad_cpu, bad_npu, bad_zone);
    fail |= frames == 0 || bad_cpu || bad_npu || bad_zone;

    /* 4. packet flow through app.c */
    uint8_t *blob_dst = malloc(app_blob_bytes());
    app_init(&npu, blob_dst, app_blob_bytes(), mem);
    txn = 0; drain(); txn = 0;
    uint8_t sel[2] = { CMD_SET_MODEL, TEST_MODEL };
    app_handle_packet(PROTO_TYPE_CMD, sel, 2);
    drain(); txn = 0;
    static uint8_t pk[4 + 4096];
    for (uint32_t off = 0; off < nb; off += 4096) {
        uint32_t n = (nb - off < 4096) ? (uint32_t)(nb - off) : 4096;
        memcpy(pk, &off, 4);
        memcpy(pk + 4, blob + off, n);
        app_handle_packet(PROTO_TYPE_WEIGHTS, pk, (uint16_t)(4 + n));
    }
    uint8_t cmd[3] = { CMD_WEIGHTS_DONE, 0, 0 };
    app_handle_packet(PROTO_TYPE_CMD, cmd, 1);
    drain();
    uint16_t len;
    const uint8_t *ack = find_pkt(txcap, txn, PROTO_TYPE_WEIGHTS_ACK, &len);
    int ack_ok = ack && len == 9 && ack[8] == 1;
    txn = 0;
    size_t n;
    uint8_t *rgb = read_file(VEC "frame_000.bin", &n);
    uint8_t *gold = read_file(VEC "frame_000_depth.bin", &n);
    static uint8_t frame[5 + PREPROC_W * PREPROC_H * 3];
    frame[0] = PREPROC_W & 0xFF; frame[1] = PREPROC_W >> 8; frame[2] = PREPROC_H & 0xFF; frame[3] = PREPROC_H >> 8; frame[4] = 3;
    memcpy(frame + 5, rgb, (size_t)PREPROC_W * PREPROC_H * 3);
    app_handle_packet(PROTO_TYPE_FRAME, frame, (uint16_t)sizeof frame);
    drain();
    const uint8_t *dm = find_pkt(txcap, txn, PROTO_TYPE_DEPTH_MAP, &len);
    int dm_ok = dm && len == 4 + PREPROC_W * PREPROC_H && memcmp(dm + 4, gold, (size_t)PREPROC_W * PREPROC_H) == 0;
    const uint8_t *res = find_pkt(txcap, txn, PROTO_TYPE_RESULT, &len);
    int res_ok = res && len == 57;
    txn = 0;
    cmd[0] = CMD_SELFTEST;
    app_handle_packet(PROTO_TYPE_CMD, cmd, 1);
    cmd[0] = CMD_BENCH;
    app_handle_packet(PROTO_TYPE_CMD, cmd, 1);
    drain();
    const uint8_t *st = find_pkt(txcap, txn, PROTO_TYPE_SELFTEST, &len);
    const uint8_t *be = find_pkt(txcap, txn, PROTO_TYPE_BENCH, &len);
    int st_ok = st && st[0] == 1, be_ok = be && be[12] == 1;
    /* model switch: the other model has no weights yet, switching back finds ours still loaded */
    txn = 0;
    sel[1] = (uint8_t)(1 - TEST_MODEL);
    app_handle_packet(PROTO_TYPE_CMD, sel, 2);
    sel[1] = TEST_MODEL;
    app_handle_packet(PROTO_TYPE_CMD, sel, 2);
    drain();
    const uint8_t *a1 = find_pkt(txcap, txn, PROTO_TYPE_WEIGHTS_ACK, &len);
    const uint8_t *a2 = a1 ? find_pkt(a1 + len + 1, (uint32_t)(txcap + txn - (a1 + len + 1)), PROTO_TYPE_WEIGHTS_ACK, &len) : NULL;
    int sw_ok = a1 && a2 && a1[8] == 0 && a2[8] == 1;
    printf("test_depth: model switch %s (other model: not loaded, back: still loaded)\n", sw_ok ? "ok" : "FAIL");
    fail |= !sw_ok;
    printf("test_depth: app WEIGHTS_ACK %s, DEPTH_MAP %s, RESULT %s, SELFTEST %s, BENCH outputs_match %s\n",
           ack_ok ? "ok" : "FAIL", dm_ok ? "== golden" : "FAIL", res_ok ? "ok" : "FAIL", st_ok ? "pass" : "FAIL", be_ok ? "1" : "FAIL");
    fail |= !(ack_ok && dm_ok && res_ok && st_ok && be_ok);
    free(rgb); free(gold); free(blob); free(mem); free(blob_dst);
    return fail;
}
