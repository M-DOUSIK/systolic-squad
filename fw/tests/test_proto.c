#include "proto.h"
#include "test_harness.h"
#include <stdio.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)

static uint8_t buf[70000];
static uint16_t blen;

static int feed_all(const uint8_t *p, uint16_t n) {            /* returns the status of the last byte */
    int st = PROTO_BUSY;
    for (uint16_t i = 0; i < n; i++) { st = proto_rx_feed(p[i]); if (i < n - 1 && st != PROTO_BUSY) return -1; }
    return st;
}

static int round_trip(const char *what, uint8_t type, uint16_t payload) {
    harness_drain_tx(buf, &blen);
    if (blen != payload + 6u) { printf("  FAIL %s: packet is %u bytes, expected %u\n", what, blen, payload + 6u); return 1; }
    CHECK(buf[0] == 0xA5 && buf[1] == 0x5A && buf[2] == type);
    CHECK((buf[3] | (buf[4] << 8)) == payload);
    CHECK(feed_all(buf, blen) == PROTO_COMPLETE);
    CHECK(proto_get_rx_type() == type && proto_get_rx_len() == payload);
    return 0;
}

int test_proto(void) {
    printf("test_proto: START\n");
    harness_tx_head = harness_tx_tail = 0;
    /* structure sizes frozen in*/
    CHECK(sizeof(result_pkt_t) == 57);
    CHECK(sizeof(telemetry_pkt_t) == 32);

    result_pkt_t r; memset(&r, 0, sizeof(r)); r.frame_id = 0x01020304u; r.logits[7] = -5;
    proto_build_result(&r);        if (round_trip("RESULT", 0x81, 57)) return 1;
    CHECK(proto_get_rx_payload()[0] == 4 && proto_get_rx_payload()[3] == 1);   /* little endian */
    telemetry_pkt_t t; memset(&t, 0, sizeof(t)); t.uptime_ms = 1234;
    proto_build_telemetry(&t);     if (round_trip("TELEMETRY", 0x82, 32)) return 1;
    proto_build_log("hello");      if (round_trip("LOG", 0x83, 5)) return 1;
    static int8_t map[10 * 6]; memset(map, 127, sizeof(map));
    proto_build_edge_map(10, 6, map); if (round_trip("EDGE_MAP", 0x84, 4 + 60)) return 1;
    proto_build_bench(1, 2, 3, 1); if (round_trip("BENCH", 0x85, 13)) return 1;
    proto_build_selftest(1, 7, 0); if (round_trip("SELFTEST", 0x86, 5)) return 1;
    proto_build_pong(0x100, 0x53514431u); if (round_trip("PONG", 0x8F, 8)) return 1;

    /* wrong checksum is reported, garbage before a packet is skipped (resync on A5 5A) */
    proto_build_pong(1, 2); harness_drain_tx(buf, &blen);
    uint8_t bad[32]; memcpy(bad, buf, blen); bad[blen - 1] ^= 0x01;
    CHECK(feed_all(bad, blen) == PROTO_BAD_CHK);
    uint8_t mix[64]; int n = 0;
    mix[n++] = 0x00; mix[n++] = 0xA5; mix[n++] = 0xA5; mix[n++] = 0x13;      /* garbage incl. a false start */
    memcpy(mix + n, buf, blen); n += blen;
    int st = PROTO_BUSY;
    for (int i = 0; i < n; i++) { st = proto_rx_feed(mix[i]); if (st == PROTO_COMPLETE) break; }
    CHECK(st == PROTO_COMPLETE && proto_get_rx_type() == 0x8F);
    printf("test_proto: PASS (all board->host packet sizes, little endian, bad CHK, resync)\n");
    return 0;
}
