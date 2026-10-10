#include "proto.h"
#include "platform.h"
#include <string.h>

#define MAX_PAYLOAD 65535u     /* contract v1.1: LEN is u16 */

static uint8_t rx_buf[MAX_PAYLOAD];
static uint16_t rx_len;
static uint8_t rx_type;
static uint16_t rx_idx;
static uint8_t rx_chk;
static int rx_state = 0;
static uint32_t last_rx_time;

int proto_rx_feed(uint8_t b) {
    uint32_t now = time_us();
    if (rx_state > 0 && now - last_rx_time > 200000) { // 200 ms inter-byte timeout
        rx_state = 0;
    }
    last_rx_time = now;

    switch (rx_state) {
        case 0:
            if (b == 0xA5) rx_state = 1;
            break;
        case 1:
            if (b == 0x5A) rx_state = 2;
            else if (b == 0xA5) rx_state = 1;
            else rx_state = 0;
            break;
        case 2:
            rx_type = b;
            rx_chk = b;
            rx_state = 3;
            break;
        case 3:
            rx_len = b;
            rx_chk += b;
            rx_state = 4;
            break;
        case 4:
            rx_len |= ((uint16_t)b << 8);
            rx_chk += b;
            if (rx_len == 0) rx_state = 6;               /* LEN is u16: never above MAX_PAYLOAD */
            else {
                rx_idx = 0;
                rx_state = 5;
            }
            break;
        case 5:
            rx_buf[rx_idx++] = b;
            rx_chk += b;
            if (rx_idx == rx_len) rx_state = 6;
            break;
        case 6:
            rx_state = 0;
            if (b == rx_chk) return PROTO_COMPLETE;
            return PROTO_BAD_CHK;
    }
    return PROTO_BUSY;
}

uint8_t proto_get_rx_type(void) { return rx_type; }
uint16_t proto_get_rx_len(void) { return rx_len; }
const uint8_t* proto_get_rx_payload(void) { return rx_buf; }

static void send_header(uint8_t type, uint16_t len, uint8_t *chk_out) {
    uint8_t hdr[5];
    hdr[0] = 0xA5;
    hdr[1] = 0x5A;
    hdr[2] = type;
    hdr[3] = len & 0xFF;
    hdr[4] = (len >> 8) & 0xFF;
    uart_write_bytes(hdr, 5);
    *chk_out = (type + hdr[3] + hdr[4]) & 0xFF;
}

static void send_chk(uint8_t chk) {
    uart_write_bytes(&chk, 1);
}

static void send_payload(const uint8_t *payload, uint16_t len, uint8_t *chk) {
    uart_write_bytes(payload, len);
    for (uint16_t i = 0; i < len; i++) {
        *chk = (*chk + payload[i]) & 0xFF;
    }
}

void proto_build_result(const result_pkt_t *res) {
    uint8_t chk;
    send_header(PROTO_TYPE_RESULT, sizeof(result_pkt_t), &chk);
    send_payload((const uint8_t*)res, sizeof(result_pkt_t), &chk);
    send_chk(chk);
}

void proto_build_telemetry(const telemetry_pkt_t *tel) {
    uint8_t chk;
    send_header(PROTO_TYPE_TELEMETRY, sizeof(telemetry_pkt_t), &chk);
    send_payload((const uint8_t*)tel, sizeof(telemetry_pkt_t), &chk);
    send_chk(chk);
}

void proto_build_log(const char *msg) {
    uint16_t len = 0;
    while (msg[len] && len < MAX_PAYLOAD) len++;
    uint8_t chk;
    send_header(PROTO_TYPE_LOG, len, &chk);
    send_payload((const uint8_t*)msg, len, &chk);
    send_chk(chk);
}

void proto_build_edge_map(uint16_t W, uint16_t H, const int8_t *map) {
    uint16_t len = 4 + W * H;
    uint8_t chk;
    send_header(PROTO_TYPE_EDGE_MAP, len, &chk);
    uint8_t dims[4] = { W & 0xFF, (W >> 8) & 0xFF, H & 0xFF, (H >> 8) & 0xFF };
    send_payload(dims, 4, &chk);
    send_payload((const uint8_t*)map, W * H, &chk);
    send_chk(chk);
}

void proto_build_bench(uint32_t cpu_us, uint32_t npu_us, uint32_t macs, uint8_t outputs_match) {
    uint8_t chk;
    send_header(PROTO_TYPE_BENCH, 13, &chk);
    uint8_t p[13];
    memcpy(&p[0], &cpu_us, 4);
    memcpy(&p[4], &npu_us, 4);
    memcpy(&p[8], &macs, 4);
    p[12] = outputs_match;
    send_payload(p, 13, &chk);
    send_chk(chk);
}

void proto_build_selftest(uint8_t pass, uint16_t n_tests, uint16_t n_fail) {
    uint8_t chk;
    send_header(PROTO_TYPE_SELFTEST, 5, &chk);
    uint8_t p[5];
    p[0] = pass;
    memcpy(&p[1], &n_tests, 2);
    memcpy(&p[3], &n_fail, 2);
    send_payload(p, 5, &chk);
    send_chk(chk);
}

void proto_build_pong(uint32_t fw_version, uint32_t npu_id) {
    uint8_t chk;
    send_header(PROTO_TYPE_PONG, 8, &chk);
    uint8_t p[8];
    memcpy(&p[0], &fw_version, 4);
    memcpy(&p[4], &npu_id, 4);
    send_payload(p, 8, &chk);
    send_chk(chk);
}

void proto_build_depth_map(uint16_t W, uint16_t H, const uint8_t *bytes) {
    uint16_t len = (uint16_t)(4 + W * H);
    uint8_t chk;
    send_header(PROTO_TYPE_DEPTH_MAP, len, &chk);
    uint8_t dims[4] = { W & 0xFF, (W >> 8) & 0xFF, H & 0xFF, (H >> 8) & 0xFF };
    send_payload(dims, 4, &chk);
    send_payload(bytes, (uint16_t)(W * H), &chk);
    send_chk(chk);
}

void proto_build_weights_ack(uint32_t bytes_received, uint32_t crc32, uint8_t ok) {
    uint8_t chk, p[9];
    memcpy(&p[0], &bytes_received, 4);
    memcpy(&p[4], &crc32, 4);
    p[8] = ok;
    send_header(PROTO_TYPE_WEIGHTS_ACK, 9, &chk);
    send_payload(p, 9, &chk);
    send_chk(chk);
}
