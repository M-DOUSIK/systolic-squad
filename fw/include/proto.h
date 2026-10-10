#ifndef PROTO_H
#define PROTO_H

#include <stdint.h>

/* Packet Types */
#define PROTO_TYPE_FRAME    0x01
#define PROTO_TYPE_CMD      0x02
#define PROTO_TYPE_RESULT   0x81
#define PROTO_TYPE_TELEMETRY 0x82
#define PROTO_TYPE_LOG      0x83
#define PROTO_TYPE_EDGE_MAP 0x84
#define PROTO_TYPE_BENCH    0x85
#define PROTO_TYPE_SELFTEST 0x86
#define PROTO_TYPE_PONG     0x8F
#define PROTO_TYPE_WEIGHTS  0x03
#define PROTO_TYPE_DEPTH_MAP 0x87
#define PROTO_TYPE_WEIGHTS_ACK 0x88

/* Commands */
#define CMD_PING      0x01
#define CMD_SET_MODE  0x02
#define CMD_SELFTEST  0x03
#define CMD_BENCH     0x04
#define CMD_SET_GATE  0x05
#define CMD_EDGE_NEXT 0x06
#define CMD_SEND_DEPTH   0x07
#define CMD_SET_THRESH   0x08
#define CMD_WEIGHTS_DONE 0x09
#define CMD_USE_NPU      0x0A     /* v1.2: u8 0 = depth network on the CPU only, 1 = conv layers on the NPU (default) */
#define CMD_SET_MODEL    0x0B     /* v1.3: u8 model index (0 FastDepth, 1 MiDaS); answer WEIGHTS_ACK of that model's weight slot */
#define CMD_PROFILE      0x0C     /* v1.3: LOG packet per layer of the last depth run: index, type, microseconds */

/* Parser Status */
#define PROTO_BUSY      0
#define PROTO_COMPLETE  1
#define PROTO_BAD_CHK   2
#define PROTO_TIMEOUT   3

#pragma pack(push, 1)

typedef struct {
    uint32_t frame_id;
    uint8_t n_classes;
    uint8_t g_raw;
    uint8_t g_stable;
    uint8_t command;
    uint8_t flags;
    int32_t logits[8];
    uint32_t t_total_us;
    uint32_t t_cnn_us;
    uint32_t npu_cycles;
    uint32_t npu_active;
} __attribute__((packed)) result_pkt_t;

typedef struct {
    uint8_t mode;
    uint8_t reason;
    uint16_t fps_x10;
    uint32_t frames_seen;
    uint32_t frames_gated;
    uint32_t cnn_runs;
    uint32_t uptime_ms;
    uint16_t motion_score;
    uint16_t edge_score;
    uint8_t engines_awake;
    uint8_t gate_engine;
    uint16_t zero_skip_permille;
    uint32_t big_sleep_ms;
} __attribute__((packed)) telemetry_pkt_t;

#pragma pack(pop)

/* Parser API */
int proto_rx_feed(uint8_t b);
uint8_t proto_get_rx_type(void);
uint16_t proto_get_rx_len(void);
const uint8_t* proto_get_rx_payload(void);

/* Builder API */
void proto_build_result(const result_pkt_t *res);
void proto_build_telemetry(const telemetry_pkt_t *tel);
void proto_build_log(const char *msg);
void proto_build_edge_map(uint16_t W, uint16_t H, const int8_t *map);
void proto_build_bench(uint32_t cpu_us, uint32_t npu_us, uint32_t macs, uint8_t outputs_match);
void proto_build_selftest(uint8_t pass, uint16_t n_tests, uint16_t n_fail);
void proto_build_pong(uint32_t fw_version, uint32_t npu_id);
void proto_build_depth_map(uint16_t W, uint16_t H, const uint8_t *bytes);
void proto_build_weights_ack(uint32_t bytes_received, uint32_t crc32, uint8_t ok);

#endif
