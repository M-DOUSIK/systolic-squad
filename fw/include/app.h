/* app.h - the depth application on the board (portable C; the board layer only supplies platform.h and memory).
 * Protocol: WEIGHTS + WEIGHTS_DONE -> WEIGHTS_ACK; FRAME (RGB) -> Canny gate -> depth network (NPU or CPU)
 * -> zones -> obstacle agent -> DEPTH_MAP + RESULT + TELEMETRY; CMD PING/SELFTEST/BENCH/SET_GATE/EDGE_NEXT/SEND_DEPTH/SET_THRESH/USE_NPU. */
#ifndef APP_H
#define APP_H
#include <stdint.h>
#include <stddef.h>
#include "npu.h"

#define FW_VERSION 0x00030000u          /* 3.0: two models (CMD_SET_MODEL), NPU X replay buffer (v1.3) */

size_t app_work_bytes(void);            /* bytes of working memory app_init needs (activations + Canny buffers + frame copy) */
void   app_init(npu_t *npu, uint8_t *blob_buf, uint32_t blob_cap, uint8_t *work_mem);
uint32_t app_blob_bytes(void);          /* blob_cap needed to hold every model's weights */
void   app_handle_packet(uint8_t type, const uint8_t *payload, uint16_t len);
void   app_poll(void);                  /* feed every received byte to the parser, dispatch complete packets */
#endif
