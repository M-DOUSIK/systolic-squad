#ifndef TEST_HARNESS_H
#define TEST_HARNESS_H

#include <stdint.h>
#include "npu.h"

extern uint8_t harness_tx_buf[65536];
extern uint16_t harness_tx_head, harness_tx_tail;

extern uint8_t harness_rx_buf[65536];
extern uint16_t harness_rx_head, harness_rx_tail;

extern uint32_t harness_time_us;
extern uint8_t harness_leds;

void harness_inject_rx(const uint8_t *data, uint16_t len);
void harness_drain_tx(uint8_t *out, uint16_t *len);
void harness_time_advance_us(uint32_t us);

void npu_sim_setup(void); // initialize NPU simulation state

#endif
