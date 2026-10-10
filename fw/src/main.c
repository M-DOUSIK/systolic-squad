/* main.c - platform layer for the laptop host tests (make test, -DHOST_TEST): UART, time and LEDs simulated in memory,
 * the NPU registers by src/npu_sim.c. The board's platform layer and entry point are in fw/board/e51.c. */
#include "platform.h"
#include "proto.h"
#include "npu.h"

#ifdef HOST_TEST
#include "test_harness.h"

uint8_t harness_tx_buf[65536];
uint16_t harness_tx_head, harness_tx_tail;

uint8_t harness_rx_buf[65536];
uint16_t harness_rx_head, harness_rx_tail;

uint32_t harness_time_us = 0;
uint8_t harness_leds = 0;

void harness_inject_rx(const uint8_t *data, uint16_t len) {
    for (int i = 0; i < len; i++) {
        harness_rx_buf[harness_rx_head] = data[i];
        harness_rx_head = (harness_rx_head + 1) % 65536;
    }
}

void harness_drain_tx(uint8_t *out, uint16_t *len) {
    *len = 0;
    while (harness_tx_tail != harness_tx_head) {
        out[(*len)++] = harness_tx_buf[harness_tx_tail];
        harness_tx_tail = (harness_tx_tail + 1) % 65536;
    }
}

void harness_time_advance_us(uint32_t us) {
    harness_time_us += us;
}

void uart_write_bytes(const uint8_t *buf, uint16_t len) {
    for (int i = 0; i < len; i++) {
        harness_tx_buf[harness_tx_head] = buf[i];
        harness_tx_head = (harness_tx_head + 1) % 65536;
    }
}

int uart_read_byte(uint8_t *out, uint32_t timeout_us) {
    uint32_t t0 = harness_time_us;
    while (harness_rx_tail == harness_rx_head) {
        if (harness_time_us - t0 > timeout_us) return 0;
    }
    *out = harness_rx_buf[harness_rx_tail];
    harness_rx_tail = (harness_rx_tail + 1) % 65536;
    return 1;
}

uint32_t time_us(void) {
    return harness_time_us;
}

void leds_set(uint8_t val) {
    harness_leds = val;
}

/* host tests: the parts run one after the other (4 parts, so the row splitting is exercised) */
void platform_parallel(par_fn_t fn, void *arg, int nparts) { for (int p = 0; p < nparts; p++) fn(arg, p, nparts); }
int  platform_cores(void) { return 4; }

void console_log(const char *line) {
    (void)line;                       /* host tests: the board's text console is not checked */
}

#endif /* HOST_TEST */
