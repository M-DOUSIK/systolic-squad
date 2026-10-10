#ifndef PLATFORM_H
#define PLATFORM_H

#include <stdint.h>

void uart_write_bytes(const uint8_t *buf, uint16_t len);
int uart_read_byte(uint8_t *out, uint32_t timeout_us);
uint32_t time_us(void);
void npu_reg_write(uintptr_t addr, uint32_t val);
uint32_t npu_reg_read(uintptr_t addr);
void leds_set(uint8_t val);
void console_log(const char *line);      /* one human-readable line on the serial monitor (board: the UART the host is NOT using) */

/* run fn(arg, part, nparts) for part = 0 .. nparts-1, in parallel where the platform can (board: the app hart + the worker harts
   U54_2..4), return when all parts are done. Parts must write disjoint outputs. platform_cores() = the nparts to use. */
typedef void (*par_fn_t)(void *arg, int part, int nparts);
void platform_parallel(par_fn_t fn, void *arg, int nparts);
int  platform_cores(void);

#endif
