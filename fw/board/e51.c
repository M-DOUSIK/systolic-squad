/*******************************************************************************
 * Systolic Squad depth application - board entry on the E51 (hart 0) of the PolarFire SoC Discovery Kit.
 * Built as ONE compilation unit inside Microchip's mpfs-blank-baremetal project (see fw/board/README.md): this file includes the
 * portable firmware sources from fw/src (copied flat next to it by fw/board/build_app.ps1), so the Eclipse makefiles need no edits.
 *   - platform.h implementation: MSS UART0 + UART1 polled (whichever port the host talks to gets the replies), mtime = 1 MHz,
 *     NPU registers at NPU_BASE (FIC3 slot 0), LEDs on the reference design's CoreGPIO at 0x4000_0100
 *   - memory: model_blob.bin and all activations in DDR (trained by the MPFS HAL at boot: DDR_SUPPORT + MPFS_HAL_HW_CONFIG)
 ******************************************************************************/
#pragma GCC optimize ("O2")
#define MAX_CANNY_PIXELS (128 * 128)
#include "board_cfg.h"                   /* APP_HART: 0 = the app loop runs on the E51, 1 = on U54_1 (L1 D-cache, FPU), written by build_app.ps1 */
#include <stdint.h>
#include <string.h>
#include "mpfs_hal/mss_hal.h"
#include "drivers/mss/mss_mmuart/mss_uart.h"

#include "fw/platform.h"
#include "fw/proto.c"
#include "fw/qmath.c"
#include "fw/npu.c"
#include "fw/npu_conv3x3.c"
#include "fw/canny.c"
#include "fw/depth.c"
#include "fw/models.c"
#include "fw/obstacle.c"
#include "fw/app.c"

#ifndef NPU_BASE
#define NPU_BASE   0x40000000UL          /* FIC3 slot 0 */
#endif
#define GPIO_BASE  0x40000100UL          /* CoreGPIO (LEDs) of the reference design */
#define UART_BAUD  MSS_UART_921600_BAUD
#define DDR_BLOB   0x80000000UL          /* cached DDR: 32 MB reserved for model_blob.bin */
#define BLOB_CAP   (32u * 1024u * 1024u)
#define DDR_WORK   0x82000000UL          /* activations, Canny buffers, frame copy */

volatile uint32_t count_sw_ints_h0 = 0U;
volatile uint32_t app_ready = 0U;        /* set by the E51 when UARTs, NPU, DDR and the app state are initialised */

static mss_uart_instance_t * const uarts[2] = { &g_mss_uart0_lo, &g_mss_uart1_lo };
static int tx_uart = 0;                  /* replies go to the UART the last byte came from */

void uart_write_bytes(const uint8_t *buf, uint16_t len) { MSS_UART_polled_tx(uarts[tx_uart], buf, len); }

int uart_read_byte(uint8_t *out, uint32_t timeout_us)
{
    uint64_t t0 = readmtime();
    do {
        for (int u = 0; u < 2; u++) {
            if (MSS_UART_get_rx(uarts[u], out, 1) == 1) { tx_uart = u; return 1; }
        }
    } while (readmtime() - t0 < timeout_us);
    return 0;
}

uint32_t time_us(void) { return (uint32_t)readmtime(); }           /* mtime runs at the 1 MHz RTC toggle clock */
void npu_reg_write(uintptr_t addr, uint32_t val) { *(volatile uint32_t *)addr = val; }
uint32_t npu_reg_read(uintptr_t addr) { return *(volatile uint32_t *)addr; }

void leds_set(uint8_t val)
{
    *(volatile uint32_t *)(GPIO_BASE + 0xA0) = val;                 /* CoreGPIO GPIO_OUT */
}

static void say(const char *s) { for (int u = 0; u < 2; u++) MSS_UART_polled_tx_string(uarts[u], (const uint8_t *)s); }

/* human-readable log for a serial monitor: goes to the UART the laptop app is NOT using (COM10 when the app uses COM8).
   Do not type into that monitor: a received byte makes that UART the protocol port. */
void console_log(const char *line)
{
    mss_uart_instance_t *u = uarts[1 - tx_uart];
    MSS_UART_polled_tx_string(u, (const uint8_t *)line);
    MSS_UART_polled_tx_string(u, (const uint8_t *)"\r\n");
}

void e51(void)
{
    static npu_t npu;
    (void)mss_config_clk_rst(MSS_PERIPH_MMUART0, (uint8_t)MPFS_HAL_FIRST_HART, PERIPHERAL_ON);
    (void)mss_config_clk_rst(MSS_PERIPH_MMUART1, (uint8_t)MPFS_HAL_FIRST_HART, PERIPHERAL_ON);
    for (int u = 0; u < 2; u++) MSS_UART_init(uarts[u], UART_BAUD, MSS_UART_DATA_8_BITS | MSS_UART_NO_PARITY | MSS_UART_ONE_STOP_BIT);
    for (int i = 0; i < 8; i++) *(volatile uint32_t *)(GPIO_BASE + 4u * i) = 0x5u;   /* CoreGPIO CONFIG: output reg + output buffer */
    leds_set(0x81);

    npu_init(&npu, NPU_BASE);
    /* DDR sanity check before we trust it with the weights */
    volatile uint32_t *ddr = (volatile uint32_t *)DDR_WORK;
    int ddr_ok = 1;
    for (uint32_t i = 0; i < 4096; i++) ddr[i * 64] = i * 2654435761u;
    for (uint32_t i = 0; i < 4096; i++) if (ddr[i * 64] != i * 2654435761u) ddr_ok = 0;

    app_init(&npu, (uint8_t *)DDR_BLOB, BLOB_CAP, (uint8_t *)DDR_WORK);
    say(npu.present ? "\r\nSystolic Squad depth app: NPU present (N=16, v1.2 accumulator)" : "\r\nSystolic Squad depth app: NPU NOT present (CPU only)");
    say(ddr_ok ? ", DDR OK\r\n" : ", DDR FAILED\r\n");
    say(APP_HART ? "app loop on U54_1\r\n" : "app loop on E51\r\n");
    proto_build_pong(FW_VERSION, npu.present ? 0x53514431u : 0);
    leds_set(0x80);
    __sync_synchronize();
    app_ready = 1U;
    if (APP_HART == 0) {
        for (;;) app_poll();
    }
    for (;;) { __asm__ volatile ("wfi"); }
}

/* ---------------------------------------------------------------------------------------------- worker harts (U54_2..4)
 * platform_parallel: the app hart publishes (fn, arg) and bumps seq; every worker that has announced itself (alive) runs its part and
 * reports done = seq. Parts of workers that never started are run by the app hart itself, so a missing worker can never hang a frame.
 * The U54 L1 caches are coherent; the fences order the job fields against seq / done. */
#define N_WORKERS 3
static volatile struct {
    par_fn_t fn;
    void    *arg;
    int      nparts;
    uint32_t seq;
    uint32_t done[N_WORKERS];
    uint32_t alive[N_WORKERS];
} pool;

int platform_cores(void) { return 1 + N_WORKERS; }

void platform_parallel(par_fn_t fn, void *arg, int nparts)
{
    uint32_t alive[N_WORKERS];
    for (int w = 0; w < N_WORKERS; w++) alive[w] = pool.alive[w];
    pool.fn = fn; pool.arg = arg; pool.nparts = nparts;
    __sync_synchronize();
    uint32_t s = pool.seq + 1u;
    pool.seq = s;
    __sync_synchronize();
    fn(arg, 0, nparts);
    for (int w = 0; w < N_WORKERS; w++) if (!alive[w] && w + 1 < nparts) fn(arg, w + 1, nparts);
    for (int w = 0; w < N_WORKERS; w++) if (alive[w]) while (pool.done[w] != s) { }
    __sync_synchronize();
}

/* called by u54_2() .. u54_4() (fw/board/u54_n.c) with w = 0..2 */
void worker_main(int w)
{
    while (!app_ready) { }
    uint32_t last = pool.seq;
    pool.alive[w] = 1u;
    __sync_synchronize();
    for (;;) {
        uint32_t s = pool.seq;
        if (s == last) continue;
        __sync_synchronize();
        last = s;
        if (w + 1 < pool.nparts) pool.fn(pool.arg, w + 1, pool.nparts);
        __sync_synchronize();
        pool.done[w] = s;
    }
}

/* called by u54_1() (fw/board/u54_1.c) when APP_HART == 1 */
void app_run_on_this_hart(void)
{
    while (!app_ready) { }
    __sync_synchronize();
    for (;;) app_poll();
}

/* hart0 software interrupt handler */
void Software_h0_IRQHandler(void)
{
    count_sw_ints_h0++;
}
