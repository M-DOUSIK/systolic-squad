/*******************************************************************************
 * Systolic Squad depth application - U54_1 (hart 1). Replaces the bring-up probe's u54_1.c.
 * APP_HART == 1: waits until the E51 has initialised everything, then runs the protocol / depth loop here (the U54 has a 32 KB
 * L1 data cache; the E51 has none, so the CPU layers - depthwise, Canny, im2col - run several times faster).
 * APP_HART == 0: idles (the E51 runs the loop).
 ******************************************************************************/
#include <stdint.h>
#include "mpfs_hal/mss_hal.h"
#include "../hart0/board_cfg.h"

volatile uint32_t count_sw_ints_h1 = 0U;
void app_run_on_this_hart(void);

void u54_1(void)
{
    clear_soft_interrupt();
    if (APP_HART == 1) app_run_on_this_hart();
    for (;;) { __asm__ volatile ("wfi"); }
}

void Software_h1_IRQHandler(void)
{
    count_sw_ints_h1++;
}
