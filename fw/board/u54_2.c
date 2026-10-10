/*******************************************************************************
 * Systolic Squad depth application - U54_2 (hart 2): worker for the CPU layers (fw/board/e51.c worker_main, platform_parallel).
 * Replaces the example's u54_2.c.
 ******************************************************************************/
#include <stdint.h>
#include "mpfs_hal/mss_hal.h"

volatile uint32_t count_sw_ints_h2 = 0U;
void worker_main(int w);

void u54_2(void)
{
    clear_soft_interrupt();
    worker_main(0);
}

void Software_h2_IRQHandler(void)
{
    count_sw_ints_h2++;
}