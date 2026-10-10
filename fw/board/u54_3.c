/*******************************************************************************
 * Systolic Squad depth application - U54_3 (hart 3): worker for the CPU layers (fw/board/e51.c worker_main, platform_parallel).
 * Replaces the example's u54_3.c.
 ******************************************************************************/
#include <stdint.h>
#include "mpfs_hal/mss_hal.h"

volatile uint32_t count_sw_ints_h3 = 0U;
void worker_main(int w);

void u54_3(void)
{
    clear_soft_interrupt();
    worker_main(1);
}

void Software_h3_IRQHandler(void)
{
    count_sw_ints_h3++;
}