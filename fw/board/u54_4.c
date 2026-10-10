/*******************************************************************************
 * Systolic Squad depth application - U54_4 (hart 4): worker for the CPU layers (fw/board/e51.c worker_main, platform_parallel).
 * Replaces the example's u54_4.c.
 ******************************************************************************/
#include <stdint.h>
#include "mpfs_hal/mss_hal.h"

volatile uint32_t count_sw_ints_h4 = 0U;
void worker_main(int w);

void u54_4(void)
{
    clear_soft_interrupt();
    worker_main(2);
}

void Software_h4_IRQHandler(void)
{
    count_sw_ints_h4++;
}