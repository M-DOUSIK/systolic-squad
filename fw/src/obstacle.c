/* obstacle.c - see obstacle.h; line-by-line port of reference/depth_ref.py. */
#include "obstacle.h"
#include <string.h>

void zone_nearness(const uint8_t *depth_bytes, int H, int W, int kind, uint8_t n[3])
{
    int c0[3] = { 0, W / 3, 2 * (W / 3) }, c1[3] = { W / 3, 2 * (W / 3), W };
    for (int z = 0; z < 3; z++) {
        uint32_t hist[256];
        memset(hist, 0, sizeof(hist));
        int size = 0;
        for (int y = H / 4; y < H; y++)
            for (int x = c0[z]; x < c1[z]; x++) {
                int b = depth_bytes[y * W + x];
                hist[kind == 0 ? 255 - b : b]++;
                size++;
            }
        uint32_t need = (uint32_t)((size * OBS_PERCENTILE + 99) / 100);     /* ceil(size * pct / 100) */
        uint32_t acc = 0;
        int t = 0;
        for (int v = 255; v >= 0; v--) {
            acc += hist[v];
            if (acc >= need) { t = v; break; }
        }
        n[z] = (uint8_t)t;
    }
}

void obstacle_init(obstacle_agent_t *a, uint8_t t_warn, uint8_t t_near)
{
    memset(a, 0, sizeof(*a));
    a->t_warn = t_warn;
    a->t_near = t_near;
    a->stable = a->cand = G_CLEAR;
}

void obstacle_update(obstacle_agent_t *a, const uint8_t n[3])
{
    for (int z = 0; z < 3; z++) {
        a->warn[z] = a->warn[z] ? (n[z] >= a->t_warn - OBS_EXIT_MARGIN) : (n[z] >= a->t_warn);
        a->near[z] = a->near[z] ? (n[z] >= a->t_near - OBS_EXIT_MARGIN) : (n[z] >= a->t_near);
        a->levels[z] = a->near[z] ? 2 : (a->warn[z] ? 1 : 0);
    }
    const uint8_t *lv = a->levels;
    uint8_t raw;
    if (lv[1] == 0) raw = G_CLEAR;
    else {
        int left_free = lv[0] < 2, right_free = lv[2] < 2;
        if (!left_free && !right_free) raw = G_STOP;
        else if (left_free && right_free) raw = (n[0] <= n[2]) ? G_GO_LEFT : G_GO_RIGHT;
        else raw = left_free ? G_GO_LEFT : G_GO_RIGHT;
    }
    a->raw = raw;
    if (raw == G_STOP) { a->stable = G_STOP; a->cand = G_STOP; a->cand_n = 0; }
    else if (raw == a->stable) { a->cand = raw; a->cand_n = 0; }
    else {
        a->cand_n = (raw == a->cand) ? a->cand_n + 1 : 1;
        a->cand = raw;
        if (a->cand_n >= OBS_CONFIRM) { a->stable = raw; a->cand_n = 0; }
    }
    uint8_t leds = (uint8_t)((lv[0] >= 1) | ((lv[0] == 2) << 1) | ((lv[1] >= 1) << 2) | ((lv[1] == 2) << 3) |
                             ((lv[2] >= 1) << 4) | ((lv[2] == 2) << 5));
    leds |= (uint8_t)(((a->stable == G_STOP) << 6) | (1 << 7));
    a->leds = leds;
}
