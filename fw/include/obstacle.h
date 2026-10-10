/* obstacle.h - depth map -> three zones -> guidance with hysteresis. Bit-exact port of reference/depth_ref.py (zone_nearness,
 * ObstacleAgent); the golden `_zones.bin` vectors of ml/make_vectors.py check it. */
#ifndef OBSTACLE_H
#define OBSTACLE_H
#include <stdint.h>

enum { G_CLEAR = 0, G_GO_LEFT = 1, G_GO_RIGHT = 2, G_STOP = 3 };
#define OBS_T_WARN_DEFAULT 96
#define OBS_T_NEAR_DEFAULT 160
#define OBS_EXIT_MARGIN    8
#define OBS_PERCENTILE     10
#define OBS_CONFIRM        3

typedef struct {
    uint8_t t_warn, t_near;
    uint8_t warn[3], near[3];
    uint8_t stable, cand, cand_n;
    uint8_t levels[3], raw, leds;
} obstacle_agent_t;

/* depth bytes (q + 128), H x W; kind 0 = depth (small = near), 1 = disparity. Writes n[3] = nearness L, C, R (0..255). */
void zone_nearness(const uint8_t *depth_bytes, int H, int W, int kind, uint8_t n[3]);
void obstacle_init(obstacle_agent_t *a, uint8_t t_warn, uint8_t t_near);
void obstacle_update(obstacle_agent_t *a, const uint8_t n[3]);     /* fills levels, raw, stable, leds */
#endif
