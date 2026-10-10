#include "canny.h"
#include "npu.h"
#include "test_harness.h"
#include "vec/vectors.h"
#include <stdio.h>
#include <string.h>

#define CHECK(c) do { if (!(c)) { printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); return 1; } } while (0)

static int32_t gx[MAX_CANNY_PIXELS], gy[MAX_CANNY_PIXELS];
static int8_t  blur[MAX_CANNY_PIXELS], out[MAX_CANNY_PIXELS];

static int compare(const char *name, const char *path, const canny_vec_t *v) {
    int n = v->H * v->W;
    for (int i = 0; i < n; i++)
        if (out[i] != v->edge[i]) {
            printf("  FAIL canny %s on %s: pixel %d (y=%d x=%d) got %d expect %d\n", path, name, i, i / v->W, i % v->W, out[i], v->edge[i]);
            return 1;
        }
    CHECK(canny_edge_score(out, v->H, v->W) == v->score);
    return 0;
}

int test_canny(void) {
    printf("test_canny: START\n");
    npu_sim_setup();
    npu_t big, little;
    npu_init(&big, 0x0000); npu_init(&little, 0x1000);
    static const char *names[CANNY_NV] = {"square", "step_noise", "random17x23", "random9x9", "flat"};
    for (int i = 0; i < CANNY_NV; i++) {
        const canny_vec_t *v = &CANNY_VEC[i];
        memset(out, 0x55, sizeof(out));
        canny_cpu(v->gray, v->H, v->W, gx, gy, blur, out, v->low, v->high);
        if (compare(names[i], "cpu", v)) return 1;
        memset(out, 0x55, sizeof(out));
        canny_npu(&big, v->gray, v->H, v->W, gx, gy, blur, out, v->low, v->high);
        if (compare(names[i], "npu BIG (RQ blur, RAW sobel)", v)) return 1;
        memset(out, 0x55, sizeof(out));
        canny_npu(&little, v->gray, v->H, v->W, gx, gy, blur, out, v->low, v->high);
        if (compare(names[i], "npu LITTLE (K tiles of 4)", v)) return 1;
    }
    printf("test_canny: PASS (%d frames, CPU / NPU-BIG / NPU-LITTLE all equal reference/canny_ref.py)\n", CANNY_NV);
    return 0;
}
