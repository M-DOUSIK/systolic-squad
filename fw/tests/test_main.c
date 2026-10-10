#include <stdio.h>
int test_proto(void);
int test_npu(void);
int test_canny(void);
int test_qmath(void);
int test_depth(void);

int main(void) {
    int fail = 0;
    fail |= test_qmath();
    fail |= test_proto();
    fail |= test_npu();
    fail |= test_canny();
    fail |= test_depth();
    printf(fail ? "FW TESTS: FAIL\n" : "FW TESTS: ALL PASS\n");
    return fail;
}
