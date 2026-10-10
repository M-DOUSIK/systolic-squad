# Host tests for the portable firmware core. Run from PowerShell/MSYS2:  make test   (needs gcc + make)
CC = gcc
EXPORT ?= ../ml/export
CFLAGS = -std=c99 -Wall -Wextra -O2 -g -Iinclude -Itests -DHOST_TEST -DEXPORT_DIR=\"$(EXPORT)\"
SRCS = src/proto.c src/npu.c src/npu_sim.c src/npu_conv3x3.c src/qmath.c  src/canny.c src/main.c src/depth.c src/models.c src/obstacle.c src/app.c
ifneq ($(EXPORT),../ml/export)
CFLAGS += -DTEST_MODEL=MODEL_MIDAS
endif
TEST_SRCS = tests/test_main.c tests/test_proto.c tests/test_npu.c tests/test_canny.c tests/test_qmath.c tests/test_depth.c

.PHONY: test clean

test: $(TEST_SRCS) $(SRCS) tests/vec/vectors.h
	$(CC) $(CFLAGS) -o test_runner $(SRCS) $(TEST_SRCS) && ./test_runner

clean:
	rm -f test_runner test_runner.exe
