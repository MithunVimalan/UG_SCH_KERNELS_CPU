# Target build (Core Ultra 7 270K Plus):  make ARCH=arrowlake
#   needs clang >= 18 or gcc >= 14 for -march=arrowlake-s.
# Portable AVX2 build (any Haswell+ CPU, used for CI / the dev VM):  make
ARCH ?= avx2

ifeq ($(ARCH),arrowlake)
  CC ?= clang
  ARCHFLAGS = -march=arrowlake-s -mtune=arrowlake-s
else
  ARCHFLAGS = -march=x86-64-v3 -mtune=skylake
endif

CFLAGS ?= -O3 -std=c11 -D_POSIX_C_SOURCE=200809L -Wall -Wextra -Wno-unused-parameter
CFLAGS += $(ARCHFLAGS) -fopenmp -Iinclude
LDLIBS = -lm

SRC = src/cpu.c src/plan.c src/kernel_6x16.c src/direct.c src/im2col_gemm.c src/winograd.c
OBJ = $(SRC:src/%.c=build/%.o)

OPENBLAS_CFLAGS ?= $(shell pkg-config --cflags openblas 2>/dev/null || echo -I/usr/include/x86_64-linux-gnu/openblas-pthread)
OPENBLAS_LIBS ?= $(shell pkg-config --libs openblas 2>/dev/null || echo -lopenblas)

all: build/libugconv.a build/test_conv build/bench_conv build/calibrate

build:
	mkdir -p build

build/%.o: src/%.c src/ug_internal.h include/ugconv.h | build
	$(CC) $(CFLAGS) -c $< -o $@

build/libugconv.a: $(OBJ)
	ar rcs $@ $^

build/test_conv: tests/test_conv.c build/libugconv.a
	$(CC) $(CFLAGS) $< build/libugconv.a $(LDLIBS) -o $@

# bench links OpenBLAS for the im2col+SGEMM baseline when available (make NO_OPENBLAS=1 to skip)
ifeq ($(NO_OPENBLAS),1)
build/bench_conv: bench/bench_conv.c bench/layers.h build/libugconv.a
	$(CC) $(CFLAGS) -DNO_OPENBLAS $< build/libugconv.a $(LDLIBS) -o $@
else
build/bench_conv: bench/bench_conv.c bench/layers.h build/libugconv.a
	$(CC) $(CFLAGS) $(OPENBLAS_CFLAGS) $< build/libugconv.a $(OPENBLAS_LIBS) $(LDLIBS) -o $@
endif

build/calibrate: tools/calibrate.c bench/layers.h build/libugconv.a
	$(CC) $(CFLAGS) $< build/libugconv.a $(LDLIBS) -o $@

test: build/test_conv
	./build/test_conv

clean:
	rm -rf build

.PHONY: all test clean
