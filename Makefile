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
CFLAGS += $(ARCHFLAGS) -fopenmp -Iinclude $(EXTRA_CFLAGS)
LDLIBS = -lm $(EXTRA_LDFLAGS)

SRC = src/cpu.c src/plan.c src/kernel_6x16.c src/direct.c src/im2col_gemm.c src/winograd.c
QSRC = src/q_plan.c src/q_gemm.c src/q_direct.c src/q_winograd.c
OBJ = $(SRC:src/%.c=build/%.o) $(QSRC:src/%.c=build/%.o)

# INT8 VNNI backend (see src/vnni.h):
#   ARCH=arrowlake -> AVX-VNNI (VEX), the target instruction.
#   default        -> AVX512-VNNI on ymm (EVEX) so the dev VM (no AVX-VNNI) can run it;
#                     only the int8 files get these flags, 256-bit vectors only.
#   VNNI=emu       -> exact AVX2-only emulation (any AVX2 CPU).
ifeq ($(ARCH),arrowlake)
  QFLAGS =
else ifeq ($(VNNI),emu)
  QFLAGS = -DUG_FORCE_VNNI_EMU
else
  QFLAGS = -mavx512vnni -mavx512vl -mavx512bw -mprefer-vector-width=256 -DUG_DEV_EVEX_VNNI
endif

OPENBLAS_CFLAGS ?= $(shell pkg-config --cflags openblas 2>/dev/null || echo -I/usr/include/x86_64-linux-gnu/openblas-pthread)
OPENBLAS_LIBS ?= $(shell pkg-config --libs openblas 2>/dev/null || echo -lopenblas)

all: build/libugconv.a build/test_conv build/test_qconv build/test_regression build/test_stress build/bench_conv build/calibrate

build:
	mkdir -p build

build/q_%.o: src/q_%.c src/q_internal.h src/vnni.h src/ug_internal.h include/ugconv.h | build
	$(CC) $(CFLAGS) $(QFLAGS) -c $< -o $@

build/%.o: src/%.c src/ug_internal.h include/ugconv.h | build
	$(CC) $(CFLAGS) -c $< -o $@

build/libugconv.a: $(OBJ)
	ar rcs $@ $^

build/test_conv: tests/test_conv.c build/libugconv.a
	$(CC) $(CFLAGS) $< build/libugconv.a $(LDLIBS) -o $@

build/test_qconv: tests/test_qconv.c build/libugconv.a
	$(CC) $(CFLAGS) $< build/libugconv.a $(LDLIBS) -o $@

build/test_regression: tests/test_regression.c build/libugconv.a
	$(CC) $(CFLAGS) $< build/libugconv.a $(LDLIBS) -o $@

build/test_stress: tests/test_stress.c build/libugconv.a
	$(CC) $(CFLAGS) $< build/libugconv.a $(LDLIBS) -lpthread -o $@

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

test: build/test_conv build/test_qconv build/test_regression build/test_stress
	./build/test_conv | tail -1
	./build/test_qconv | tail -1
	./build/test_regression | tail -1
	./build/test_stress | tail -1

# AddressSanitizer + UndefinedBehaviorSanitizer build of both test suites (separate dir)
SAN = -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer
sanitize:
	$(MAKE) -s BUILD_SAN=1 EXTRA_CFLAGS="$(SAN)" EXTRA_LDFLAGS="-fsanitize=address,undefined" clean-objs build/test_conv build/test_qconv build/test_regression build/test_stress
	./build/test_conv | tail -1
	./build/test_qconv | tail -1
	ASAN_OPTIONS=allocator_may_return_null=1 ./build/test_regression | tail -1   # malloc returns NULL, as in glibc
	./build/test_stress | tail -1
	$(MAKE) -s clean-objs

# ThreadSanitizer: concurrent first use of the library (F6). Separate build dir.
tsan:
	mkdir -p build/tsan
	for f in $(SRC); do $(CC) -O1 -g -fsanitize=thread -std=c11 -D_POSIX_C_SOURCE=200809L $(ARCHFLAGS) -fopenmp -Iinclude -c $$f -o build/tsan/$$(basename $$f .c).o || exit 1; done
	for f in $(QSRC); do $(CC) -O1 -g -fsanitize=thread -std=c11 -D_POSIX_C_SOURCE=200809L $(ARCHFLAGS) $(QFLAGS) -fopenmp -Iinclude -c $$f -o build/tsan/$$(basename $$f .c).o || exit 1; done
	$(CC) -O1 -g -fsanitize=thread $(ARCHFLAGS) -fopenmp -Iinclude tests/test_race.c build/tsan/*.o -lm -lpthread -o build/tsan/test_race
	TSAN_OPTIONS=halt_on_error=0 ./build/tsan/test_race

clean-objs:
	rm -f build/*.o build/libugconv.a build/test_conv build/test_qconv build/test_regression build/test_stress

clean:
	rm -rf build

.PHONY: all test clean clean-objs sanitize tsan
