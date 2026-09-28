# UG_SCH_KERNELS_CPU

FP32 convolution kernels for the **Intel Core Ultra 7 270K Plus** (Arrow Lake-S,
8 Lion Cove P + 16 Skymont E), organised in three algorithm groups with a
calibrated selector:

| Group | Algorithm | Used for (measured on the dev VM) |
|---|---|---|
| 1 | **Direct convolution** (`UG_ALGO_DIRECT`) | stems / few input channels (VGG conv1_1, ResNet 7x7/2 stem), some 3x3 stride-2 |
| 2 | **im2col + GEMM** (`UG_ALGO_IM2COL_GEMM`) | every 1x1, most strided convs, anything not 3x3 stride 1 |
| 3 | **Winograd F(4x4,3x3) / F(2x2,3x3)** (`UG_ALGO_WINOGRAD_F4/F2`) | 3x3 stride 1 with enough channels; F2 when the image is small (7x7) or F4's weights overflow L3 |

`UG_ALGO_AUTO` picks with a cost model that prices the exact schedule each group
will run; `UG_ALGO_TUNE` times every eligible algorithm once and keeps the fastest.

ISA: AVX2 + FMA only (Arrow Lake has no AVX-512). Layouts: NCHW in, KCRS weights,
NKPQ out; bias + ReLU fused.

## Build and test

```sh
make                    # portable AVX2 build (x86-64-v3)
make ARCH=arrowlake     # target build, -march=arrowlake-s (clang >= 18 / gcc >= 14)
make test               # 276 correctness runs vs an fp64 reference
./build/bench_conv -t <threads> [-n vgg16|resnet50] [-e2e] [-naive] [-csv file]
./build/calibrate -o costmodel.txt && export UGCONV_COSTMODEL=$PWD/costmodel.txt
```

## API

```c
#include "ugconv.h"
ug_conv_desc d = {N, C, H, W, K, R, S, stride_h, stride_w, pad_h, pad_w};
ug_conv_plan *p = ug_conv_plan_create(&d, weights, bias, /*relu*/1, UG_ALGO_AUTO, /*threads*/0);
ug_conv_execute(p, input, output);
ug_conv_plan_destroy(p);
```

## Results

See [docs/ALGORITHM_SELECTION.md](docs/ALGORITHM_SELECTION.md) for the design,
the checklist derivation of each group, the cost model and its held-out validation,
and all measured numbers. **All measurements so far are from a 4-vCPU Xeon
(Cascade Lake) development VM, not from a 270K Plus**; raw logs are in `results/`.
