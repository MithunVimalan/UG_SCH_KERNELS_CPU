# INT8 (VNNI) convolution — Core Ultra 7 270K Plus

Basis tags as in ALGORITHM_SELECTION.md. **Nothing here ran on a 270K Plus.**

## 1. What was built

| Group | Algorithm | Instruction | MAC / instruction | Code |
|---|---|---|---|---|
| 1 | Direct, input repacked 4 channels per pixel | `vpdpbusd` u8 x s8 | 32 | `src/q_direct.c` |
| 2 | im2col fused into 4-byte-interleaved GEMM panels | `vpdpbusd` u8 x s8 | 32 | `src/q_gemm.c` |
| 3 | Exact integer Winograd F(2x2,3x3) | `vpdpwssd` s16 x s16 | 16 | `src/q_winograd.c` |

fp32 FMA does 8 MAC per instruction, so the ISA ceiling of int8 over fp32 at equal
efficiency is **4x** [derived]; both use 2 instructions per cycle [spec].

Quantisation: activations u8 asymmetric (scale, zero point), weights s8 symmetric per
output channel. Accumulation is exact int32: `acc = Σ (x − zp)·w`. Epilogues: raw s32,
f32 (`fma((float)acc, in_scale·w_scale[k], bias[k])`, ReLU), or u8 requantised.

### Backends (src/vnni.h)
| Build | Instruction used | Where it runs |
|---|---|---|
| `make ARCH=arrowlake` | AVX-VNNI, VEX-encoded (`{vex} vpdpbusd`) | the target; compiled with clang 18, disassembly checked: 174 VEX VNNI instructions, 0 zmm / mask / ymm16-31 uses. **Not executed** (the dev VM has no AVX-VNNI and Intel SDE was not reachable) |
| `make` (dev VM) | AVX512-VNNI on ymm (EVEX) — same operation, same width | dev VM; all int8 numbers below |
| `make VNNI=emu` | exact AVX2 emulation (`vpmaddwd` on widened bytes) | any AVX2 CPU; tested bit-exact |

### Group 3 — why it is exact, and why it does not win
With `d = x − zp` (|d| ≤ 255), `V = Bᵀ d B` has |V| ≤ 1020 and `U' = G' g G'ᵀ` with
`G' = 2G` has |U'| ≤ 1152, both int16. `M = Σ_c U'⊙V` in int32 via `vpdpwssd`,
`4Y = Aᵀ M A`, `Y = 4Y >> 2`. Additions wrap mod 2³², so the result is exact whenever
|4Y| < 2³¹, i.e. C ≤ 1800 (checked, and tested at C = 1800 with worst-case data).
Every Q3 output equals the direct integer result bit for bit.

It cannot win by much: 16 products per 4 outputs (4/output vs 9) but int16 VNNI does
half the MACs per instruction of int8 VNNI → at best 9/4/2 = **1.125x** before
transforms [derived]. Measured it loses to Q1 on every layer (it is never picked).
F(4x4) in integers is not exact in int16 (transform coefficients up to 5 overflow), so
there is no exact int8 path to Winograd's 4x.

## 2. Verification ("debug every line")

The later line-by-line review (docs/TEST_REPORT.md) found and fixed an int8 overflow
beyond C·R·S = 65,793 (now rejected: `UG_Q_MAX_CRS`) and a u8 saturation bug.

| Check | Result |
|---|---|
| Bit-exact vs scalar integer reference (s32, f32, u8 outputs; 24 fixed + 120 random shapes: C 1-300, K 1-70, H/W 1-40, R/S 1-7 independently, strides 1-3 per axis, pads, N 1-2; zero points 0/77/128/255/random; worst-case data x=255, w=−128; 1 and 3 threads; ReLU on/off; AUTO and TUNE; C=1800 range bound) | **3706/3706 pass** (`tests/test_qconv.c`) |
| Same suite, pure-AVX2 emulation backend | 3706/3706 pass |
| fp32 suite (fixed + 100 random shapes, TUNE, API checks) | 2390/2390 pass |
| AddressSanitizer + UndefinedBehaviorSanitizer, both suites (`make sanitize`) | pass, no reports |
| Line coverage of the kernels (gcov) | q_gemm.c, q_direct.c, q_winograd.c, kernel_6x16.c, direct.c, winograd.c: **100%**; im2col_gemm.c: 100% after adding the stride-2 wide-row shape (it had never run before) |
| Mutation test: 13 injected kernel bugs + a revert of each review fix | **22/22 caught** (`tools/mutation_test.py`, `results/mutation_test.txt`) |
| Lines not executed on the VM | Arrow Lake-only branches (`ug_target` machine table, P/E core-type query) |

## 3. Selection (cost model, same method as fp32)

Features: padded MACs per algorithm, 6x16 tiles, output elements, im2col bytes on the
vector / scalar path, direct repack bytes, F2 transform work. Fitted by NNLS on 64
synthetic layers (`./build/calibrate -int8`), validated on the VGG-16 / ResNet-50
layers it never saw: **31/31 picks = measured fastest, network time 1.000x the
per-layer optimum** (1 thread, `results/calibrate_int8_devvm.txt`). At 4 threads it
picked the fastest on 29/31 (51.25 vs 49.39 ms optimum).

Which group wins (measured): 3x3 and 7x7 stride 1/2 → **Q1 direct**; all 1x1 and the
7x7-output 3x3 layers → **Q2 GEMM**; Q3 never.

## 4. Measured results (dev VM: Xeon Cascade Lake, 4 vCPU, ~3.2 GHz; NOT the target)

Accuracy: max |y_int8 − y_fp32| / max |y_fp32| = **0.41-0.66%** on every layer (the
quantisation error; identical for Q1/Q2/Q3 since they compute the same integers).

### End-to-end conv stack (every layer instance, cold weights)
```
  vgg16     | auto 247.31 ms | G1 only 471.68 ms | G2 only 433.65 ms | G3-F4 else G2 233.51 ms | G3-F2 else G2 274.61 ms | OpenBLAS 514.96 ms | auto vs OpenBLAS 2.08x, vs best single group 0.94x
  vgg16     | INT8 auto 130.93 ms | vs fp32 auto 1.89x | vs OpenBLAS fp32 3.93x
  resnet50  | auto 112.81 ms | G1 only 168.09 ms | G2 only 136.18 ms | G3-F4 else G2 124.70 ms | G3-F2 else G2 123.93 ms | OpenBLAS 144.05 ms | auto vs OpenBLAS 1.28x, vs best single group 1.10x
  resnet50  | INT8 auto 47.95 ms | vs fp32 auto 2.35x | vs OpenBLAS fp32 3.00x
  vgg16     | auto 61.06 ms | G1 only 139.04 ms | G2 only 140.82 ms | G3-F4 else G2 68.84 ms | G3-F2 else G2 74.13 ms | OpenBLAS 226.29 ms | auto vs OpenBLAS 3.71x, vs best single group 1.13x
  vgg16     | INT8 auto 45.54 ms | vs fp32 auto 1.34x | vs OpenBLAS fp32 4.97x
  resnet50  | auto 48.25 ms | G1 only 61.92 ms | G2 only 56.58 ms | G3-F4 else G2 54.16 ms | G3-F2 else G2 47.76 ms | OpenBLAS 76.66 ms | auto vs OpenBLAS 1.59x, vs best single group 0.99x
  resnet50  | INT8 auto 29.89 ms | vs fp32 auto 1.61x | vs OpenBLAS fp32 2.56x
```
(first two network blocks: 1 thread; last two: 4 threads)

### Per layer, 1 thread (ms)
| layer | x | Q1 direct | Q2 GEMM | Q3 F2 | int8 pick | fp32 auto | OpenBLAS fp32 | int8 vs fp32 | int8 vs OpenBLAS | max err vs fp32 |
|---|---|---|---|---|---|---|---|---|---|---|
| vgg16 conv1_1 | 1 | 1.849 | 2.678 | 5.041 | Q1-direct | 2.367 | 6.093 | 1.28x | 3.30x | 5.07e-03 |
| vgg16 conv1_2 | 1 | 15.524 | 27.081 | 17.671 | Q1-direct | 24.655 | 95.763 | 1.59x | 6.17x | 5.50e-03 |
| vgg16 conv2_1 | 1 | 6.959 | 10.824 | 8.904 | Q1-direct | 11.263 | 35.456 | 1.62x | 5.09x | 5.30e-03 |
| vgg16 conv2_2 | 1 | 12.931 | 20.407 | 17.000 | Q1-direct | 19.893 | 66.672 | 1.54x | 5.16x | 4.99e-03 |
| vgg16 conv3_1 | 1 | 6.702 | 8.645 | 9.142 | Q1-direct | 9.470 | 27.174 | 1.41x | 4.05x | 5.78e-03 |
| vgg16 conv3_2 | 2 | 13.838 | 17.507 | 18.114 | Q1-direct | 18.830 | 54.312 | 1.36x | 3.92x | 5.07e-03 |
| vgg16 conv4_1 | 1 | 6.597 | 8.064 | 10.994 | Q1-direct | 17.728 | 23.986 | 2.69x | 3.64x | 5.82e-03 |
| vgg16 conv4_2 | 2 | 13.637 | 15.857 | 27.888 | Q1-direct | 37.753 | 47.840 | 2.77x | 3.51x | 5.72e-03 |
| vgg16 conv5_x | 3 | 3.148 | 5.120 | 7.334 | Q1-direct | 8.905 | 13.031 | 2.83x | 4.14x | 5.28e-03 |
| resnet50 stem7x7s2 | 1 | 1.232 | 2.223 | - | Q1-direct | 2.710 | 6.234 | 2.20x | 5.06x | 5.68e-03 |
| resnet50 l1_1x1_64 | 3 | 0.337 | 0.166 | - | Q2-im2col+gemm | 0.345 | 0.338 | 2.08x | 2.03x | 4.12e-03 |
| resnet50 l1_3x3 | 3 | 1.007 | 2.285 | 1.201 | Q1-direct | 1.097 | 5.484 | 1.09x | 5.45x | 6.56e-03 |
| resnet50 l1_1x1_up | 4 | 0.841 | 0.599 | - | Q2-im2col+gemm | 1.187 | 1.374 | 1.98x | 2.29x | 5.39e-03 |
| resnet50 l2_1x1_red_s1 | 1 | 1.751 | 0.834 | - | Q2-im2col+gemm | 2.861 | 2.491 | 3.43x | 2.99x | 5.14e-03 |
| resnet50 l2_3x3_s2 | 1 | 1.210 | 2.013 | - | Q1-direct | 3.970 | 4.086 | 3.28x | 3.38x | 5.57e-03 |
| resnet50 l2_1x1_up | 4 | 0.630 | 0.429 | - | Q2-im2col+gemm | 1.133 | 1.202 | 2.64x | 2.80x | 4.53e-03 |
| resnet50 l2_ds_1x1_s2 | 1 | 1.952 | 0.957 | - | Q2-im2col+gemm | 2.685 | 2.653 | 2.81x | 2.77x | 5.00e-03 |
| resnet50 l2_1x1_red | 3 | 0.803 | 0.364 | - | Q2-im2col+gemm | 1.232 | 1.194 | 3.38x | 3.28x | 5.44e-03 |
| resnet50 l2_3x3 | 3 | 0.866 | 1.971 | 1.524 | Q1-direct | 1.124 | 3.972 | 1.30x | 4.59x | 5.61e-03 |
| resnet50 l3_1x1_red_s1 | 1 | 1.233 | 0.669 | - | Q2-im2col+gemm | 2.394 | 2.283 | 3.58x | 3.41x | 5.27e-03 |
| resnet50 l3_3x3_s2 | 1 | 0.946 | 1.871 | - | Q1-direct | 3.473 | 3.491 | 3.67x | 3.69x | 5.99e-03 |
| resnet50 l3_1x1_up | 6 | 0.485 | 0.356 | - | Q2-im2col+gemm | 1.126 | 1.167 | 3.16x | 3.27x | 6.10e-03 |
| resnet50 l3_ds_1x1_s2 | 1 | 1.263 | 0.889 | - | Q2-im2col+gemm | 2.656 | 2.509 | 2.99x | 2.82x | 5.23e-03 |
| resnet50 l3_1x1_red | 5 | 0.569 | 0.333 | - | Q2-im2col+gemm | 1.188 | 1.174 | 3.57x | 3.53x | 4.28e-03 |
| resnet50 l3_3x3 | 5 | 0.758 | 1.859 | 1.694 | Q1-direct | 1.314 | 3.415 | 1.73x | 4.50x | 5.08e-03 |
| resnet50 l4_1x1_red_s1 | 1 | 0.923 | 0.639 | - | Q2-im2col+gemm | 2.363 | 2.311 | 3.70x | 3.61x | 5.20e-03 |
| resnet50 l4_3x3_s2 | 1 | 1.582 | 1.339 | - | Q2-im2col+gemm | 4.254 | 3.816 | 3.18x | 2.85x | 5.05e-03 |
| resnet50 l4_1x1_up | 3 | 0.756 | 0.384 | - | Q2-im2col+gemm | 1.349 | 1.375 | 3.52x | 3.58x | 5.46e-03 |
| resnet50 l4_ds_1x1_s2 | 1 | 1.861 | 0.833 | - | Q2-im2col+gemm | 3.483 | 3.126 | 4.18x | 3.75x | 6.30e-03 |
| resnet50 l4_1x1_red | 2 | 0.855 | 0.384 | - | Q2-im2col+gemm | 1.387 | 1.413 | 3.61x | 3.68x | 5.16e-03 |
| resnet50 l4_3x3 | 2 | 1.454 | 1.341 | 1.668 | Q2-im2col+gemm | 2.511 | 3.819 | 1.87x | 2.85x | 4.63e-03 |

```
Totals(19.36 GMAC): int8 auto 148.02 ms | int8 best-per-layer 148.02 ms | fp32 auto 307.90 ms | OpenBLAS fp32 619.34 ms
  int8 auto vs fp32 auto 2.08x, vs OpenBLAS fp32 4.18x; int8 selector picked the fastest on 31/31 layers; 261.5 effective GOP/s
```

### Per layer, 4 threads (ms)
| layer | x | Q1 direct | Q2 GEMM | Q3 F2 | int8 pick | fp32 auto | OpenBLAS fp32 | int8 vs fp32 | int8 vs OpenBLAS | max err vs fp32 |
|---|---|---|---|---|---|---|---|---|---|---|
| vgg16 conv1_1 | 1 | 0.765 | 1.018 | 1.500 | Q1-direct | 0.709 | 2.245 | 0.93x | 2.93x | 5.07e-03 |
| vgg16 conv1_2 | 1 | 4.103 | 6.843 | 4.700 | Q1-direct | 6.736 | 30.054 | 1.64x | 7.33x | 5.50e-03 |
| vgg16 conv2_1 | 1 | 1.865 | 2.741 | 2.357 | Q1-direct | 2.624 | 9.943 | 1.41x | 5.33x | 5.30e-03 |
| vgg16 conv2_2 | 1 | 3.825 | 5.515 | 4.317 | Q1-direct | 5.202 | 24.770 | 1.36x | 6.48x | 4.99e-03 |
| vgg16 conv3_1 | 1 | 1.843 | 2.357 | 2.377 | Q1-direct | 2.490 | 9.572 | 1.35x | 5.19x | 5.78e-03 |
| vgg16 conv3_2 | 2 | 3.948 | 4.601 | 4.643 | Q1-direct | 4.757 | 17.454 | 1.20x | 4.42x | 5.07e-03 |
| vgg16 conv4_1 | 1 | 2.076 | 3.089 | 2.924 | Q1-direct | 3.230 | 7.725 | 1.56x | 3.72x | 5.82e-03 |
| vgg16 conv4_2 | 2 | 3.857 | 6.148 | 5.662 | Q1-direct | 7.317 | 14.298 | 1.90x | 3.71x | 5.72e-03 |
| vgg16 conv5_x | 3 | 1.022 | 2.366 | 1.819 | Q1-direct | 2.499 | 4.829 | 2.44x | 4.72x | 5.28e-03 |
| resnet50 stem7x7s2 | 1 | 0.529 | 0.682 | - | Q1-direct | 0.903 | 2.088 | 1.71x | 3.95x | 5.68e-03 |
| resnet50 l1_1x1_64 | 3 | 0.180 | 0.099 | - | Q2-im2col+gemm | 0.215 | 0.159 | 2.16x | 1.60x | 4.12e-03 |
| resnet50 l1_3x3 | 3 | 0.436 | 0.705 | 0.556 | Q1-direct | 0.490 | 2.195 | 1.12x | 5.03x | 6.56e-03 |
| resnet50 l1_1x1_up | 4 | 0.387 | 0.306 | - | Q2-im2col+gemm | 0.489 | 0.561 | 1.60x | 1.84x | 5.39e-03 |
| resnet50 l2_1x1_red_s1 | 1 | 0.687 | 0.324 | - | Q2-im2col+gemm | 0.802 | 0.764 | 2.47x | 2.35x | 5.14e-03 |
| resnet50 l2_3x3_s2 | 1 | 0.498 | 0.970 | - | Q1-direct | 1.357 | 1.468 | 2.73x | 2.95x | 5.57e-03 |
| resnet50 l2_1x1_up | 4 | 0.268 | 0.255 | - | Q2-im2col+gemm | 0.457 | 0.511 | 1.79x | 2.01x | 4.53e-03 |
| resnet50 l2_ds_1x1_s2 | 1 | 0.676 | 0.488 | - | Q2-im2col+gemm | 0.899 | 1.047 | 1.84x | 2.15x | 5.00e-03 |
| resnet50 l2_1x1_red | 3 | 0.564 | 0.226 | - | Q2-im2col+gemm | 0.550 | 0.644 | 2.44x | 2.85x | 5.44e-03 |
| resnet50 l2_3x3 | 3 | 0.379 | 0.953 | 0.570 | Q1-direct | 0.663 | 1.854 | 1.75x | 4.89x | 5.61e-03 |
| resnet50 l3_1x1_red_s1 | 1 | 0.562 | 0.282 | - | Q2-im2col+gemm | 0.770 | 0.787 | 2.73x | 2.79x | 5.27e-03 |
| resnet50 l3_3x3_s2 | 1 | 0.439 | 0.865 | - | Q1-direct | 1.257 | 1.419 | 2.86x | 3.23x | 5.99e-03 |
| resnet50 l3_1x1_up | 6 | 0.267 | 0.201 | - | Q2-im2col+gemm | 0.402 | 0.579 | 1.99x | 2.87x | 6.10e-03 |
| resnet50 l3_ds_1x1_s2 | 1 | 0.517 | 0.401 | - | Q2-im2col+gemm | 1.266 | 1.066 | 3.16x | 2.66x | 5.23e-03 |
| resnet50 l3_1x1_red | 5 | 0.357 | 0.187 | - | Q2-im2col+gemm | 0.478 | 0.519 | 2.55x | 2.78x | 4.28e-03 |
| resnet50 l3_3x3 | 5 | 0.333 | 0.839 | 0.616 | Q1-direct | 0.688 | 1.426 | 2.07x | 4.28x | 5.08e-03 |
| resnet50 l4_1x1_red_s1 | 1 | 0.522 | 0.289 | - | Q2-im2col+gemm | 0.766 | 1.021 | 2.65x | 3.53x | 5.20e-03 |
| resnet50 l4_3x3_s2 | 1 | 0.613 | 1.286 | - | Q2-im2col+gemm | 1.963 | 2.165 | 1.53x | 1.68x | 5.05e-03 |
| resnet50 l4_1x1_up | 3 | 0.306 | 0.195 | - | Q2-im2col+gemm | 0.476 | 0.849 | 2.45x | 4.35x | 5.46e-03 |
| resnet50 l4_ds_1x1_s2 | 1 | 0.690 | 0.491 | - | Q2-im2col+gemm | 1.104 | 2.086 | 2.25x | 4.25x | 6.30e-03 |
| resnet50 l4_1x1_red | 2 | 0.400 | 0.269 | - | Q2-im2col+gemm | 0.530 | 0.985 | 1.97x | 3.66x | 5.16e-03 |
| resnet50 l4_3x3 | 2 | 0.643 | 1.236 | 0.779 | Q2-im2col+gemm | 1.206 | 1.712 | 0.98x | 1.38x | 4.63e-03 |

```
Totals(19.36 GMAC): int8 auto 51.25 ms | int8 best-per-layer 49.39 ms | fp32 auto 86.41 ms | OpenBLAS fp32 216.20 ms
  int8 auto vs fp32 auto 1.69x, vs OpenBLAS fp32 4.22x; int8 selector picked the fastest on 29/31 layers; 755.4 effective GOP/s
```

## 5. The 10x target — what the numbers say

* vs the naive fp32 loop (46.8 s, 1 thread): int8 auto 148 ms = **316x** [measured-VM].
* vs OpenBLAS fp32 im2col+SGEMM: **4.2x** per-layer total (1 and 4 threads);
  end to end 2.6-5.0x; best single layer 7.3x (VGG conv1_2 at 4 threads, where OpenBLAS
  pays for an explicit im2col).
* vs our own fp32 path: 1.3-2.4x end to end. On 3x3 stride-1 layers the fp32 path
  already uses Winograd (up to 4x fewer multiplies), which int8 cannot use exactly, so
  int8 gains 0.93-2.4x there (4 threads; slower than fp32 on VGG conv1_1 and the 7x7
  ResNet 3x3); on 1x1 layers it gains 1.6-3.2x.

Why not 10x over a good fp32 library, on this ISA:
1. int8 VNNI does 4x the MACs per instruction of fp32 FMA — the ceiling for
   same-algorithm int8 vs fp32 is 4x.
2. OpenBLAS already runs 1x1 layers (half of ResNet-50's time) near fp32 peak, so
   those cap at ~4x.
3. Exact MAC reduction on top of int8 is limited to F(2x2) via int16 (≤ 1.125x, measured
   to lose). A larger saving needs approximation or model changes (pruning/sparsity,
   lower precision like int4, which the ISA has no instruction for).

What would move the target numbers: the 270K Plus has 3 load ports (VM: 2) which
relieves the Q1 direct kernel (measured ~65% of VNNI peak here, load-port bound on the
VM), and 24 cores. Re-run on the target:

```sh
make ARCH=arrowlake && ./build/test_qconv          # must print 3706/3706 passed
./build/calibrate -int8 -o qcostmodel.txt && export UGCONV_QCOSTMODEL=$PWD/qcostmodel.txt
OMP_WAIT_POLICY=PASSIVE ./build/bench_conv -t 24 -int8
OMP_WAIT_POLICY=PASSIVE ./build/bench_conv -t 24 -e2e -int8
```
