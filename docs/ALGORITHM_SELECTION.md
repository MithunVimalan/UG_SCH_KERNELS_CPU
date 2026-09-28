# Algorithm groups and selection — Core Ultra 7 270K Plus

Every number in this file is tagged:
**[spec]** vendor/dossier figure, **[assumed]** not yet measured,
**[measured-VM]** measured on the development VM (not the target),
**[derived]** computed from the others. Nothing here was run on a 270K Plus.

## 1. The three groups

| Group | Algorithm | Eligible | Code |
|---|---|---|---|
| 1 | Direct convolution, no lowering buffer | every shape | `src/direct.c` |
| 2 | im2col + GEMM (im2col written straight into packed GEMM panels) | every shape | `src/im2col_gemm.c` |
| 3 | Winograd F(4x4,3x3) and F(2x2,3x3), fused per tile block | 3x3, stride 1 | `src/winograd.c` |

All three share one register-tile design (6 output channels x 16 lanes, 12 ymm
accumulators) and one ISA contract: **AVX2 + FMA only**. Arrow Lake has no AVX-512;
`src/ug_internal.h` refuses to compile if `__AVX512F__` is defined, and
`objdump` of the library shows no `zmm`/mask-register use.

## 2. Target parameters used

| Parameter | Value | Basis |
|---|---|---|
| Cores | 8 Lion Cove P + 16 Skymont E, 24 threads, no SMT | [spec] |
| Clock (all-core) | P 5.4 GHz, E 4.7 GHz | [spec] |
| FP32 FMA | 2 x 256-bit per cycle, latency 4, both core types | [spec] |
| MAC/cycle/core | 2 FMA x 8 lanes = 16 | [derived] |
| Peak FP32 | (8 x 5.4 + 16 x 4.7) GHz x 32 FLOP = 3.79 TFLOP/s | [derived] |
| L1D / L2 (P) | 48 KB / 3 MB | [spec] |
| L2 (E) | 4 MB per 4-core cluster = 1 MB/core | [spec] |
| L3 | 36 MB (overridden at run time by sysfs) | [spec] |
| DRAM | 90 GB/s sustained (78% of 115.2 GB/s peak) | [assumed] |
| DRAM per core | 30 GB/s | [assumed] |

## 3. How each group was built (checklist steps)

### Shared micro-kernel (L8 register tile, D5 registers)
* Accumulators needed = FMA latency x FMA pipes = 4 x 2 = 8 ymm chains. The 6x16
  tile has 12 (>= 8), plus 2 B vectors + 1 broadcast = 15 of 16 ymm.
* Per k step: 12 FMA (6 cycles at 2/cycle) vs 8 loads (2.7 cycles on Lion Cove's 3
  load ports; ~3.3 on Skymont) -> FMA-bound on both core types.
* Measured on the VM: GEMM on 1x1 layers runs at 88.7 GFLOP/s against a measured
  FMA peak of 92-100 GFLOP/s per core **[measured-VM]** (~90-96%).
* Right-edge tiles with <= 8 live columns use a 6x8 variant (saves the dead half).

### Group 1 — direct
* Vector loop = output width (unit stride in NKPQ output), weights broadcast (D3 V1/V2).
* Strided convolution would need a gather. Instead the padded input is
  **phase-split by column** (D3 rule V3): column `ow*stride + s` lives at
  `X[c][ih][s % stride][ow + s/stride]`, so every tap is a contiguous 16-float load.
* The hot loop is one flat loop over C·R·S taps with a precomputed offset table.
  The first version computed `s % stride` inside the loop (integer division);
  removing it made the ResNet stem 1.9x faster (5.10 -> 2.71 ms) **[measured-VM]**.
* Weights packed once at plan time: `[K/6][C][R][S][6]` (D4: never pack weights at run time).

### Group 2 — im2col + GEMM
* BLIS loop order per work item: `kc -> pack B -> ic (MC=240) -> jr (16) -> ir (6)`,
  KC <= 256, NC <= 384. B micro-panel (kc x 16 x 4 B = 16 KB) stays in L1 across
  the ir loop; the MC x KC weight block (<= 240 KB) in L2, which fits a Skymont
  core's 1 MB share.
* D4 lowering choice: the im2col matrix is generated **directly into the packed B
  panels**, one KC x NC block at a time. Arithmetic is exactly im2col + SGEMM,
  memory for the column matrix drops from C·R·S·P·Q·4 B to KC·NC·4 B per thread.
* 1x1 stride-1 convs are already a GEMM: the pack is a plain contiguous copy.
* Stride-2 packing uses an in-register even/odd deinterleave (no gather). This
  took ResNet l2_3x3_s2 from 4.34 to 3.49 ms **[measured-VM]**.

### Group 3 — Winograd
* Transforms: Lavin & Gray points (0, ±1, ±2, ∞). Filter transform `G g Gᵀ` in
  fp64 at plan time; input/output transforms use the factored forms (shared
  sub-expressions, FMA).
* Summed over channels the product is α² independent GEMMs
  `M_ξ[K][T] = U_ξ[K][C] · V_ξ[C][T]` (α = 6 for F4, 4 for F2), run on the shared
  6x16 kernel.
* **Fused**: each work item (image, 16-64 tiles, ≤ 96 output channels) transforms
  its input into V, runs the α² GEMMs into M, and applies the output transform +
  bias + ReLU. V + M ≤ 768 KB so both stay in L2; neither goes to DRAM. Channel
  chunks of 128 bound V for wide layers.
* Transforms are vectorised over 8 tiles; when 8 tiles share a tile row the
  4x8 / 2x8 transposes happen in registers, otherwise a small staging buffer.
* D6: the α² planes of V and M are touched together; their pitch gets one extra
  cache line so they do not share an L1 set. On the VM the measured effect was
  within run-to-run noise (24.6 -> 24.5 ms on VGG conv1_2).
* Cost vs theory: F4 cuts GEMM MACs 4x, but the transforms cost ~21 cycles per
  tile x input channel and ~23 per tile x output channel **[measured-VM, fitted]**.
  On VGG conv1_2 (C = K = 64) the measured end result is 2.1x faster than group 2,
  not 4x. The transform share falls as C and K grow.

## 4. Selection rule: a calibrated cost model

`ug_select_algo()` evaluates, for each eligible algorithm,

    t = smoothmax( Σ_i feature_i · cycles_i / capacity(nthreads),  bytes / bandwidth )

* **features** are unit counts of the exact schedule the executor will run (the
  model calls the same blocking functions as the kernels): MACs including tile
  padding, 6x16 kernel calls, im2col elements on the vector/scalar path, Winograd
  tile x channel transforms (with a penalty for 8-tile groups that straddle rows),
  padded-copy elements.
* **bytes** = activations in + out, plus Winograd transformed weights re-read once
  per tile block when they do not fit in 75% of L3.
* **capacity** = Σ over used cores of their clock (P-cores first, then E-cores).
* smoothmax = (a⁴ + b⁴)^¼, so two algorithms on the same bandwidth floor are still
  ranked by compute.

### Calibration (tools/calibrate.c)
1. Measure the core clock with a dependency-free FMA loop.
2. Time all eligible algorithms, 1 thread, on 64 deterministic synthetic layers.
3. Fit the 12 cycles-per-unit constants by non-negative least squares on relative
   error.
4. **Validate on held-out layers** (VGG-16 + ResNet-50, never used in the fit).

Dev VM result **[measured-VM]** (`results/calibrate_devvm.txt`):
* fit error mean |log(pred/meas)|: 0.141 (hand constants) -> 0.089 (fitted)
* held-out: model picks the measured-fastest algorithm on **29/31** layers; network
  time with model picks = **1.030x** the per-layer optimum.

On the target, run `./build/calibrate -o costmodel.txt` and
`export UGCONV_COSTMODEL=$PWD/costmodel.txt`; or use `UG_ALGO_TUNE`, which times
every eligible algorithm once at plan creation and keeps the fastest.

## 5. Which layers land in which group (dev VM, measured)

Measured-fastest algorithm per layer at 4 threads (`results/bench_4t.csv`):

| Layer type | Winner | Why |
|---|---|---|
| C = 3 stems (VGG conv1_1, ResNet 7x7/2) | **G1 direct** | im2col packing and Winograd transforms cost as much as the few MACs per output; direct has no lowering work |
| 1x1 (all ResNet bottleneck / downsample) | **G2 GEMM** | 1x1 stride 1 is already a GEMM, pack is a copy; kernel runs at ~90% of peak |
| 3x3 stride 2 | **G2** on 2 layers, **G1** on l3_3x3_s2 (1.23x over G2) | Winograd not applicable; G2's stride-2 pack falls back to scalar when rows are narrow |
| 3x3 stride 1, 56x56-224x224, C >= 64 | **G3 F(4x4)** | 4x fewer MACs, transform cost amortised over C and K |
| 3x3 stride 1, 7x7 output | **G3 F(2x2)** | F4 needs 2x2 tiles padded to a 16-wide panel (4x waste); F2 gives exactly 16 tiles |
| 3x3 stride 1, 28x28, C = 512 | F4 = F2 (within 1%) | F4's transformed weights (38 MB) no longer fit L3 |
| 3x3 stride 1, 14x14, C = 512 | **G3 F(2x2)** (1.21x over F4) | 4x4 F4 tiles = 16 tiles, one panel; weight re-reads dominate |

Counts of winners: G1-direct: 3, G2-im2col+gemm: 16, G3-winograd-F2: 2, G3-winograd-F4: 10

## 6. Measured results (dev VM — NOT the target)

Machine: Intel Xeon (Cascade Lake), KVM, 4 vCPU, ~3.0 GHz measured, L2 1 MB,
L3 33 MB. Library built `-march=x86-64-v3` (AVX2+FMA, no AVX-512). OpenBLAS 0.3.26
baseline forced to its AVX2 (Haswell) kernels with `OPENBLAS_CORETYPE=Haswell` so
both sides use the same ISA; OpenMP run with `OMP_WAIT_POLICY=PASSIVE` because
spinning OpenMP threads starved OpenBLAS's pthreads (OpenBLAS 1x1 layers went
from 0.34 ms at 1 thread to 4.0 ms at 4 threads before this). Times are the
minimum over >= 5 runs. The VM is shared; run-to-run variation is ~5-15%.

Times in ms, per layer (min of runs).

#### 1 thread

| layer | x | GMAC | G1 | G2 | G3-F4 | G3-F2 | OpenBLAS | auto pick | measured best | auto vs OpenBLAS |
|---|---|---|---|---|---|---|---|---|---|---|
| vgg16 conv1_1 | 1 | 0.087 | 2.27 | 3.05 | 3.90 | 5.73 | 5.39 | G1-direct | G1-direct | 2.37x |
| vgg16 conv1_2 | 1 | 1.850 | 49.66 | 49.60 | 23.54 | 29.10 | 94.18 | G3-winograd-F4 | G3-winograd-F4 | 4.00x |
| vgg16 conv2_1 | 1 | 0.925 | 22.69 | 23.07 | 10.35 | 13.06 | 33.64 | G3-winograd-F4 | G3-winograd-F4 | 3.25x |
| vgg16 conv2_2 | 1 | 1.850 | 49.92 | 46.61 | 19.00 | 24.95 | 68.06 | G3-winograd-F4 | G3-winograd-F4 | 3.58x |
| vgg16 conv3_1 | 1 | 0.925 | 27.23 | 24.97 | 8.47 | 11.90 | 26.80 | G3-winograd-F4 | G3-winograd-F4 | 3.17x |
| vgg16 conv3_2 | 2 | 1.850 | 54.85 | 47.30 | 18.82 | 24.60 | 53.43 | G3-winograd-F4 | G3-winograd-F4 | 2.84x |
| vgg16 conv4_1 | 1 | 0.925 | 27.69 | 23.90 | 16.90 | 16.00 | 24.04 | G3-winograd-F4 | G3-winograd-F2 | 1.42x |
| vgg16 conv4_2 | 2 | 1.850 | 59.40 | 51.59 | 37.63 | 34.24 | 48.80 | G3-winograd-F4 | G3-winograd-F2 | 1.30x |
| vgg16 conv5_x | 3 | 0.462 | 12.43 | 13.10 | 9.35 | 9.43 | 13.02 | G3-winograd-F4 | G3-winograd-F4 | 1.39x |
| resnet50 stem7x7s2 | 1 | 0.118 | 2.67 | 3.13 | - | - | 5.79 | G1-direct | G1-direct | 2.17x |
| resnet50 l1_1x1_64 | 3 | 0.013 | 0.44 | 0.35 | - | - | 0.34 | G2-im2col+gemm | G2-im2col+gemm | 0.95x |
| resnet50 l1_3x3 | 3 | 0.116 | 3.18 | 3.45 | 1.10 | 1.52 | 5.31 | G3-winograd-F4 | G3-winograd-F4 | 4.82x |
| resnet50 l1_1x1_up | 4 | 0.051 | 1.58 | 1.25 | - | - | 1.34 | G2-im2col+gemm | G2-im2col+gemm | 1.07x |
| resnet50 l2_1x1_red_s1 | 1 | 0.103 | 3.46 | 2.55 | - | - | 2.45 | G2-im2col+gemm | G2-im2col+gemm | 0.96x |
| resnet50 l2_3x3_s2 | 1 | 0.116 | 3.62 | 3.50 | - | - | 4.03 | G1-direct | G2-im2col+gemm | 1.11x |
| resnet50 l2_1x1_up | 4 | 0.051 | 1.52 | 1.13 | - | - | 1.20 | G2-im2col+gemm | G2-im2col+gemm | 1.07x |
| resnet50 l2_ds_1x1_s2 | 1 | 0.103 | 4.11 | 2.48 | - | - | 2.62 | G2-im2col+gemm | G2-im2col+gemm | 1.06x |
| resnet50 l2_1x1_red | 3 | 0.051 | 1.71 | 1.26 | - | - | 1.19 | G2-im2col+gemm | G2-im2col+gemm | 0.94x |
| resnet50 l2_3x3 | 3 | 0.116 | 3.25 | 3.43 | 1.12 | 1.47 | 3.96 | G3-winograd-F4 | G3-winograd-F4 | 3.55x |
| resnet50 l3_1x1_red_s1 | 1 | 0.103 | 3.22 | 2.34 | - | - | 2.27 | G2-im2col+gemm | G2-im2col+gemm | 0.97x |
| resnet50 l3_3x3_s2 | 1 | 0.116 | 3.32 | 3.49 | - | - | 3.35 | G1-direct | G1-direct | 1.01x |
| resnet50 l3_1x1_up | 6 | 0.051 | 1.48 | 1.14 | - | - | 1.16 | G2-im2col+gemm | G2-im2col+gemm | 1.02x |
| resnet50 l3_ds_1x1_s2 | 1 | 0.103 | 3.80 | 2.57 | - | - | 2.50 | G2-im2col+gemm | G2-im2col+gemm | 0.98x |
| resnet50 l3_1x1_red | 5 | 0.051 | 1.53 | 1.18 | - | - | 1.17 | G2-im2col+gemm | G2-im2col+gemm | 0.99x |
| resnet50 l3_3x3 | 5 | 0.116 | 3.03 | 3.49 | 1.30 | 2.04 | 3.33 | G3-winograd-F4 | G3-winograd-F4 | 2.57x |
| resnet50 l4_1x1_red_s1 | 1 | 0.103 | 3.05 | 2.33 | - | - | 2.33 | G2-im2col+gemm | G2-im2col+gemm | 1.00x |
| resnet50 l4_3x3_s2 | 1 | 0.116 | 6.33 | 3.67 | - | - | 3.68 | G2-im2col+gemm | G2-im2col+gemm | 1.00x |
| resnet50 l4_1x1_up | 3 | 0.051 | 2.79 | 1.34 | - | - | 1.38 | G2-im2col+gemm | G2-im2col+gemm | 1.03x |
| resnet50 l4_ds_1x1_s2 | 1 | 0.103 | 6.63 | 3.11 | - | - | 3.09 | G2-im2col+gemm | G2-im2col+gemm | 0.99x |
| resnet50 l4_1x1_red | 2 | 0.051 | 3.00 | 1.38 | - | - | 1.38 | G2-im2col+gemm | G2-im2col+gemm | 1.00x |
| resnet50 l4_3x3 | 2 | 0.116 | 5.94 | 3.81 | 8.35 | 2.53 | 3.49 | G3-winograd-F2 | G3-winograd-F2 | 1.38x |

```
Totals over the selected network(s), weighted by layer multiplicity (19.36 GMAC):
  G1 only 581.55 ms | G2 only 517.04 ms | G3-F4 (else G2) 315.98 ms | G3-F2 (else G2) 333.99 ms
  auto (cost model) 302.23 ms | measured best per layer 294.43 ms | OpenBLAS im2col+sgemm 613.70 ms
  naive loop (1 thread) 46781.70 ms
  auto picked the measured-fastest algorithm on 28/31 layers; auto vs OpenBLAS speed-up 2.03x; effective 128.1 GFLOP/s (direct-conv FLOPs)
```

#### 4 threads

| layer | x | GMAC | G1 | G2 | G3-F4 | G3-F2 | OpenBLAS | auto pick | measured best | auto vs OpenBLAS |
|---|---|---|---|---|---|---|---|---|---|---|
| vgg16 conv1_1 | 1 | 0.087 | 0.76 | 1.23 | 2.10 | 1.67 | 3.00 | G1-direct | G1-direct | 3.95x |
| vgg16 conv1_2 | 1 | 1.850 | 14.11 | 13.52 | 6.52 | 8.29 | 27.62 | G3-winograd-F4 | G3-winograd-F4 | 4.24x |
| vgg16 conv2_1 | 1 | 0.925 | 6.62 | 6.88 | 2.57 | 3.45 | 11.33 | G3-winograd-F4 | G3-winograd-F4 | 4.41x |
| vgg16 conv2_2 | 1 | 1.850 | 13.81 | 13.02 | 5.16 | 6.84 | 28.88 | G3-winograd-F4 | G3-winograd-F4 | 5.59x |
| vgg16 conv3_1 | 1 | 0.925 | 7.73 | 6.90 | 2.56 | 3.22 | 8.08 | G3-winograd-F4 | G3-winograd-F4 | 3.16x |
| vgg16 conv3_2 | 2 | 1.850 | 15.22 | 13.43 | 4.68 | 6.66 | 18.18 | G3-winograd-F4 | G3-winograd-F4 | 3.88x |
| vgg16 conv4_1 | 1 | 0.925 | 8.00 | 7.51 | 3.40 | 3.63 | 9.61 | G3-winograd-F4 | G3-winograd-F4 | 2.83x |
| vgg16 conv4_2 | 2 | 1.850 | 14.93 | 14.84 | 7.72 | 7.75 | 16.71 | G3-winograd-F2 | G3-winograd-F4 | 2.16x |
| vgg16 conv5_x | 3 | 0.462 | 3.50 | 4.69 | 2.97 | 2.46 | 5.41 | G3-winograd-F2 | G3-winograd-F2 | 2.20x |
| resnet50 stem7x7s2 | 1 | 0.118 | 0.91 | 0.93 | - | - | 2.90 | G1-direct | G1-direct | 3.21x |
| resnet50 l1_1x1_64 | 3 | 0.013 | 0.33 | 0.18 | - | - | 0.19 | G2-im2col+gemm | G2-im2col+gemm | 1.07x |
| resnet50 l1_3x3 | 3 | 0.116 | 1.01 | 1.05 | 0.49 | 0.56 | 2.41 | G3-winograd-F4 | G3-winograd-F4 | 4.94x |
| resnet50 l1_1x1_up | 4 | 0.051 | 0.66 | 0.48 | - | - | 0.46 | G2-im2col+gemm | G2-im2col+gemm | 0.96x |
| resnet50 l2_1x1_red_s1 | 1 | 0.103 | 1.47 | 0.75 | - | - | 0.81 | G2-im2col+gemm | G2-im2col+gemm | 1.07x |
| resnet50 l2_3x3_s2 | 1 | 0.116 | 1.41 | 1.29 | - | - | 1.35 | G1-direct | G2-im2col+gemm | 0.96x |
| resnet50 l2_1x1_up | 4 | 0.051 | 0.63 | 0.40 | - | - | 0.59 | G2-im2col+gemm | G2-im2col+gemm | 1.46x |
| resnet50 l2_ds_1x1_s2 | 1 | 0.103 | 1.28 | 0.83 | - | - | 0.95 | G2-im2col+gemm | G2-im2col+gemm | 1.15x |
| resnet50 l2_1x1_red | 3 | 0.051 | 0.82 | 0.47 | - | - | 0.35 | G2-im2col+gemm | G2-im2col+gemm | 0.74x |
| resnet50 l2_3x3 | 3 | 0.116 | 1.03 | 1.24 | 0.60 | 0.66 | 2.26 | G3-winograd-F4 | G3-winograd-F4 | 3.78x |
| resnet50 l3_1x1_red_s1 | 1 | 0.103 | 1.12 | 0.74 | - | - | 0.76 | G2-im2col+gemm | G2-im2col+gemm | 1.02x |
| resnet50 l3_3x3_s2 | 1 | 0.116 | 1.28 | 1.58 | - | - | 1.40 | G1-direct | G1-direct | 1.09x |
| resnet50 l3_1x1_up | 6 | 0.051 | 0.55 | 0.43 | - | - | 0.54 | G2-im2col+gemm | G2-im2col+gemm | 1.26x |
| resnet50 l3_ds_1x1_s2 | 1 | 0.103 | 1.29 | 0.96 | - | - | 1.14 | G1-direct | G2-im2col+gemm | 0.88x |
| resnet50 l3_1x1_red | 5 | 0.051 | 0.68 | 0.47 | - | - | 0.52 | G2-im2col+gemm | G2-im2col+gemm | 1.09x |
| resnet50 l3_3x3 | 5 | 0.116 | 0.90 | 1.53 | 0.67 | 0.74 | 1.33 | G3-winograd-F4 | G3-winograd-F4 | 1.98x |
| resnet50 l4_1x1_red_s1 | 1 | 0.103 | 1.13 | 0.80 | - | - | 1.01 | G2-im2col+gemm | G2-im2col+gemm | 1.27x |
| resnet50 l4_3x3_s2 | 1 | 0.116 | 2.24 | 1.89 | - | - | 2.28 | G2-im2col+gemm | G2-im2col+gemm | 1.21x |
| resnet50 l4_1x1_up | 3 | 0.051 | 0.77 | 0.47 | - | - | 0.66 | G2-im2col+gemm | G2-im2col+gemm | 1.41x |
| resnet50 l4_ds_1x1_s2 | 1 | 0.103 | 2.12 | 1.12 | - | - | 1.87 | G2-im2col+gemm | G2-im2col+gemm | 1.67x |
| resnet50 l4_1x1_red | 2 | 0.051 | 1.03 | 0.54 | - | - | 0.67 | G2-im2col+gemm | G2-im2col+gemm | 1.24x |
| resnet50 l4_3x3 | 2 | 0.116 | 1.63 | 1.87 | 2.18 | 1.10 | 1.75 | G3-winograd-F2 | G3-winograd-F2 | 1.58x |

```
Totals over the selected network(s), weighted by layer multiplicity (19.36 GMAC):
  G1 only 169.65 ms | G2 only 161.75 ms | G3-F4 (else G2) 90.85 ms | G3-F2 (else G2) 96.67 ms
  auto (cost model) 86.01 ms | measured best per layer 85.49 ms | OpenBLAS im2col+sgemm 228.19 ms
  auto picked the measured-fastest algorithm on 28/31 layers; auto vs OpenBLAS speed-up 2.65x; effective 450.1 GFLOP/s (direct-conv FLOPs)
```

#### End-to-end conv stack (every layer instance, own weights, back to back; `-e2e`)

```
End-to-end conv stack, cold weights, min of 3 runs:
  vgg16     | auto 212.99 ms | G1 only 505.82 ms | G2 only 420.26 ms | G3-F4 else G2 221.87 ms | G3-F2 else G2 258.34 ms | OpenBLAS 500.36 ms | auto vs OpenBLAS 2.35x, vs best single group 1.04x
  resnet50  | auto 110.86 ms | G1 only 168.77 ms | G2 only 142.76 ms | G3-F4 else G2 131.01 ms | G3-F2 else G2 114.39 ms | OpenBLAS 145.20 ms | auto vs OpenBLAS 1.31x, vs best single group 1.03x
End-to-end conv stack, cold weights, min of 3 runs:
  vgg16     | auto 66.08 ms | G1 only 132.99 ms | G2 only 122.76 ms | G3-F4 else G2 68.38 ms | G3-F2 else G2 71.12 ms | OpenBLAS 205.73 ms | auto vs OpenBLAS 3.11x, vs best single group 1.03x
  resnet50  | auto 46.96 ms | G1 only 60.06 ms | G2 only 57.67 ms | G3-F4 else G2 50.69 ms | G3-F2 else G2 44.93 ms | OpenBLAS 77.15 ms | auto vs OpenBLAS 1.64x, vs best single group 0.96x
```
(first block 1 thread, second block 4 threads; this run predates the final
re-run above by a few minutes, same code except the ISA check.)

### Accuracy (tests/test_conv.c, 2390 runs incl. 100 random shapes, TUNE and API checks, all pass)
Error per output normalised by Σ|w·x| + |bias| against an fp64 reference, over 17
shapes that hit every tail path (K not a multiple of 6, widths not multiples of
8/16, straddling tile groups, strides 1-3, pads 0-3, 1x1/3x3/5x5/7x7, N=2, 1 and 3
threads, ReLU on/off):

| Algorithm | worst normalised error | tolerance |
|---|---|---|
| G1 direct | 2.1e-7 | 1e-5 |
| G2 im2col+GEMM | 2.1e-7 | 1e-5 |
| G3 F(2x2,3x3) | 1.3e-7 | 3e-5 |
| G3 F(4x4,3x3) | 2.1e-6 | 2e-4 |

F4 loses about 3 bits against direct (≈10x the error); F2 does not.

## 7. What is not verified

* **Nothing ran on a Core Ultra 7 270K Plus.** The VM has no hybrid cores, 2 load
  ports instead of 3, a 1 MB L2 and 4 threads instead of 24.
* E-core behaviour: work is split into >= 3 items per thread with dynamic
  scheduling so slower Skymont cores take fewer items; not measured.
  `ug_cpu_core_type()` (CPUID leaf 1AH) is provided but not used for scheduling.
* Power and energy: not measured (no RAPL in the VM).
* DRAM bandwidth figures in the model are assumptions until measured on the target.
* The cost-model constants are VM-fitted; re-run `tools/calibrate.c` on the target.

## 8. Running on the target

```sh
make ARCH=arrowlake            # clang >= 18 or gcc >= 14 (-march=arrowlake-s)
./build/test_conv              # correctness, must print 2390/2390 passed
./build/calibrate -o costmodel.txt
export UGCONV_COSTMODEL=$PWD/costmodel.txt
export OMP_WAIT_POLICY=PASSIVE OMP_PROC_BIND=close OMP_PLACES=cores
./build/bench_conv -t 24 -csv results/bench_24t.csv     # per layer
./build/bench_conv -t 24 -e2e                           # whole networks
./build/bench_conv -t 8                                  # P-cores only (pin with taskset)
# energy: perf stat -e power/energy-pkg/ ./build/bench_conv -t 24 -e2e
```
