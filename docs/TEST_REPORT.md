# Test report — line-by-line review of ugconv

Scope: every line of `src/*.c`, `src/*.h`, `include/ugconv.h` (≈2,900 lines, fp32 and
int8). Method: read each line → log each suspected defect → write a test that **fails on
the unfixed code** → fix → the test passes, all other suites stay green → check that the
test really protects the fix by reverting the fix (mutation). Everything ran on the dev VM
(Xeon Cascade Lake, 4 vCPU); the Arrow Lake build was compiled and disassembled only.

## 1. Defects found in the library

"Before" is the output of `tests/test_regression.c` on the unfixed code
(`results/regression_before_fix.txt`: **5/32 checks passed**); "after" is
`results/regression_after_fix.txt` (**42/42**, more checks were added later).

| # | Sev. | Location | Defect | Reproduction / evidence before the fix | Fix |
|---|---|---|---|---|---|
| F1 | **High** | `q_plan.c` eligibility, `zcorr` | int8 accepted any C·R·S. The centred int32 accumulator overflows once 255·128·C·R·S ≥ 2³¹, i.e. C·R·S > 65,793; `in_zp * s` was signed-overflow UB | C=7400, 3×3: centre output **2,121,143,296**, exact **−2,173,824,000** (Q1 and Q2) | reject C·R·S > `UG_Q_MAX_CRS` (65,793); `zcorr` in 64-bit. At the bound (65,790) results are exact |
| F2 | **Med-High** | `ugconv.h` `ug_out_h/w`, `plan.c` validation | C division truncates toward zero, so H+2p < R gave output size 1 for stride ≥ 2 and the shape was accepted; the direct kernel then reads past its padded buffer | H=2, R=3, stride 2: `ug_out_h` = **1**, accepted by **5** plan kinds | `ug_out_*` return 0 when the kernel does not fit; validation requires H+2p ≥ R |
| F3 | Medium | `q_plan.c` u8 epilogue, `execute_u8` | `cvtps_epi32` returns INT_MIN for \|y/scale\| ≥ 2³¹, so a huge positive value became **0** instead of 255; `out_zp` unchecked | y = +3.2e9 → **0** (want 255); `out_zp` = 300 accepted | clamp in float to ±512 before conversion (identical to the reference for every input); `out_zp` must be in [0,255] |
| F4 | Medium | `kernel_6x16.c:81` vs `:102`, `direct.c`, `winograd.c`, `q_plan.c` | ReLU as `max_ps(v, 0)` turns NaN into 0 (MAXPS returns its 2nd operand on NaN); the scalar edge-tile path kept NaN, so the **same tensor** had NaN in edge tiles and 0 in full tiles | NaN input pixel: 54–63 receptive-field outputs lost the NaN (every fp32 algorithm); int8 NaN bias: 735/735 lost | `max_ps(0, v)` everywhere: NaN propagates (like PyTorch `relu`) and `-0.0` matches the scalar path |
| F5 | Medium | `im2col_gemm.c:146`, `winograd.c:340`, `q_gemm.c:169`, `q_winograd.c` V/M | per-thread scratch allocations inside `omp parallel` never checked → NULL dereference on out-of-memory | fault injection on each allocation: **148 of 241** injected failures crashed | check, skip the work, return −1 |
| F6 | Low-Med | `plan.c` `ug_machine_get`, `costmodel_env`, `isa_ok`; `q_plan.c` same | lazily initialised globals written without synchronisation | ThreadSanitizer: data races at `plan.c:25/29/37/63/64` with 8 threads creating plans | `pthread_once` for all one-time init |
| F7 | Low | `cpu.c` | CPUID leaves 7 / 7.1 / 0x80000002-4 / 0x1A read without checking the maximum leaf; AVX2 trusted without checking the OS enables YMM state (XGETBV) → would SIGILL instead of refusing | not reproducible on this VM (all leaves present, OS support on) | max-leaf checks; `OSXSAVE` + `XCR0` bits 1,2 (and 5-7 for the EVEX dev build) |
| F8 | Low | `plan.c` `ug_malloc` | `(bytes + 63) / 64 * 64` wraps for sizes near `SIZE_MAX` → tiny allocation | `ug_malloc(SIZE_MAX-8)` returned a **valid pointer** | reject sizes > `PTRDIFF_MAX` − 64 (also: no C object may exceed `PTRDIFF_MAX`) |
| F9 | Low | `plan.c` validation | int products C·R·S, P·Q, H·W, H+2p not bounded; e.g. C·R·S = 2³²+5 wraps to 5 | 2 of 3 overflow shapes **accepted** | contract: dims ≤ 2²⁴, products ≤ `INT_MAX` (documented in `ugconv.h`) |
| F10 | Low | `q_gemm.c:42`, `q_direct.c:41`, `q_winograd.c:129` | `int8_t`/`int16_t` weights read through `int32_t*`: strict-aliasing UB | not observable with current compilers | `may_alias` 32-bit type |
| F11 | Low | `plan.c` / `q_plan.c` TUNE | failed executes were timed; a candidate that always fails is "fastest" | (shown by the persistent-failure test below) | only candidates whose execute succeeds are timed |
| F12 | Doc/API | `plan.c` validation | `pad < kernel` silently required, undocumented; padding ≥ kernel is legal elsewhere | 4 valid shapes **rejected** | restriction lifted; all 7 algorithms verified on pad ≥ kernel |
| — | Hardening | `plan.c`, `q_plan.c` cost-model files | a `nan` / negative / truncated `UGCONV_COSTMODEL` file was loaded silently (clang-tidy cert-err34) | manual check | a file is used only if every value is finite and ≥ 0 |
| — | Hardening | `im2col_gemm.c` blocking | clang analyzer: division by zero if P·Q = 0 (unreachable after validation, but the function is shared with the cost model) | — | `NCb ≥ 16` guard |

Documented, not changed: Winograd computes each output tile from its whole input patch,
so (a) a NaN/Inf input can make all outputs of its tile NaN, and (b) an output whose
receptive field is all padding gets rounding noise (≈1e-7 relative to the tile) instead
of an exact 0. Both are inherent to the algorithm; `ugconv.h` states them.

## 2. Defects found in my own tests (a test engineer's tests are code too)

| Problem | Found by | Consequence | Fix |
|---|---|---|---|
| fault-injection children hung | first F5 run timed out | GNU OpenMP is not fork-safe once its pool exists | each case runs in a fresh process (`popen` → exec) |
| F12 only tested with ReLU on | widened fp32 suite failed on the same shapes | ReLU zeroed negative garbage and hid it | F12 runs with and without ReLU |
| fp32 error metric divided Winograd error by the output's own Σ\|w·x\| | 6.56e23 "errors" on pad ≥ kernel | false failures where the true value is 0 | Winograd normalised by the tile's largest Σ\|w·x\|, the quantity its error bound scales with; direct/GEMM unchanged |
| random-shape generators skipped H+2p < R | review of F2 | the F2 bug could never be hit | generators now produce such shapes and require rejection |
| F9 shapes failed several checks at once | mutation test: removing the C·R·S check went **unnoticed** | test protected nothing | one shape per check (C·R·S only, P·Q only, H·W only) |
| F11 fault injection was one-shot | mutation test: F11 revert **unnoticed** | a transient failure is harmless | persistent "fail allocations > N bytes" seam; TUNE must pick a working algorithm |
| harness leaked its own buffers in child mode | LeakSanitizer, intermittently | noise | freed |
| `ug_malloc(SIZE_MAX/2)` test | ASan aborts instead of returning NULL; Valgrind "fishy size" | tool-specific failure | `ASAN_OPTIONS=allocator_may_return_null=1`; test uses legal sizes; `PTRDIFF_MAX` guard |

## 3. Tool results on the final code

| Check | Result | Evidence |
|---|---|---|
| fp32 suite: 30 fixed shapes + 100 random (C 1-80, K 1-40, H/W 1-30, R/S 1-7, strides 1-3 per axis, pad 0..R+1), 1 and 3 threads, ReLU on/off, AUTO/TUNE, rejection of non-fitting shapes | **2390/2390** | `results/final_tests.txt` |
| int8 suite, bit-exact (s32 / f32 / u8) vs integer reference | **3706/3706** | same |
| regression suite (F1-F12, B1-B2) | **42/42** (was 5/32) | `results/regression_*` |
| stress: bitwise-identical output for 1/2/3/4/7/16 threads, repeated runs, one plan executed from 4 threads at once | **34/34** | `results/stress.txt` |
| AddressSanitizer + UndefinedBehaviorSanitizer, all four suites | clean | `results/sanitize.txt` |
| ThreadSanitizer, concurrent first use | **0** warnings (before: races in F6) | `results/tsan_after_fix.txt` |
| Valgrind memcheck (uninitialised reads, invalid access, leaks), pure-AVX2 build | **0 errors** in all four suites | `results/valgrind_summary.txt` |
| gcc `-fanalyzer` | 3 "uninitialised" warnings in the F(2x2) transform helpers: false positives (fixed-trip loops fill the arrays; Valgrind runs these paths with 0 errors) | — |
| clang-tidy (clang-analyzer-\*, bugprone-\*, cert-\*) | 75 warnings triaged: 48 implicit-widening (all operands small or already 64-bit), 10 intended narrowing, 1 intended integer division, 11 Annex-K "use memcpy_s" (not applicable), 2 `fscanf` → hardened, 1 division by zero → hardened | — |
| Mutation test: 13 kernel mutants + one revert of every fix | **22/22 caught** | `results/mutation_test.txt` |
| Arrow Lake target build (clang 18, `-march=arrowlake-s`) | compiles; 150 `{vex} vpdpbusd` + 24 `{vex} vpdpwssd`; 0 zmm / mask / ymm16-31 | `results/target_build_check.txt` |
| Performance after fixes | ResNet-50, 1 thread, auto: 87.6 ms (before: 88.4 ms) | — |

Branch coverage (`gcov -b`, all four suites; `results/coverage_branches.txt`):

| File | Lines | Branches taken |
|---|---|---|
| kernel_6x16.c, direct.c, winograd.c, q_gemm.c, q_direct.c, q_winograd.c | 100% | 100% |
| im2col_gemm.c | 100% | 98.9% |
| plan.c | 98.1% | 92.8% |
| q_plan.c | 98.7% | 91.9% |
| cpu.c | 76.3% | 35.2% |

The 48 remaining untaken branch lines (`results/coverage_untaken.txt`) are: CPUID/XGETBV
outcomes and Arrow Lake detection (depend on the CPU), "ISA not supported" refusals,
`calloc` failure of the plan struct itself, unreachable `switch` defaults, and one
short-circuit sub-condition of the 1x1 test in the im2col packer.

## 4. Residual risk (what this campaign could not show)

* Nothing executed on a Core Ultra 7 270K Plus: AVX-VNNI (VEX) code was compiled and
  disassembled only; hybrid P/E scheduling, 24 threads and the Arrow Lake cache sizes
  were not exercised. First thing to run on the target: `make ARCH=arrowlake test`.
* F7 (CPUID / OS support) is fixed by inspection; no machine here lacks the features.
* Valgrind cannot decode AVX-512, so it ran on the AVX2-emulation int8 backend; the EVEX
  backend was covered by ASan/UBSan and by bit-exact equality with the emulation.
* Sizes beyond the VM's memory (near the 2³¹ limits) were tested for rejection only.

## 5. Re-running

```sh
make test                          # 2390 + 3706 + 42 + 34 checks
make sanitize                      # all four suites under ASan + UBSan
make tsan                          # concurrent first use under ThreadSanitizer
python3 tools/mutation_test.py     # 22 mutants, all must be caught
make VNNI=emu EXTRA_CFLAGS=-g build/test_conv && valgrind ./build/test_conv
```
