#ifndef UG_INTERNAL_H
#define UG_INTERNAL_H

#include <stddef.h>
#include "ugconv.h"

#if !defined(__AVX2__) || !defined(__FMA__)
#error "ugconv needs -mavx2 -mfma (use -march=arrowlake-s on the target)"
#endif
/* Only the int8 files of the dev-VM build may see AVX-512 (they use EVEX VNNI on
 * ymm as a stand-in for AVX-VNNI); they define UG_DEV_EVEX_VNNI. */
#if defined(__AVX512F__) && !defined(UG_DEV_EVEX_VNNI)
#error "ugconv targets Arrow Lake: build without AVX-512 (e.g. -march=x86-64-v3 or -march=arrowlake-s)"
#endif

/* GEMM register tile: 6 rows (output channels) x 16 columns (pixels/tiles).
 * 12 ymm accumulators >= FMA latency 4 x 2 pipes = 8 independent chains. */
#define UG_MR 6
#define UG_NR 16

#define UG_ALIGN 64
void *ug_malloc(size_t bytes);
void ug_test_fail_alloc_at(long n); /* test seam, see plan.c */
int ug_desc_valid(const ug_conv_desc *d);
long ug_test_alloc_count(void);
void ug_test_fail_alloc_over(size_t n);
void ug_free(void *p);

static inline int ug_ceil_div(int a, int b) { return (a + b - 1) / b; }
static inline int ug_round_up(int a, int b) { return ug_ceil_div(a, b) * b; }
static inline int ug_min(int a, int b) { return a < b ? a : b; }
static inline int ug_max(int a, int b) { return a > b ? a : b; }

/* C[mr x nr] (+)= A_panel(kc x 6) * B_panel(kc x 16).
 * accumulate: 0 -> overwrite C, 1 -> add to C.
 * If bias != NULL (6 values) it is added, then relu applied if relu != 0. */
void ug_kernel_6x16(int kc, const float *A, const float *B, float *C, long ldc,
                    int mr, int nr, int accumulate, const float *bias, int relu);

/* Machine model used by the cost model (Core Ultra 7 270K Plus). */
typedef struct {
    int p_cores, e_cores;
    double p_ghz, e_ghz;
    double fma_mac_per_cycle;   /* fp32 MAC/cycle/core: 2 FMA x 8 lanes = 16 */
    double dram_gbs;            /* sustained all-core, not peak */
    double core_gbs;            /* sustained DRAM bandwidth of one core */
    double l3_bytes;
    double l2_p_bytes, l2_e_bytes_per_core;
} ug_machine;
extern const ug_machine ug_target;
/* ug_target on an Arrow Lake-S part, otherwise a description of the host (sysfs + OpenMP). */
const ug_machine *ug_machine_get(void);

struct ug_conv_plan {
    ug_conv_desc d;
    ug_algo algo;
    int relu;
    int nthreads;
    int P, Q;
    float *bias;       /* K floats, zero if none (padded to multiple of 6) */
    /* group 1 */
    float *wdir;       /* [Kp/6][C][R][S][6] */
    /* group 2 */
    float *wgemm;      /* [Kp/6][C*R*S][6] */
    /* group 3 */
    int wino_m;        /* 4 or 2 */
    float *wwino;      /* [alpha^2][Kp/6][C][6] */
    int Kp;            /* K rounded up to multiple of 6 */
};

/* Blocking decisions, shared by the executors and the cost model so the model
 * prices exactly the schedule that runs. */
typedef struct { int KC, NCb, nblk, mblk, pan_per_mblk; } ug_gemm_blk;
void ug_gemm_blocking(const ug_conv_desc *d, int nt, ug_gemm_blk *b);
typedef struct { int m, Th, Tw, T, TB, KB, CC, tblk, kblk; } ug_wino_blk;
void ug_wino_blocking(const ug_conv_desc *d, int m, int nt, ug_wino_blk *b);

int ug_direct_prepare(ug_conv_plan *p, const float *w);
int ug_direct_execute(const ug_conv_plan *p, const float *in, float *out);
int ug_gemm_prepare(ug_conv_plan *p, const float *w);
int ug_gemm_execute(const ug_conv_plan *p, const float *in, float *out);
int ug_wino_prepare(ug_conv_plan *p, const float *w);
int ug_wino_execute(const ug_conv_plan *p, const float *in, float *out);

#endif
