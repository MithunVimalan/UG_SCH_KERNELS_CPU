/* Plan creation, algorithm eligibility, cost model and selection. */
#include <math.h>
#include <limits.h>
#include <omp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "ug_internal.h"

/* Core Ultra 7 270K Plus (see docs/ALGORITHM_SELECTION.md for the basis of each value). */
const ug_machine ug_target = {
    .p_cores = 8, .e_cores = 16,
    .p_ghz = 5.4, .e_ghz = 4.7,          /* spec all-core turbo */
    .fma_mac_per_cycle = 16.0,           /* 2 FMA x 8 lanes, both core types (spec) */
    .dram_gbs = 90.0,                    /* assumed: ~78% of 115.2 GB/s DDR5-7200 peak */
    .core_gbs = 30.0,                    /* assumed */
    .l3_bytes = 36.0 * 1024 * 1024,
    .l2_p_bytes = 3.0 * 1024 * 1024,
    .l2_e_bytes_per_core = 1.0 * 1024 * 1024,
};

/* One-time initialisation of every lazily computed global (F6: these raced when the
 * first plans were created concurrently). */
static ug_machine g_host;
static int g_isa_ok;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static void costmodel_env(void);

static void init_once(void)
{
    ug_cpu_info ci;
    ug_cpu_detect(&ci);
    if (ci.is_arrow_lake_s) {
        g_host = ug_target;
    } else {
        g_host = (ug_machine){.p_cores = omp_get_num_procs(), .e_cores = 0, .p_ghz = 3.0, .e_ghz = 0,
                              .fma_mac_per_cycle = 16.0, .dram_gbs = 20.0, .core_gbs = 10.0,
                              .l3_bytes = 32.0 * 1024 * 1024, .l2_p_bytes = 1024.0 * 1024,
                              .l2_e_bytes_per_core = 0};
    }
    if (ci.l3_bytes > 0) g_host.l3_bytes = (double)ci.l3_bytes;  /* D0: measured beats spec */
    g_isa_ok = ci.avx2 && ci.fma;  /* ug_cpu_detect also checks the OS saves YMM state */
    costmodel_env();
}

const ug_machine *ug_machine_get(void)
{
    pthread_once(&g_once, init_once);
    return &g_host;
}

/* Cost-model constants: core clock cycles per feature unit. Defaults were fitted
 * single-threaded on the dev VM (Xeon Cascade Lake @ ~3.0 GHz, AVX2 build) with
 * tools/calibrate.c; run the same tool on the target to replace them. */
enum {
    F_MAC_DIRECT, F_TILE_DIRECT, F_MAC_GEMM, F_CALL, F_PACK_FAST, F_PACK_SLOW,
    F_MAC_WINO, F_WIN_IN4, F_WIN_OUT4, F_WIN_IN2, F_WIN_OUT2, F_PAD
};
const char *const ug_cm_feature_names[UG_CM_NFEAT] = {
    "direct MAC", "direct 6x16 tile", "gemm MAC", "6x16 kernel call", "im2col elem (vector)",
    "im2col elem (scalar)", "winograd GEMM MAC", "F4 input tile*ch", "F4 output tile*ch",
    "F2 input tile*ch", "F2 output tile*ch", "padded-copy elem"};
static double CM[UG_CM_NFEAT] = {
    0.0682, 12.3, 0.0670, 39.2, 0.190, 5.16,
    0.0729, 21.1, 23.5, 7.45, 3.05, 2.07};

void ug_costmodel_get(double c[UG_CM_NFEAT]) { memcpy(c, CM, sizeof CM); }
void ug_costmodel_set(const double c[UG_CM_NFEAT]) { memcpy(CM, c, sizeof CM); }

static void costmodel_env(void) /* called once, from init_once */
{
    const char *path = getenv("UGCONV_COSTMODEL");
    if (!path) return;
    FILE *f = fopen(path, "r");
    if (!f) return;
    double c[UG_CM_NFEAT];
    int n = 0;
    while (n < UG_CM_NFEAT && fscanf(f, "%lf%*[^\n]", &c[n]) == 1 && isfinite(c[n]) && c[n] >= 0) ++n;
    fclose(f);
    if (n == UG_CM_NFEAT) ug_costmodel_set(c); /* all 12 finite and >= 0, else keep defaults */
}

/* Test seam: make exactly the n-th ug_malloc call (0-based, counted from the call to
 * ug_test_fail_alloc_at) return NULL; n < 0 disables. Used by tests/test_regression.c
 * to prove every allocation failure is reported instead of crashing. */
static long g_alloc_count = 0, g_fail_at = -1;
void ug_test_fail_alloc_at(long n)
{
    __atomic_store_n(&g_alloc_count, 0, __ATOMIC_SEQ_CST);
    __atomic_store_n(&g_fail_at, n, __ATOMIC_SEQ_CST);
}
long ug_test_alloc_count(void) { return __atomic_load_n(&g_alloc_count, __ATOMIC_SEQ_CST); }
/* Persistent variant: every request larger than n bytes fails (0 disables). */
static size_t g_fail_over = 0;
void ug_test_fail_alloc_over(size_t n) { __atomic_store_n(&g_fail_over, n, __ATOMIC_SEQ_CST); }

void *ug_malloc(size_t bytes)
{
    void *p = NULL;
    const long idx = __atomic_fetch_add(&g_alloc_count, 1, __ATOMIC_RELAXED);
    if (idx == __atomic_load_n(&g_fail_at, __ATOMIC_RELAXED)) return NULL;
    const size_t over = __atomic_load_n(&g_fail_over, __ATOMIC_RELAXED);
    if (over && bytes > over) return NULL;
    /* F8: rounding up must not wrap, and no C object may exceed PTRDIFF_MAX */
    if (bytes > (size_t)PTRDIFF_MAX - UG_ALIGN) return NULL;
    if (bytes == 0) bytes = UG_ALIGN;
    if (posix_memalign(&p, UG_ALIGN, (bytes + UG_ALIGN - 1) / UG_ALIGN * UG_ALIGN)) return NULL;
    return p;
}
void ug_free(void *p) { free(p); }

int ug_algo_group(ug_algo a)
{
    switch (a) {
    case UG_ALGO_DIRECT: return 1;
    case UG_ALGO_IM2COL_GEMM: return 2;
    case UG_ALGO_WINOGRAD_F4:
    case UG_ALGO_WINOGRAD_F2: return 3;
    default: return 0;
    }
}

const char *ug_algo_name(ug_algo a)
{
    switch (a) {
    case UG_ALGO_AUTO: return "auto";
    case UG_ALGO_DIRECT: return "G1-direct";
    case UG_ALGO_IM2COL_GEMM: return "G2-im2col+gemm";
    case UG_ALGO_WINOGRAD_F4: return "G3-winograd-F4";
    case UG_ALGO_WINOGRAD_F2: return "G3-winograd-F2";
    case UG_ALGO_TUNE: return "tune";
    }
    return "?";
}

/* Shape contract, see ugconv.h. Every int product the kernels form must fit in int. */
int ug_desc_valid(const ug_conv_desc *d)
{
    if (!d) return 0;
    const long long LIM = 1LL << 24;
    const long long v[] = {d->N, d->C, d->H, d->W, d->K, d->R, d->S, d->stride_h, d->stride_w};
    for (unsigned i = 0; i < sizeof v / sizeof v[0]; ++i)
        if (v[i] < 1 || v[i] > LIM) return 0;
    if (d->pad_h < 0 || d->pad_w < 0 || d->pad_h > LIM || d->pad_w > LIM) return 0;
    const long long P = ug_out_h(d), Q = ug_out_w(d);
    if (P < 1 || Q < 1) return 0;                     /* kernel does not fit (F2) */
    if ((long long)d->C * d->R * d->S > INT_MAX) return 0;
    if (P * Q > INT_MAX || (long long)d->H * d->W > INT_MAX) return 0;
    return 1;
}
#define desc_valid ug_desc_valid

int ug_algo_eligible(const ug_conv_desc *d, ug_algo a)
{
    if (!desc_valid(d)) return 0;
    switch (a) {
    case UG_ALGO_DIRECT:
    case UG_ALGO_IM2COL_GEMM: return 1;
    case UG_ALGO_WINOGRAD_F4:
    case UG_ALGO_WINOGRAD_F2:
        return d->R == 3 && d->S == 3 && d->stride_h == 1 && d->stride_w == 1;
    default: return 0;
    }
}

int ug_costmodel_features(const ug_conv_desc *d, ug_algo a, int nthreads, double f[UG_CM_NFEAT], double *bytes)
{
    memset(f, 0, UG_CM_NFEAT * sizeof(double));
    if (!ug_algo_eligible(d, a)) return -1;
    const ug_machine *mc = ug_machine_get();
    if (nthreads <= 0) nthreads = mc->p_cores + mc->e_cores;
    const double N = d->N, C = d->C, K = d->K, R = d->R, S = d->S, CRS = C * R * S;
    const int P = ug_out_h(d), Q = ug_out_w(d);
    const double PQ = (double)P * Q;
    const double Kp = ug_round_up(d->K, UG_MR);
    *bytes = 4.0 * N * (C * d->H * d->W + K * PQ);

    switch (a) {
    case UG_ALGO_DIRECT: {
        double Q16 = ug_round_up(Q, 16);
        f[F_MAC_DIRECT] = N * Kp * CRS * P * Q16;
        f[F_TILE_DIRECT] = N * (Kp / UG_MR) * P * (Q16 / 16);
        f[F_PAD] = N * C * (d->H + 2 * d->pad_h) * (Q16 * d->stride_w + S);
        *bytes += 4.0 * Kp * CRS;
        break;
    }
    case UG_ALGO_IM2COL_GEMM: {
        ug_gemm_blk bk;
        ug_gemm_blocking(d, nthreads, &bk);
        double PQ16 = ug_round_up(P * Q, UG_NR), nkc = ceil(CRS / bk.KC);
        f[F_MAC_GEMM] = N * Kp * CRS * PQ16;
        f[F_CALL] = N * (Kp / UG_MR) * (PQ16 / UG_NR) * nkc;
        int pointwise = R == 1 && S == 1 && d->stride_h == 1 && d->stride_w == 1 && d->pad_h == 0 && d->pad_w == 0;
        double fast = pointwise ? 1.0
                    : d->stride_w == 1 ? fmax(0.0, (Q - 15.0) / Q)
                    : d->stride_w == 2 ? fmax(0.0, (Q - 17.0 - d->pad_w) / Q) : 0.0;
        f[F_PACK_FAST] = N * CRS * PQ * bk.mblk * fast;
        f[F_PACK_SLOW] = N * CRS * PQ * bk.mblk * (1 - fast);
        *bytes += 4.0 * Kp * CRS;
        break;
    }
    case UG_ALGO_WINOGRAD_F4:
    case UG_ALGO_WINOGRAD_F2: {
        const int m = a == UG_ALGO_WINOGRAD_F4 ? 4 : 2;
        const double a2 = (m + 2.0) * (m + 2.0);
        ug_wino_blk bk;
        ug_wino_blocking(d, m, nthreads, &bk);
        double T = bk.T;
        double T16 = (bk.tblk - 1.0) * bk.TB + ug_round_up(bk.T - (bk.tblk - 1) * bk.TB, UG_NR);
        f[F_MAC_WINO] = N * a2 * Kp * C * T16;
        f[F_CALL] = N * a2 * (Kp / UG_MR) * (T16 / UG_NR) * ceil(C / bk.CC);
        /* 8-tile groups inside one tile row take the register-transpose path */
        double fast = bk.Tw >= 8 ? (double)(bk.Tw / 8 * 8) / bk.Tw : 0.0;
        double stage = fast + (1 - fast) * 1.5;
        f[m == 4 ? F_WIN_IN4 : F_WIN_IN2] = N * C * T * bk.kblk * stage;
        f[m == 4 ? F_WIN_OUT4 : F_WIN_OUT2] = N * K * T * stage;
        f[F_PAD] = N * C * (bk.Th * m + 2.0) * (bk.Tw * m + 8.0);
        /* Transformed weights (alpha^2/9 x the raw size) are re-read once per tile
         * block. If they do not fit L3 every re-read comes from DRAM. */
        double ubytes = a2 * Kp * C * 4.0;
        if (ubytes > 0.75 * mc->l3_bytes) *bytes += ubytes * N * bk.tblk;
        break;
    }
    default: return -1;
    }
    return 0;
}

/* Effective core-cycles per second for nt threads: P-cores first, then E-cores. */
static double capacity(const ug_machine *m, int nt)
{
    int np = ug_min(nt, m->p_cores), ne = ug_min(ug_max(nt - np, 0), m->e_cores);
    return (np * m->p_ghz + ne * m->e_ghz) * 1e9;
}

double ug_algo_cost(const ug_conv_desc *d, ug_algo a, int nthreads)
{
    double f[UG_CM_NFEAT], bytes;
    const ug_machine *mc = ug_machine_get();
    if (nthreads <= 0) nthreads = mc->p_cores + mc->e_cores;
    if (ug_costmodel_features(d, a, nthreads, f, &bytes)) return -1.0;
    double cyc = 0;
    for (int i = 0; i < UG_CM_NFEAT; ++i) cyc += f[i] * CM[i];
    double t_comp = cyc / capacity(mc, nthreads);
    double t_mem = bytes / (fmin(mc->dram_gbs, nthreads * mc->core_gbs) * 1e9);
    /* smooth roofline max: ~max(t_comp, t_mem), but when two algorithms sit on the
     * same bandwidth floor the one with less compute still ranks first */
    return pow(pow(t_comp, 4) + pow(t_mem, 4), 0.25);
}

ug_algo ug_select_algo(const ug_conv_desc *d, int nthreads)
{
    static const ug_algo cand[] = {UG_ALGO_WINOGRAD_F4, UG_ALGO_WINOGRAD_F2, UG_ALGO_IM2COL_GEMM, UG_ALGO_DIRECT};
    ug_algo best = UG_ALGO_IM2COL_GEMM;
    double bt = 1e300;
    for (unsigned i = 0; i < sizeof cand / sizeof cand[0]; ++i) {
        double t = ug_algo_cost(d, cand[i], nthreads);
        if (t >= 0 && t < bt) { bt = t; best = cand[i]; }
    }
    return best;
}

static ug_conv_plan *plan_create_fixed(const ug_conv_desc *d, const float *w, const float *bias,
                                       int relu, ug_algo algo, int nt)
{
    ug_conv_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->d = *d;
    p->algo = algo;
    p->relu = relu;
    p->nthreads = nt;
    p->P = ug_out_h(d);
    p->Q = ug_out_w(d);
    p->Kp = ug_round_up(d->K, UG_MR);
    p->bias = ug_malloc((size_t)p->Kp * sizeof(float));
    if (!p->bias) { free(p); return NULL; }
    for (int k = 0; k < p->Kp; ++k) p->bias[k] = (bias && k < d->K) ? bias[k] : 0.f;
    int rc = -1;
    switch (algo) {
    case UG_ALGO_DIRECT: rc = ug_direct_prepare(p, w); break;
    case UG_ALGO_IM2COL_GEMM: rc = ug_gemm_prepare(p, w); break;
    case UG_ALGO_WINOGRAD_F4: p->wino_m = 4; rc = ug_wino_prepare(p, w); break;
    case UG_ALGO_WINOGRAD_F2: p->wino_m = 2; rc = ug_wino_prepare(p, w); break;
    default: break;
    }
    if (rc) { ug_conv_plan_destroy(p); return NULL; }
    return p;
}

ug_conv_plan *ug_conv_plan_create(const ug_conv_desc *d, const float *w, const float *bias,
                                  int relu, ug_algo algo, int nthreads)
{
    if (!desc_valid(d) || !w) return NULL;
    pthread_once(&g_once, init_once);
    if (!g_isa_ok) return NULL; /* kernels are AVX2+FMA */
    if (nthreads <= 0) nthreads = omp_get_max_threads();
    if (algo == UG_ALGO_AUTO) algo = ug_select_algo(d, nthreads);
    if (algo != UG_ALGO_TUNE) {
        if (!ug_algo_eligible(d, algo)) return NULL;
        return plan_create_fixed(d, w, bias, relu, algo, nthreads);
    }

    /* TUNE: time each eligible algorithm on a synthetic input, keep the fastest. */
    size_t in_n = (size_t)d->N * d->C * d->H * d->W, out_n = (size_t)d->N * d->K * ug_out_h(d) * ug_out_w(d);
    float *x = ug_malloc(in_n * sizeof(float)), *y = ug_malloc(out_n * sizeof(float));
    if (!x || !y) { ug_free(x); ug_free(y); return NULL; }
    for (size_t i = 0; i < in_n; ++i) x[i] = (float)((i * 2654435761u) % 1000) * 1e-3f - 0.5f;
    static const ug_algo cand[] = {UG_ALGO_DIRECT, UG_ALGO_IM2COL_GEMM, UG_ALGO_WINOGRAD_F4, UG_ALGO_WINOGRAD_F2};
    ug_conv_plan *best = NULL;
    double bt = 1e300;
    for (unsigned i = 0; i < sizeof cand / sizeof cand[0]; ++i) {
        if (!ug_algo_eligible(d, cand[i])) continue;
        ug_conv_plan *p = plan_create_fixed(d, w, bias, relu, cand[i], nthreads);
        if (!p) continue;
        double t = 1e300;
        int rc = ug_conv_execute(p, x, y); /* warm-up */
        for (int rep = 0; rep < 3 && rc == 0; ++rep) {
            double t0 = omp_get_wtime();
            rc = ug_conv_execute(p, x, y);
            double dt = omp_get_wtime() - t0;
            if (dt < t) t = dt;
        }
        if (rc) { ug_conv_plan_destroy(p); continue; } /* F11: never pick a failing candidate */
        if (t < bt) { bt = t; ug_conv_plan_destroy(best); best = p; }
        else ug_conv_plan_destroy(p);
    }
    ug_free(x);
    ug_free(y);
    return best;
}

ug_algo ug_conv_plan_algo(const ug_conv_plan *p) { return p ? p->algo : UG_ALGO_AUTO; }

int ug_conv_execute(const ug_conv_plan *p, const float *in, float *out)
{
    if (!p || !in || !out) return -1;
    switch (p->algo) {
    case UG_ALGO_DIRECT: return ug_direct_execute(p, in, out);
    case UG_ALGO_IM2COL_GEMM: return ug_gemm_execute(p, in, out);
    case UG_ALGO_WINOGRAD_F4:
    case UG_ALGO_WINOGRAD_F2: return ug_wino_execute(p, in, out);
    default: return -1;
    }
}

void ug_conv_plan_destroy(ug_conv_plan *p)
{
    if (!p) return;
    ug_free(p->bias);
    ug_free(p->wdir);
    ug_free(p->wgemm);
    ug_free(p->wwino);
    free(p);
}
