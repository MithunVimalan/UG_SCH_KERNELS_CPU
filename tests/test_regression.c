/* Regression tests, one per finding of the line-by-line review (docs/TEST_REPORT.md).
 * Every test was written BEFORE its fix and failed on the unfixed code.
 *
 *   F1  int8 reduction length beyond the exact-int32 bound must be rejected
 *   F2  shapes where the kernel does not fit (H + 2p < R) must be rejected
 *   F3  u8 requantisation must saturate, not wrap, for |y/scale| >= 2^31
 *   F4  ReLU must propagate NaN identically in full and edge tiles, all algorithms
 *   F5  every allocation failure must return an error, never crash or return garbage
 *       (fault injection on each ug_malloc call, each case in a forked child)
 *   F8  ug_malloc size rounding must not wrap
 *   F9  int products (C*R*S, P*Q) that overflow must be rejected
 *   F11 TUNE must not select a candidate whose execute failed (covered by F5)
 *   F12 padding >= kernel size is a valid shape and must work in every algorithm
 */
#include <math.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../src/ug_internal.h"

static int fails = 0, total = 0;
static void report(const char *id, int ok, const char *fmt, const char *detail)
{
    ++total;
    if (!ok) ++fails;
    printf("%s %-4s %s%s%s\n", ok ? "PASS" : "FAIL", id, fmt, detail && *detail ? " -- " : "", detail ? detail : "");
}

static void ref_f64(const ug_conv_desc *d, const float *x, const float *w, const float *b, int relu, double *y)
{
    const int P = ug_out_h(d), Q = ug_out_w(d);
    for (int n = 0; n < d->N; ++n)
        for (int k = 0; k < d->K; ++k)
            for (int oh = 0; oh < P; ++oh)
                for (int ow = 0; ow < Q; ++ow) {
                    double s = b ? b[k] : 0;
                    for (int c = 0; c < d->C; ++c)
                        for (int r = 0; r < d->R; ++r)
                            for (int q = 0; q < d->S; ++q) {
                                int ih = oh * d->stride_h - d->pad_h + r, iw = ow * d->stride_w - d->pad_w + q;
                                if (ih < 0 || ih >= d->H || iw < 0 || iw >= d->W) continue;
                                s += (double)w[((k * d->C + c) * d->R + r) * d->S + q] *
                                     x[(((size_t)n * d->C + c) * d->H + ih) * d->W + iw];
                            }
                    y[(((size_t)n * d->K + k) * P + oh) * Q + ow] = relu && s < 0 ? 0 : s;
                }
}

static void ref_i64(const ug_conv_desc *d, const uint8_t *x, const int8_t *w, int zp, int64_t *y)
{
    const int P = ug_out_h(d), Q = ug_out_w(d);
    for (int n = 0; n < d->N; ++n)
        for (int k = 0; k < d->K; ++k)
            for (int oh = 0; oh < P; ++oh)
                for (int ow = 0; ow < Q; ++ow) {
                    int64_t s = 0;
                    for (int c = 0; c < d->C; ++c)
                        for (int r = 0; r < d->R; ++r)
                            for (int q = 0; q < d->S; ++q) {
                                int ih = oh * d->stride_h - d->pad_h + r, iw = ow * d->stride_w - d->pad_w + q;
                                if (ih < 0 || ih >= d->H || iw < 0 || iw >= d->W) continue;
                                s += (int64_t)(x[(((size_t)n * d->C + c) * d->H + ih) * d->W + iw] - zp) *
                                     w[(((size_t)k * d->C + c) * d->R + r) * d->S + q];
                            }
                    y[(((size_t)n * d->K + k) * P + oh) * Q + ow] = s;
                }
}

static size_t out_elems(const ug_conv_desc *d) { return (size_t)d->N * d->K * ug_out_h(d) * ug_out_w(d); }
static size_t in_elems(const ug_conv_desc *d) { return (size_t)d->N * d->C * d->H * d->W; }

static const ug_algo FALGOS[] = {UG_ALGO_DIRECT, UG_ALGO_IM2COL_GEMM, UG_ALGO_WINOGRAD_F4, UG_ALGO_WINOGRAD_F2};
static const ug_qalgo QALGOS[] = {UG_QALGO_DIRECT, UG_QALGO_IM2COL_GEMM, UG_QALGO_WINOGRAD_F2};

/* ---------------- F1 ---------------- */
static void test_f1(void)
{
    char msg[256];
    for (int pass = 0; pass < 2; ++pass) {
        /* pass 0: C*R*S = 66600 > 65793 (must be rejected); pass 1: 65790 (must be exact) */
        ug_conv_desc d = {1, pass ? 7310 : 7400, 3, 3, 1, 3, 3, 1, 1, 1, 1};
        uint8_t *x = malloc(in_elems(&d));
        int8_t *w = malloc((size_t)d.C * 9);
        memset(x, 255, in_elems(&d));
        memset(w, 0x80, (size_t)d.C * 9); /* -128 */
        float s = 1.f;
        int64_t ref[9];
        int32_t y[9];
        ref_i64(&d, x, w, 0, ref);
        for (int a = 0; a < 2; ++a) {
            ug_qconv_plan *p = ug_qconv_plan_create(&d, (const signed char *)w, &s, NULL, 0, 1.f, 0, QALGOS[a], 1);
            int ok;
            if (pass == 0) {
                ok = p == NULL;
                if (p) {
                    ug_qconv_execute_s32(p, x, y);
                    snprintf(msg, sizeof msg, "C*R*S=66600 accepted by %s; centre = %d, exact = %lld",
                             ug_qalgo_name(QALGOS[a]), y[4], (long long)ref[4]);
                } else snprintf(msg, sizeof msg, "C*R*S=66600 rejected by %s", ug_qalgo_name(QALGOS[a]));
            } else {
                ok = p != NULL;
                if (p) {
                    ug_qconv_execute_s32(p, x, y);
                    for (int i = 0; i < 9; ++i) ok &= y[i] == ref[i];
                    snprintf(msg, sizeof msg, "C*R*S=65790 (bound) %s centre %d exact %lld", ug_qalgo_name(QALGOS[a]),
                             y[4], (long long)ref[4]);
                } else snprintf(msg, sizeof msg, "C*R*S=65790 rejected by %s (should be accepted)", ug_qalgo_name(QALGOS[a]));
            }
            report("F1", ok, pass ? "int8 exactness at the bound" : "int8 overflow beyond the bound", msg);
            ug_qconv_plan_destroy(p);
        }
        free(x); free(w);
    }
}

/* ---------------- F2 ---------------- */
static void test_f2(void)
{
    char msg[256];
    const ug_conv_desc bad[] = {
        {1, 1, 2, 2, 1, 3, 3, 2, 2, 0, 0},   /* H + 2p = 2 < R = 3, stride 2 -> truncation gives P = 1 */
        {1, 2, 4, 9, 3, 7, 3, 3, 1, 1, 1},   /* H + 2p = 6 < R = 7, stride 3 */
    };
    float w[256] = {0}, s[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    signed char wq[256] = {0};
    for (unsigned i = 0; i < 2; ++i) {
        int accepted = 0;
        for (int a = 0; a < 4; ++a) {
            ug_conv_plan *p = ug_conv_plan_create(&bad[i], w, NULL, 0, FALGOS[a], 1);
            accepted += p != NULL;
            ug_conv_plan_destroy(p);
        }
        ug_conv_plan *pa = ug_conv_plan_create(&bad[i], w, NULL, 0, UG_ALGO_AUTO, 1);
        accepted += pa != NULL;
        ug_conv_plan_destroy(pa);
        for (int a = 0; a < 3; ++a) {
            ug_qconv_plan *q = ug_qconv_plan_create(&bad[i], wq, s, NULL, 0, 1.f, 0, QALGOS[a], 1);
            accepted += q != NULL;
            ug_qconv_plan_destroy(q);
        }
        snprintf(msg, sizeof msg, "shape %u: H+2p=%d < R=%d, ug_out_h()=%d, accepted by %d plan kinds", i,
                 bad[i].H + 2 * bad[i].pad_h, bad[i].R, ug_out_h(&bad[i]), accepted);
        report("F2", accepted == 0, "kernel larger than padded input is rejected", msg);
    }
}

/* ---------------- F3 ---------------- */
static void test_f3(void)
{
    char msg[256];
    ug_conv_desc d = {1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 0};
    signed char w[2] = {127, -127};
    float s[2] = {1, 1};
    uint8_t x = 255, y[2] = {7, 7};
    ug_qconv_plan *p = ug_qconv_plan_create(&d, w, s, NULL, 0, 1.f, 0, UG_QALGO_IM2COL_GEMM, 1);
    /* y = +-32385; / 1e-5 = +-3.2e9, beyond int32 */
    int rc = ug_qconv_execute_u8(p, &x, y, 1e-5f, 0);
    snprintf(msg, sizeof msg, "rc %d, +3.2e9 -> %u (want 255), -3.2e9 -> %u (want 0)", rc, y[0], y[1]);
    report("F3", rc == 0 && y[0] == 255 && y[1] == 0, "u8 requantisation saturates", msg);
    rc = ug_qconv_execute_u8(p, &x, y, 1.f, 300);
    snprintf(msg, sizeof msg, "execute_u8 with out_zp=300 returned %d (want -1)", rc);
    report("F3", rc == -1, "out_zp outside [0,255] rejected", msg);
    ug_qconv_plan_destroy(p);
}

/* ---------------- F4 ---------------- */
static void test_f4(void)
{
    char msg[256];
    /* Q = 37: columns 0-15 and 16-31 are full 16-wide tiles, 32-36 an edge tile.
     * NaN at (c=1, ih=10, iw=31) -> outputs ow 30..32 straddle a full and an edge tile.
     * K = 7 -> the second 6-row panel is an edge panel (mr = 1). */
    ug_conv_desc d = {1, 3, 20, 37, 7, 3, 3, 1, 1, 1, 1};
    const int P = ug_out_h(&d), Q = ug_out_w(&d);
    float *x = malloc(in_elems(&d) * 4), *y = malloc(out_elems(&d) * 4);
    float w[7 * 3 * 9], b[7];
    for (size_t i = 0; i < in_elems(&d); ++i) x[i] = 1.f;
    for (int i = 0; i < 7 * 27; ++i) w[i] = 0.1f;
    for (int i = 0; i < 7; ++i) b[i] = -100.f;   /* ReLU input negative everywhere except NaN */
    x[(1 * 20 + 10) * 37 + 31] = NAN;
    for (int a = 0; a < 4; ++a) {
        ug_conv_plan *p = ug_conv_plan_create(&d, w, b, 1, FALGOS[a], 2);
        ug_conv_execute(p, x, y);
        int missing = 0, extra = 0;
        for (int k = 0; k < 7; ++k)
            for (int oh = 0; oh < P; ++oh)
                for (int ow = 0; ow < Q; ++ow) {
                    int recept = abs(oh - 10) <= 1 && abs(ow - 31) <= 1;
                    int isnan_ = isnan(y[(k * P + oh) * Q + ow]);
                    missing += recept && !isnan_;
                    extra += !recept && isnan_;
                }
        /* direct / GEMM: NaN exactly on the receptive field. Winograd mixes a whole tile,
         * so NaN may spread inside the tile (documented), but none may be lost. */
        const int wino = ug_algo_group(FALGOS[a]) == 3;
        snprintf(msg, sizeof msg, "%s: %d receptive-field outputs lost the NaN, %d extra NaN outputs",
                 ug_algo_name(FALGOS[a]), missing, extra);
        report("F4", missing == 0 && (wino || extra == 0), "fp32 ReLU propagates NaN in every tile", msg);
        ug_conv_plan_destroy(p);
    }
    /* int8 f32 epilogue: NaN bias must survive ReLU too */
    {
        ug_conv_desc dq = {1, 4, 5, 21, 7, 3, 3, 1, 1, 1, 1};
        signed char wq[7 * 36];
        float s[7], bq[7];
        uint8_t xq[4 * 5 * 21];
        float yq[7 * 5 * 21];
        memset(wq, 1, sizeof wq);
        memset(xq, 9, sizeof xq);
        for (int i = 0; i < 7; ++i) { s[i] = 1.f; bq[i] = NAN; }
        for (int a = 0; a < 3; ++a) {
            ug_qconv_plan *p = ug_qconv_plan_create(&dq, wq, s, bq, 1, 1.f, 0, QALGOS[a], 1);
            ug_qconv_execute_f32(p, xq, yq);
            int lost = 0;
            for (int i = 0; i < 7 * 5 * 21; ++i) lost += !isnan(yq[i]);
            snprintf(msg, sizeof msg, "%s: %d of %d outputs lost the NaN", ug_qalgo_name(QALGOS[a]), lost, 7 * 5 * 21);
            report("F4", lost == 0, "int8 f32 epilogue ReLU propagates NaN", msg);
            ug_qconv_plan_destroy(p);
        }
    }
    free(x); free(y);
}

/* ---------------- F5 / F11 ---------------- */
typedef struct { int int8; int algo; } fcase;

static int run_case(const fcase *fc, const ug_conv_desc *d, const float *x, const float *w, const float *b,
                    const uint8_t *xq, const int8_t *wq, const float *ws, const double *ref, const int64_t *refq)
{
    const size_t ny = out_elems(d);
    if (!fc->int8) {
        float *y = malloc(ny * 4);
        ug_conv_plan *p = ug_conv_plan_create(d, w, b, 0, (ug_algo)fc->algo, 2);
        if (!p) { free(y); return 0; }                 /* reported failure: fine */
        int rc = ug_conv_execute(p, x, y);
        int bad = 0;
        if (rc == 0)
            for (size_t i = 0; i < ny; ++i) bad += !(fabs(y[i] - ref[i]) <= 1e-3 * (1 + fabs(ref[i])));
        ug_conv_plan_destroy(p);
        free(y);
        return bad ? 3 : 0;                           /* 3: returned success with wrong data */
    }
    int32_t *y = malloc(ny * 4);
    ug_qconv_plan *p = ug_qconv_plan_create(d, (const signed char *)wq, ws, NULL, 0, 1.f, 3, (ug_qalgo)fc->algo, 2);
    if (!p) { free(y); return 0; }
    int rc = ug_qconv_execute_s32(p, xq, y);
    int bad = 0;
    if (rc == 0)
        for (size_t i = 0; i < ny; ++i) bad += y[i] != refq[i];
    ug_qconv_plan_destroy(p);
    free(y);
    return bad ? 3 : 0;
}

/* Each injected case runs in a fresh process (popen -> exec of this binary): GNU OpenMP
 * is not fork-safe once its thread pool exists, so a plain fork() child can deadlock. */
static const fcase F5_CASES[] = {
    {0, UG_ALGO_DIRECT}, {0, UG_ALGO_IM2COL_GEMM}, {0, UG_ALGO_WINOGRAD_F4}, {0, UG_ALGO_WINOGRAD_F2}, {0, UG_ALGO_TUNE},
    {1, UG_QALGO_DIRECT}, {1, UG_QALGO_IM2COL_GEMM}, {1, UG_QALGO_WINOGRAD_F2}, {1, UG_QALGO_TUNE},
};
static const char *g_self;

/* child entry: run case ci with allocation #fail_at failing; print "<rc> <allocations>" */
static int f5_child(int ci, long fail_at)
{
    ug_conv_desc d = {2, 5, 13, 22, 9, 3, 3, 1, 1, 1, 1};
    const size_t nx = in_elems(&d), nw = (size_t)d.K * d.C * 9, ny = out_elems(&d);
    float *x = malloc(nx * 4), *w = malloc(nw * 4), b[9], ws[9];
    uint8_t *xq = malloc(nx);
    int8_t *wq = malloc(nw);
    double *ref = malloc(ny * 8);
    int64_t *refq = malloc(ny * 8);
    for (size_t i = 0; i < nx; ++i) { x[i] = (float)((i * 37) % 11) - 5.f; xq[i] = (uint8_t)(i * 29); }
    for (size_t i = 0; i < nw; ++i) { w[i] = (float)((i * 13) % 7) - 3.f; wq[i] = (int8_t)(i * 17); }
    for (int i = 0; i < 9; ++i) { b[i] = 0.5f * i; ws[i] = 1.f; }
    ref_f64(&d, x, w, b, 0, ref);
    ref_i64(&d, xq, wq, 3, refq);
    ug_test_fail_alloc_at(fail_at);
    int rc = run_case(&F5_CASES[ci], &d, x, w, b, xq, wq, ws, ref, refq);
    const long count = ug_test_alloc_count();
    free(x); free(w); free(xq); free(wq); free(ref); free(refq); /* LeakSanitizer must see no leaks */
    printf("%d %ld\n", rc, count);
    return 0;
}

static void test_f5(void)
{
    char msg[512], cmd[1024], line[128];
    for (unsigned ci = 0; ci < sizeof F5_CASES / sizeof F5_CASES[0]; ++ci) {
        const fcase *fc = &F5_CASES[ci];
        const char *nm = fc->int8 ? ug_qalgo_name((ug_qalgo)fc->algo) : ug_algo_name((ug_algo)fc->algo);
        long nalloc = -1;
        int base = -1;
        snprintf(cmd, sizeof cmd, "'%s' --f5 %u -1", g_self, ci);
        FILE *f = popen(cmd, "r");
        if (f && fgets(line, sizeof line, f)) sscanf(line, "%d %ld", &base, &nalloc);
        if (f) pclose(f);
        int crashes = 0, silent = 0, first_bad = -1;
        for (long i = 0; i < nalloc + 2; ++i) {
            snprintf(cmd, sizeof cmd, "timeout 60 '%s' --f5 %u %ld 2>/dev/null", g_self, ci, i);
            f = popen(cmd, "r");
            int rc = -1;
            long cnt = 0;
            int got = f && fgets(line, sizeof line, f) && sscanf(line, "%d %ld", &rc, &cnt) == 2;
            int st = f ? pclose(f) : -1;
            if (!got || st != 0) { ++crashes; if (first_bad < 0) first_bad = (int)i; }  /* signal, hang or no output */
            else if (rc == 3) { ++silent; if (first_bad < 0) first_bad = (int)i; }
        }
        snprintf(msg, sizeof msg, "%s: %ld allocations failed one at a time; %d crashed/hung, %d returned success with "
                 "wrong output (first at alloc #%d); fault-free run %s", nm, nalloc + 2, crashes, silent, first_bad,
                 base == 0 ? "ok" : "WRONG");
        report("F5", nalloc > 0 && crashes == 0 && silent == 0 && base == 0, "allocation failure is reported, never a crash", msg);
    }
}

/* ---------------- F11 ---------------- */
/* TUNE with one algorithm that can never allocate its scratch: failing fast must not
 * make it the "fastest"; the chosen plan must execute successfully. */
static void test_f11(void)
{
    char msg[256];
    ug_conv_desc d = {1, 64, 40, 40, 64, 3, 3, 1, 1, 1, 1};
    const size_t nx = in_elems(&d), nw = (size_t)d.K * d.C * 9, ny = out_elems(&d);
    float *x = calloc(nx, 4), *w = calloc(nw, 4), *y = malloc(ny * 4);
    int works[4], nwork = 0;
    /* limit chosen so the Winograd executors' per-thread V/M buffers (> 400 KB) fail
     * while every plan's weights and the direct/GEMM scratch fit */
    const size_t limit = 400 * 1024;
    ug_test_fail_alloc_over(limit);
    for (int a = 0; a < 4; ++a) {
        ug_conv_plan *p = ug_conv_plan_create(&d, w, NULL, 0, FALGOS[a], 2);
        works[a] = p && ug_conv_execute(p, x, y) == 0;
        nwork += works[a];
        ug_conv_plan_destroy(p);
    }
    ug_conv_plan *t = ug_conv_plan_create(&d, w, NULL, 0, UG_ALGO_TUNE, 2);
    const int rc = t ? ug_conv_execute(t, x, y) : -2;
    const ug_algo picked = ug_conv_plan_algo(t);
    ug_test_fail_alloc_over(0);
    snprintf(msg, sizeof msg, "working algorithms under a %zu-byte allocation cap: direct %d gemm %d F4 %d F2 %d; "
             "TUNE picked %s, execute rc %d", limit, works[0], works[1], works[2], works[3], ug_algo_name(picked), rc);
    report("F11", nwork > 0 && nwork < 4 && rc == 0, "TUNE never picks a failing algorithm", msg);
    ug_conv_plan_destroy(t);
    free(x); free(w); free(y);
}

/* ---------------- F8 ---------------- */
static void test_f8(void)
{
    char msg[256];
    void *a = ug_malloc(SIZE_MAX - 8), *b = ug_malloc(SIZE_MAX);
    snprintf(msg, sizeof msg, "ug_malloc(SIZE_MAX-8) = %p, ug_malloc(SIZE_MAX) = %p (want NULL, NULL)", a, b);
    report("F8", a == NULL && b == NULL, "ug_malloc rejects sizes that wrap when rounded", msg);
    ug_free(a);
    ug_free(b);
}

/* ---------------- F9 ---------------- */
static void test_f9(void)
{
    char msg[256];
    float w[64] = {0};
    /* each shape is rejected by exactly one check (mutation testing showed that shapes
     * failing several checks at once do not protect any single one of them) */
    const ug_conv_desc big[] = {
        {1, 477218589, 3, 3, 1, 3, 3, 1, 1, 1, 1},     /* dimension > 2^24 */
        {1, 1, 65537, 65537, 1, 1, 1, 1, 1, 0, 0},     /* P*Q and H*W overflow */
        {1, 1, 8, 2147483000, 1, 1, 1, 1, 1, 0, 1000}, /* W + 2*pad overflows int */
        {1, 1 << 24, 16, 16, 1, 16, 16, 1, 1, 0, 0},   /* only C*R*S = 2^32 > INT_MAX */
        {1, 1, 46340, 46340, 1, 1, 1, 1, 1, 1, 1},     /* only P*Q = 46342^2 > INT_MAX (H*W fits) */
        {1, 1, 46341, 46341, 1, 1, 1, 2, 2, 0, 0},     /* only H*W > INT_MAX (P*Q fits) */
    };
    for (unsigned i = 0; i < sizeof big / sizeof big[0]; ++i) {
        ug_conv_plan *p = ug_conv_plan_create(&big[i], w, NULL, 0, UG_ALGO_IM2COL_GEMM, 1);
        snprintf(msg, sizeof msg, "shape %u accepted: %s", i, p ? "yes" : "no");
        report("F9", p == NULL, "int-overflowing shape is rejected", msg);
        ug_conv_plan_destroy(p);
    }
}

/* ---------------- F12 ---------------- */
static void test_f12(void)
{
    char msg[256];
    const ug_conv_desc sh[] = {
        {1, 3, 5, 6, 4, 3, 3, 1, 1, 3, 3},   /* pad == R */
        {2, 5, 4, 7, 7, 3, 3, 1, 1, 4, 2},   /* pad > R */
        {1, 4, 9, 9, 5, 5, 5, 2, 2, 6, 5},   /* pad > S, strided */
        {1, 3, 1, 1, 2, 1, 1, 1, 1, 2, 2},   /* 1x1 kernel with padding */
    };
    for (unsigned i = 0; i < 4; ++i) {
        const ug_conv_desc *d = &sh[i];
        const size_t nx = in_elems(d), nw = (size_t)d->K * d->C * d->R * d->S, ny = out_elems(d);
        float *x = malloc(nx * 4), *w = malloc(nw * 4), *b = malloc(d->K * 4), *y = malloc(ny * 4), *ws = malloc(d->K * 4);
        double *ref = malloc(ny * 8);
        uint8_t *xq = malloc(nx);
        int8_t *wq = malloc(nw);
        int64_t *rq = malloc(ny * 8);
        int32_t *yq = malloc(ny * 4);
        for (size_t j = 0; j < nx; ++j) { x[j] = (float)(j % 7) - 3; xq[j] = (uint8_t)(j * 31); }
        for (size_t j = 0; j < nw; ++j) { w[j] = (float)(j % 5) - 2; wq[j] = (int8_t)(j * 11); }
        for (int j = 0; j < d->K; ++j) { b[j] = 1; ws[j] = 1; }
        ref_i64(d, xq, wq, 5, rq);
        int ok = 1, tried = 0;
        for (int relu = 0; relu < 2; ++relu) {   /* without ReLU too: ReLU can hide wrong negatives */
            ref_f64(d, x, w, b, relu, ref);
            for (int a = 0; a < 4; ++a) {
                if (!ug_algo_eligible(d, FALGOS[a]) && !(d->R == 3 && d->stride_h == 1 && a >= 2)) continue;
                ug_conv_plan *p = ug_conv_plan_create(d, w, b, relu, FALGOS[a], 1);
                ++tried;
                if (!p) { ok = 0; continue; }
                ug_conv_execute(p, x, y);
                for (size_t j = 0; j < ny; ++j) ok &= fabs(y[j] - ref[j]) <= 1e-4 * (1 + fabs(ref[j]));
                ug_conv_plan_destroy(p);
            }
        }
        for (int a = 0; a < 3; ++a) {
            if (a == 2 && !(d->R == 3 && d->S == 3 && d->stride_h == 1 && d->stride_w == 1)) continue;
            ug_qconv_plan *p = ug_qconv_plan_create(d, (const signed char *)wq, ws, NULL, 0, 1.f, 5, QALGOS[a], 1);
            ++tried;
            if (!p) { ok = 0; continue; }
            ug_qconv_execute_s32(p, xq, yq);
            for (size_t j = 0; j < ny; ++j) ok &= yq[j] == rq[j];
            ug_qconv_plan_destroy(p);
        }
        snprintf(msg, sizeof msg, "shape %u (pad %d,%d vs kernel %dx%d): %d plans, %s", i, d->pad_h, d->pad_w, d->R, d->S,
                 tried, ok ? "all accepted and correct" : "rejected or wrong");
        report("F12", ok, "padding >= kernel size works", msg);
        free(x); free(w); free(b); free(y); free(ws); free(ref); free(xq); free(wq); free(rq); free(yq);
    }
}

/* ---------------- B: branch completion (validation paths, env files) ---------------- */
static int cm_child(void)
{
    /* runs in a fresh process with UGCONV_COSTMODEL / UGCONV_QCOSTMODEL set by the parent */
    ug_conv_desc d = {1, 8, 10, 10, 8, 3, 3, 1, 1, 1, 1};
    double c[UG_CM_NFEAT], q[UG_QCM_NFEAT];
    (void)ug_select_algo(&d, 1);
    (void)ug_qselect_algo(&d, 1);
    ug_costmodel_get(c);
    ug_qcostmodel_get(q);
    printf("%g %g\n", c[1], q[1]);
    return 0;
}

static void test_branches(void)
{
    char msg[256], cmd[1024], line[128];
    float w[512] = {0}, s8[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    signed char wq[512] = {0};
    const ug_conv_desc ok = {1, 4, 6, 6, 3, 3, 3, 1, 1, 1, 1};
    int bad = 0;
    /* every rejecting branch of ug_desc_valid */
    ug_conv_desc v;
    int *fields[] = {&v.N, &v.C, &v.H, &v.W, &v.K, &v.R, &v.S, &v.stride_h, &v.stride_w};
    for (int i = 0; i < 9; ++i) {
        v = ok; *fields[i] = 0;
        bad += ug_desc_valid(&v) != 0;
        v = ok; *fields[i] = (1 << 24) + 1;
        bad += ug_desc_valid(&v) != 0;
    }
    v = ok; v.pad_h = -1; bad += ug_desc_valid(&v) != 0;
    v = ok; v.pad_w = -1; bad += ug_desc_valid(&v) != 0;
    v = ok; v.pad_h = (1 << 24) + 1; bad += ug_desc_valid(&v) != 0;
    v = ok; v.pad_w = (1 << 24) + 1; bad += ug_desc_valid(&v) != 0;
    v = ok; v.C = 1 << 24; v.R = v.S = 16; bad += ug_desc_valid(&v) != 0;            /* C*R*S > INT_MAX */
    v = ok; v.H = v.W = 1 << 16; v.R = v.S = 1; v.pad_h = v.pad_w = 0; bad += ug_desc_valid(&v) != 0; /* P*Q */
    bad += ug_desc_valid(NULL) != 0;
    bad += ug_desc_valid(&ok) != 1;
    /* NULL / invalid arguments on every entry point */
    ug_conv_plan *p = ug_conv_plan_create(&ok, w, NULL, 0, UG_ALGO_AUTO, 0);   /* nthreads <= 0 -> default */
    bad += p == NULL;
    float x[256] = {0}, y[256];
    bad += ug_conv_execute(p, NULL, y) != -1 || ug_conv_execute(p, x, NULL) != -1 || ug_conv_execute(NULL, x, y) != -1;
    bad += ug_conv_execute(p, x, y) != 0;
    bad += ug_conv_plan_algo(NULL) != UG_ALGO_AUTO;
    ug_conv_plan_destroy(p);
    ug_conv_plan_destroy(NULL);
    bad += ug_conv_plan_create(&ok, NULL, NULL, 0, UG_ALGO_AUTO, 1) != NULL;
    bad += ug_conv_plan_create(&ok, w, NULL, 0, (ug_algo)42, 1) != NULL;
    bad += ug_algo_cost(&ok, (ug_algo)42, 1) >= 0 || ug_algo_eligible(&ok, (ug_algo)42);
    bad += ug_algo_cost(&ok, UG_ALGO_DIRECT, 0) < 0;                            /* default thread count */
    ug_conv_desc huge = {1, 2048, 14, 14, 2048, 3, 3, 1, 1, 1, 1};              /* F4 weights > L3 */
    bad += ug_algo_cost(&huge, UG_ALGO_WINOGRAD_F4, 24) <= 0;
    ug_qconv_plan *qp = ug_qconv_plan_create(&ok, wq, s8, NULL, 0, 1.f, 0, UG_QALGO_AUTO, 0);
    unsigned char xq[256] = {0}, yq8[256];
    int yq[256];
    bad += qp == NULL;
    bad += ug_qconv_execute_f32(qp, NULL, y) != -1 || ug_qconv_execute_f32(NULL, xq, y) != -1;
    bad += ug_qconv_execute_s32(qp, xq, NULL) != -1 || ug_qconv_execute_u8(qp, xq, NULL, 1.f, 0) != -1;
    bad += ug_qconv_execute_u8(qp, xq, yq8, 1.f, -1) != -1;
    bad += ug_qconv_execute_s32(qp, xq, yq) != 0;
    bad += ug_qconv_plan_algo(NULL) != UG_QALGO_AUTO;
    ug_qconv_plan_destroy(qp);
    ug_qconv_plan_destroy(NULL);
    bad += ug_qconv_plan_create(&ok, wq, s8, NULL, 0, 1.f, -1, UG_QALGO_AUTO, 1) != NULL;
    bad += ug_qconv_plan_create(&ok, wq, s8, NULL, 0, 1.f, 0, (ug_qalgo)42, 1) != NULL;
    bad += ug_qalgo_cost(&ok, (ug_qalgo)42, 1) >= 0 || ug_qalgo_cost(&ok, UG_QALGO_DIRECT, 0) < 0;
    v = ok; v.H = 1; v.R = 3; v.pad_h = 0;                                       /* invalid shape */
    bad += ug_qconv_plan_create(&v, wq, s8, NULL, 0, 1.f, 0, UG_QALGO_AUTO, 1) != NULL;
    void *z = ug_malloc(0);
    bad += z == NULL;                                                            /* 0 bytes: valid pointer */
    ug_free(z);
    bad += ug_malloc((size_t)PTRDIFF_MAX) != NULL;                               /* > PTRDIFF_MAX guard */
    bad += ug_malloc((size_t)1 << 62) != NULL;                                   /* legal size, ENOMEM */
    snprintf(msg, sizeof msg, "%d validation branches misbehaved", bad);
    report("B1", bad == 0, "every argument-validation branch", msg);

    /* cost-model files: only read once per process, so each case runs in a fresh process */
    static const struct { const char *content; int accept; } files[] = {
        {"0.1\n7\n0.1\n1\n1\n1\n0.1\n1\n1\n1\n1\n1\n", 1},   /* 12 valid values */
        {"0.1\nnan\n0.1\n1\n1\n1\n0.1\n1\n1\n1\n1\n1\n", 0}, /* NaN */
        {"0.1\n-3\n0.1\n1\n1\n1\n0.1\n1\n1\n1\n1\n1\n", 0},  /* negative */
        {"0.1\n7\n", 0},                                          /* too short */
        {NULL, 0},                                                 /* missing file */
    };
    for (unsigned i = 0; i < 5; ++i) {
        char path[] = "/tmp/ugcm_XXXXXX";
        int fd = mkstemp(path);
        if (files[i].content) { FILE *f = fdopen(fd, "w"); fputs(files[i].content, f); fclose(f); }
        else { close(fd); unlink(path); }
        snprintf(cmd, sizeof cmd, "UGCONV_COSTMODEL=%s UGCONV_QCOSTMODEL=%s '%s' --cm", path, path, g_self);
        FILE *pp = popen(cmd, "r");
        double c1 = -1, q1 = -1;
        if (pp && fgets(line, sizeof line, pp)) sscanf(line, "%lf %lf", &c1, &q1);
        if (pp) pclose(pp);
        unlink(path);
        const int loaded_f = c1 == 7, loaded_q = q1 == 7;
        /* the int8 model has 10 constants: a 12-line file loads its first 10 */
        const int pass = files[i].accept ? (loaded_f && loaded_q) : (!loaded_f && !loaded_q && c1 > 0 && q1 > 0);
        snprintf(msg, sizeof msg, "file case %u: fp32 constant[1] = %g, int8 constant[1] = %g (%s expected)", i, c1, q1,
                 files[i].accept ? "loaded" : "defaults");
        report("B2", pass, "cost-model file accepted only if every value is finite and >= 0", msg);
    }
}

int main(int argc, char **argv)
{
    g_self = argv[0];
    if (argc == 4 && !strcmp(argv[1], "--f5")) return f5_child(atoi(argv[2]), atol(argv[3]));
    if (argc == 2 && !strcmp(argv[1], "--cm")) return cm_child();
    setvbuf(stdout, NULL, _IOLBF, 0);
    test_f1();
    test_f2();
    test_f3();
    test_f4();
    test_f8();
    test_f9();
    test_f12();
    test_f5();
    test_f11();
    test_branches();
    printf("%d/%d regression checks passed\n", total - fails, total);
    return fails ? 1 : 0;
}
