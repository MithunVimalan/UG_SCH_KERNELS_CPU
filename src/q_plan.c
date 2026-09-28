/* INT8 plan creation, epilogues, quantisation helpers, selection. */
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "q_internal.h"

const char *ug_qvnni_backend(void) { return ug_vnni_backend(); }

const char *ug_qalgo_name(ug_qalgo a)
{
    switch (a) {
    case UG_QALGO_AUTO: return "q-auto";
    case UG_QALGO_DIRECT: return "Q1-direct";
    case UG_QALGO_IM2COL_GEMM: return "Q2-im2col+gemm";
    case UG_QALGO_WINOGRAD_F2: return "Q3-winograd-F2";
    case UG_QALGO_TUNE: return "q-tune";
    }
    return "?";
}

int ug_qalgo_eligible(const ug_conv_desc *d, ug_qalgo a)
{
    /* same shape rules as fp32, plus the exact-int32 bound on the reduction (F1) */
    if (!ug_algo_eligible(d, UG_ALGO_IM2COL_GEMM)) return 0;
    if ((long long)d->C * d->R * d->S > UG_Q_MAX_CRS) return 0;
    switch (a) {
    case UG_QALGO_DIRECT:
    case UG_QALGO_IM2COL_GEMM: return 1;
    case UG_QALGO_WINOGRAD_F2:
        return d->R == 3 && d->S == 3 && d->stride_h == 1 && d->stride_w == 1 && d->C <= UG_QWINO_MAX_C;
    default: return 0;
    }
}

void ug_quantize_weights_s8(const float *w, int K, int CRS, signed char *q, float *scale)
{
    for (int k = 0; k < K; ++k) {
        float m = 0;
        for (int i = 0; i < CRS; ++i) m = fmaxf(m, fabsf(w[(size_t)k * CRS + i]));
        float s = m > 0 ? m / 127.f : 1.f;
        scale[k] = s;
        for (int i = 0; i < CRS; ++i) {
            float v = nearbyintf(w[(size_t)k * CRS + i] / s);
            q[(size_t)k * CRS + i] = (signed char)fminf(127.f, fmaxf(-127.f, v));
        }
    }
}

void ug_quantize_u8(const float *x, long n, float scale, int zp, unsigned char *q)
{
    for (long i = 0; i < n; ++i) {
        float v = nearbyintf(x[i] / scale) + zp;
        q[i] = (unsigned char)fminf(255.f, fmaxf(0.f, v));
    }
}

/* y = fma((float)(acc - corr), oscale[k], bias[k]); relu; then per kind. */
void ug_q_epilogue16(const ug_qconv_plan *p, const ug_qout *o, int k, long off,
                     __m256i v0, __m256i v1, int n, int centered)
{
    if (!centered) {
        __m256i c = _mm256_set1_epi32(p->zcorr[k]);
        v0 = _mm256_sub_epi32(v0, c);
        v1 = _mm256_sub_epi32(v1, c);
    }
    if (o->kind == UG_QOUT_S32) {
        int32_t *dst = (int32_t *)o->base + off;
        if (n == 16) {
            _mm256_storeu_si256((__m256i *)dst, v0);
            _mm256_storeu_si256((__m256i *)(dst + 8), v1);
        } else {
            int32_t t[16] __attribute__((aligned(32)));
            _mm256_store_si256((__m256i *)t, v0);
            _mm256_store_si256((__m256i *)(t + 8), v1);
            for (int j = 0; j < n; ++j) dst[j] = t[j];
        }
        return;
    }
    const __m256 sc = _mm256_set1_ps(p->oscale[k]), bs = _mm256_set1_ps(p->bias[k]);
    __m256 f0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(v0), sc, bs);
    __m256 f1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(v1), sc, bs);
    if (p->relu) {
        f0 = _mm256_max_ps(_mm256_setzero_ps(), f0); /* keeps NaN, like the fp32 path */
        f1 = _mm256_max_ps(_mm256_setzero_ps(), f1);
    }
    if (o->kind == UG_QOUT_F32) {
        float *dst = (float *)o->base + off;
        if (n == 16) {
            _mm256_storeu_ps(dst, f0);
            _mm256_storeu_ps(dst + 8, f1);
        } else {
            float t[16] __attribute__((aligned(32)));
            _mm256_store_ps(t, f0);
            _mm256_store_ps(t + 8, f1);
            for (int j = 0; j < n; ++j) dst[j] = t[j];
        }
        return;
    }
    /* u8: round-to-nearest-even (MXCSR default) == nearbyintf in the reference.
     * Clamp in float first (F3): cvtps_epi32 returns INT_MIN for |x| >= 2^31, which
     * would turn a huge positive value into 0. With out_zp in [0,255], any value
     * outside [-512, 512] saturates the same way after the integer clamp; NaN -> 0
     * (max_ps returns the second operand), matching the reference. */
    const __m256 inv = _mm256_set1_ps(o->inv_out_scale);
    const __m256 lo = _mm256_set1_ps(-512.f), hi = _mm256_set1_ps(512.f);
    const __m256i zp = _mm256_set1_epi32(o->out_zp);
    __m256 s0 = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(f0, inv), lo), hi);
    __m256 s1 = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(f1, inv), lo), hi);
    __m256i q0 = _mm256_add_epi32(_mm256_cvtps_epi32(s0), zp);
    __m256i q1 = _mm256_add_epi32(_mm256_cvtps_epi32(s1), zp);
    q0 = _mm256_min_epi32(_mm256_max_epi32(q0, _mm256_setzero_si256()), _mm256_set1_epi32(255));
    q1 = _mm256_min_epi32(_mm256_max_epi32(q1, _mm256_setzero_si256()), _mm256_set1_epi32(255));
    int32_t t[16] __attribute__((aligned(32)));
    _mm256_store_si256((__m256i *)t, q0);
    _mm256_store_si256((__m256i *)(t + 8), q1);
    uint8_t *dst = (uint8_t *)o->base + off;
    for (int j = 0; j < n; ++j) dst[j] = (uint8_t)t[j];
}

static int g_qisa_ok;
static pthread_once_t g_qonce = PTHREAD_ONCE_INIT;

static void qcostmodel_env(void);
static void qinit_once(void) /* F6: one-time, thread-safe */
{
    ug_cpu_info ci;
    ug_cpu_detect(&ci); /* avx2/fma/avx512f already include the OS-support (XGETBV) check */
    int ok = ci.avx2 && ci.fma;
#if defined(UG_VNNI_VEX)
    ok = ok && ci.avx_vnni;
#elif defined(UG_VNNI_EVEX)
    unsigned a, b, c, d;
    __asm__("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(7), "c"(0));
    ok = ok && ci.avx512f && ((c >> 11) & 1) && ((b >> 31) & 1); /* AVX512_VNNI, AVX512VL */
#endif
    g_qisa_ok = ok;
    qcostmodel_env();
}

static void q_destroy_fields(ug_qconv_plan *p)
{
    ug_free(p->oscale); ug_free(p->bias); ug_free(p->zcorr);
    ug_free(p->wd); ug_free(p->wg); ug_free(p->ww);
}

static ug_qconv_plan *q_create_fixed(const ug_conv_desc *d, const int8_t *w, const float *w_scale,
                                     const float *bias, int relu, float in_scale, int in_zp,
                                     ug_qalgo algo, int nt)
{
    ug_qconv_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->d = *d;
    p->algo = algo;
    p->relu = relu;
    p->nthreads = nt;
    p->P = ug_out_h(d);
    p->Q = ug_out_w(d);
    p->Kp = ug_round_up(d->K, UG_MR);
    p->in_scale = in_scale;
    p->in_zp = in_zp;
    p->oscale = ug_malloc(p->Kp * sizeof(float));
    p->bias = ug_malloc(p->Kp * sizeof(float));
    p->zcorr = ug_malloc(p->Kp * sizeof(int32_t));
    if (!p->oscale || !p->bias || !p->zcorr) { q_destroy_fields(p); free(p); return NULL; }
    const long CRS = (long)d->C * d->R * d->S;
    for (int k = 0; k < p->Kp; ++k) {
        int real = k < d->K;
        p->oscale[k] = real ? in_scale * w_scale[k] : 0.f;
        p->bias[k] = (real && bias) ? bias[k] : 0.f;
        long long s = 0; /* |in_zp * s| <= 255*128*CRS < 2^31 by the F1 bound: no overflow */
        if (real)
            for (long i = 0; i < CRS; ++i) s += w[k * CRS + i];
        p->zcorr[k] = (int32_t)(in_zp * s);
    }
    int rc = -1;
    switch (algo) {
    case UG_QALGO_DIRECT: rc = ug_qdirect_prepare(p, w); break;
    case UG_QALGO_IM2COL_GEMM: rc = ug_qgemm_prepare(p, w); break;
    case UG_QALGO_WINOGRAD_F2: rc = ug_qwino_prepare(p, w); break;
    default: break;
    }
    if (rc) { q_destroy_fields(p); free(p); return NULL; }
    return p;
}

static int q_run(const ug_qconv_plan *p, const uint8_t *in, const ug_qout *o)
{
    long img = (long)p->d.K * p->P * p->Q;
    switch (p->algo) {
    case UG_QALGO_DIRECT: return ug_qdirect_execute(p, in, o, img);
    case UG_QALGO_IM2COL_GEMM: return ug_qgemm_execute(p, in, o, img);
    case UG_QALGO_WINOGRAD_F2: return ug_qwino_execute(p, in, o, img);
    default: return -1;
    }
}

ug_qconv_plan *ug_qconv_plan_create(const ug_conv_desc *d, const signed char *w, const float *w_scale,
                                    const float *bias, int relu, float in_scale, int in_zp,
                                    ug_qalgo algo, int nthreads)
{
    pthread_once(&g_qonce, qinit_once);
    if (!g_qisa_ok || !w || !w_scale || in_zp < 0 || in_zp > 255) return NULL;
    if (!ug_qalgo_eligible(d, UG_QALGO_IM2COL_GEMM)) return NULL;
    if (nthreads <= 0) nthreads = omp_get_max_threads();
    if (algo == UG_QALGO_AUTO) algo = ug_qselect_algo(d, nthreads);
    if (algo != UG_QALGO_TUNE) {
        if (!ug_qalgo_eligible(d, algo)) return NULL;
        return q_create_fixed(d, (const int8_t *)w, w_scale, bias, relu, in_scale, in_zp, algo, nthreads);
    }
    size_t nx = (size_t)d->N * d->C * d->H * d->W, ny = (size_t)d->N * d->K * ug_out_h(d) * ug_out_w(d);
    uint8_t *x = ug_malloc(nx);
    float *y = ug_malloc(ny * sizeof(float));
    if (!x || !y) { ug_free(x); ug_free(y); return NULL; }
    for (size_t i = 0; i < nx; ++i) x[i] = (uint8_t)((i * 2654435761u) >> 24);
    static const ug_qalgo cand[] = {UG_QALGO_DIRECT, UG_QALGO_IM2COL_GEMM, UG_QALGO_WINOGRAD_F2};
    ug_qconv_plan *best = NULL;
    double bt = 1e300;
    for (unsigned i = 0; i < sizeof cand / sizeof cand[0]; ++i) {
        if (!ug_qalgo_eligible(d, cand[i])) continue;
        ug_qconv_plan *p = q_create_fixed(d, (const int8_t *)w, w_scale, bias, relu, in_scale, in_zp, cand[i], nthreads);
        if (!p) continue;
        double t = 1e300;
        int rc = ug_qconv_execute_f32(p, x, y); /* warm-up */
        for (int rep = 0; rep < 3 && rc == 0; ++rep) {
            double t0 = omp_get_wtime();
            rc = ug_qconv_execute_f32(p, x, y);
            t = fmin(t, omp_get_wtime() - t0);
        }
        if (rc) { ug_qconv_plan_destroy(p); continue; } /* F11 */
        if (t < bt) { bt = t; ug_qconv_plan_destroy(best); best = p; }
        else ug_qconv_plan_destroy(p);
    }
    ug_free(x);
    ug_free(y);
    return best;
}

ug_qalgo ug_qconv_plan_algo(const ug_qconv_plan *p) { return p ? p->algo : UG_QALGO_AUTO; }

int ug_qconv_execute_s32(const ug_qconv_plan *p, const unsigned char *in, int *out)
{
    if (!p || !in || !out) return -1;
    ug_qout o = {UG_QOUT_S32, out, 0.f, 0};
    return q_run(p, in, &o);
}

int ug_qconv_execute_f32(const ug_qconv_plan *p, const unsigned char *in, float *out)
{
    if (!p || !in || !out) return -1;
    ug_qout o = {UG_QOUT_F32, out, 0.f, 0};
    return q_run(p, in, &o);
}

int ug_qconv_execute_u8(const ug_qconv_plan *p, const unsigned char *in, unsigned char *out,
                        float out_scale, int out_zp)
{
    if (!p || !in || !out || !(out_scale > 0) || out_zp < 0 || out_zp > 255) return -1; /* F3 */
    ug_qout o = {UG_QOUT_U8, out, 1.f / out_scale, out_zp};
    return q_run(p, in, &o);
}

void ug_qconv_plan_destroy(ug_qconv_plan *p)
{
    if (!p) return;
    q_destroy_fields(p);
    free(p);
}

/* ---- INT8 cost model: sum(feature * cycles) / capacity (same form as fp32).
 * Peak: 2 VNNI/cycle x 8 lanes x 4 (u8s8) = 64 MAC/cycle; int16 pairs: 32 MAC/cycle.
 * Defaults fitted single-threaded on the dev VM with tools/calibrate.c -int8. */
enum { QF_MAC_DIRECT, QF_MAC_GEMM, QF_MAC_WINO, QF_TILE, QF_EPI, QF_PACK_G2, QF_PACK_G1, QF_WIN_IN, QF_WIN_OUT,
       QF_PACK_G2_FAST };
const char *const ug_qcm_feature_names[UG_QCM_NFEAT] = {
    "direct u8 MAC", "gemm u8 MAC", "winograd s16 MAC", "6x16 tile", "output element",
    "im2col byte (scalar)", "direct repack byte", "F2 input tile*ch", "F2 output tile*ch", "im2col byte (vector)"};
static double QCM[UG_QCM_NFEAT] = {
    0.01609, 0.01608, 0.0772, 29.4, 0.893, 6.61, 2.29, 11.7, 0.0, 0.711};

void ug_qcostmodel_get(double c[UG_QCM_NFEAT]) { memcpy(c, QCM, sizeof QCM); }
void ug_qcostmodel_set(const double c[UG_QCM_NFEAT]) { memcpy(QCM, c, sizeof QCM); }

int ug_qcostmodel_features(const ug_conv_desc *d, ug_qalgo a, double f[UG_QCM_NFEAT])
{
    memset(f, 0, UG_QCM_NFEAT * sizeof(double));
    if (!ug_qalgo_eligible(d, a)) return -1;
    const double N = d->N, K = d->K, R = d->R, S = d->S;
    const int P = ug_out_h(d), Q = ug_out_w(d);
    const double PQ = (double)P * Q, Kp = ug_round_up(d->K, UG_MR);
    f[QF_EPI] = N * K * PQ;
    switch (a) {
    case UG_QALGO_DIRECT: {
        double C4 = ug_round_up(d->C, 4), Q16 = ug_round_up(Q, 16);
        f[QF_MAC_DIRECT] = N * Kp * C4 * R * S * P * Q16;
        f[QF_TILE] = N * (Kp / UG_MR) * P * (Q16 / 16);
        f[QF_PACK_G1] = N * C4 * (d->H + 2.0 * d->pad_h) * (Q16 + S) * d->stride_w;
        break;
    }
    case UG_QALGO_IM2COL_GEMM: {
        double CRS4 = ug_round_up(d->C * d->R * d->S, 4), PQ16 = ug_round_up(P * Q, 16);
        f[QF_MAC_GEMM] = N * Kp * CRS4 * PQ16;
        f[QF_TILE] = N * (Kp / UG_MR) * (PQ16 / 16);
        /* fraction of 16-pixel rows packed with one vector load (see im2col_row16) */
        const int pointwise = d->R == 1 && d->S == 1 && d->stride_h == 1 && d->stride_w == 1 && d->pad_h == 0 && d->pad_w == 0;
        double fast = pointwise ? 1.0
                    : d->stride_w == 1 ? fmax(0.0, (Q - 15.0 - d->pad_w) / Q)
                    : d->stride_w == 2 ? fmax(0.0, (Q - 16.0 - d->pad_w) / Q) : 0.0;
        f[QF_PACK_G2_FAST] = N * CRS4 * PQ16 * fast;
        f[QF_PACK_G2] = N * CRS4 * PQ16 * (1 - fast);
        break;
    }
    case UG_QALGO_WINOGRAD_F2: {
        double C2 = ug_round_up(d->C, 2), T = (double)ug_ceil_div(P, 2) * ug_ceil_div(Q, 2);
        double T16 = ug_round_up((int)T, 16);
        f[QF_MAC_WINO] = N * 16 * Kp * C2 * T16;
        f[QF_TILE] = N * 16 * (Kp / UG_MR) * (T16 / 16);
        f[QF_WIN_IN] = N * C2 * T16;
        f[QF_WIN_OUT] = N * K * T;
        break;
    }
    default: return -1;
    }
    return 0;
}

static void qcostmodel_env(void) /* called once, from qinit_once */
{
    const char *path = getenv("UGCONV_QCOSTMODEL");
    FILE *fp = path ? fopen(path, "r") : NULL;
    if (!fp) return;
    double c[UG_QCM_NFEAT];
    int n = 0;
    while (n < UG_QCM_NFEAT && fscanf(fp, "%lf%*[^\n]", &c[n]) == 1 && isfinite(c[n]) && c[n] >= 0) ++n;
    fclose(fp);
    if (n == UG_QCM_NFEAT) ug_qcostmodel_set(c);
}

double ug_qalgo_cost(const ug_conv_desc *d, ug_qalgo a, int nthreads)
{
    double f[UG_QCM_NFEAT];
    pthread_once(&g_qonce, qinit_once);
    if (ug_qcostmodel_features(d, a, f)) return -1.0;
    const ug_machine *mc = ug_machine_get();
    if (nthreads <= 0) nthreads = mc->p_cores + mc->e_cores;
    double cyc = 0;
    for (int i = 0; i < UG_QCM_NFEAT; ++i) cyc += f[i] * QCM[i];
    int np = ug_min(nthreads, mc->p_cores), ne = ug_min(ug_max(nthreads - np, 0), mc->e_cores);
    return cyc / ((np * mc->p_ghz + ne * mc->e_ghz) * 1e9);
}

ug_qalgo ug_qselect_algo(const ug_conv_desc *d, int nthreads)
{
    static const ug_qalgo cand[] = {UG_QALGO_IM2COL_GEMM, UG_QALGO_DIRECT, UG_QALGO_WINOGRAD_F2};
    ug_qalgo best = UG_QALGO_IM2COL_GEMM;
    double bt = 1e300;
    for (unsigned i = 0; i < sizeof cand / sizeof cand[0]; ++i) {
        double t = ug_qalgo_cost(d, cand[i], nthreads);
        if (t >= 0 && t < bt) { bt = t; best = cand[i]; }
    }
    return best;
}
