/* INT8 correctness: BIT-EXACT against a scalar integer reference.
 *
 * All three algorithms compute the same integer  acc = sum (x - zp) * w  with no
 * rounding, so s32 output must match exactly; f32 output must equal
 * fmaf((float)acc, in_scale*w_scale[k], bias[k]) (+ReLU) exactly; u8 output must
 * equal clamp(nearbyintf(y * (1/out_scale)) + out_zp, 0, 255) exactly.
 *
 * Shapes: 24 hand-picked tail/edge cases + 120 random shapes (C 1..300, K 1..70,
 * H,W 1..40, R,S 1..7 independently, stride 1..3 per axis, pad 0..R-1, N 1..2),
 * zero points {0, 1..254, 255}, data random / extreme (x = 255, w = -128),
 * threads 1 and 3, plus a C = 1800 extreme case for the Winograd range bound.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ugconv.h"
#define UG_QWINO_MAX_C_TEST 1801

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd(void)
{
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (uint32_t)(rng >> 16);
}
static int rint_(int lo, int hi) { return lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); }

static int fails = 0, runs = 0, do_tune = 0;

static void check_shape(const ug_conv_desc *d, int zp, int extreme, int verbose)
{
    const int P = ug_out_h(d), Q = ug_out_w(d);
    const size_t nx = (size_t)d->N * d->C * d->H * d->W, nw = (size_t)d->K * d->C * d->R * d->S;
    const size_t ny = (size_t)d->N * d->K * P * Q;
    uint8_t *x = malloc(nx);
    int8_t *w = malloc(nw);
    float *ws = malloc(d->K * 4), *b = malloc(d->K * 4);
    int32_t *ref = malloc(ny * 4), *ys = malloc(ny * 4);
    float *yf = malloc(ny * 4);
    uint8_t *yq = malloc(ny);
    for (size_t i = 0; i < nx; ++i) x[i] = extreme ? 255 : (uint8_t)rnd();
    for (size_t i = 0; i < nw; ++i) w[i] = extreme ? -128 : (int8_t)rnd();
    for (int k = 0; k < d->K; ++k) { ws[k] = 0.001f + (rnd() % 1000) * 1e-5f; b[k] = ((int)(rnd() % 2001) - 1000) * 1e-3f; }
    const float in_scale = 0.0173f, out_scale = 0.37f;
    const int out_zp = 17;

    /* scalar integer reference, int64 to prove no overflow in the reference itself */
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
                    if (s != (int32_t)s) { printf("reference overflow\n"); exit(2); }
                    ref[(((size_t)n * d->K + k) * P + oh) * Q + ow] = (int32_t)s;
                }

    static const ug_qalgo algos[] = {UG_QALGO_DIRECT, UG_QALGO_IM2COL_GEMM, UG_QALGO_WINOGRAD_F2, UG_QALGO_AUTO, UG_QALGO_TUNE};
    for (unsigned ai = 0; ai < 5; ++ai) {
        if (algos[ai] == UG_QALGO_TUNE && !do_tune) continue;
        const int fixed_algo = algos[ai] != UG_QALGO_AUTO && algos[ai] != UG_QALGO_TUNE;
        if (fixed_algo && !ug_qalgo_eligible(d, algos[ai])) continue;
        for (int nt = 1; nt <= 3; nt += 2) {
            for (int relu = 0; relu < 2; ++relu) {
                ug_qconv_plan *p = ug_qconv_plan_create(d, (const signed char *)w, ws, b, relu, in_scale, zp, algos[ai], nt);
                ++runs;
                if (!p) { printf("FAIL plan_create %s\n", ug_qalgo_name(algos[ai])); ++fails; continue; }
                memset(ys, 0xAB, ny * 4);
                memset(yf, 0xAB, ny * 4);
                memset(yq, 0xAB, ny);
                ug_qconv_execute_s32(p, x, ys);
                ug_qconv_execute_f32(p, x, yf);
                ug_qconv_execute_u8(p, x, yq, out_scale, out_zp);
                long bad_s = 0, bad_f = 0, bad_q = 0, first = -1;
                const float inv = 1.f / out_scale;
                for (size_t i = 0; i < ny; ++i) {
                    const int k = (int)((i / ((size_t)P * Q)) % d->K);
                    float f = fmaf((float)ref[i], in_scale * ws[k], b[k]);
                    if (relu && f < 0) f = 0;
                    float qf = nearbyintf(f * inv) + out_zp;
                    uint8_t q = (uint8_t)(qf < 0 ? 0 : qf > 255 ? 255 : qf);
                    if (ys[i] != ref[i]) { ++bad_s; if (first < 0) first = (long)i; }
                    if (memcmp(&yf[i], &f, 4)) { ++bad_f; if (first < 0) first = (long)i; }
                    if (yq[i] != q) { ++bad_q; if (first < 0) first = (long)i; }
                }
                const int ok = !bad_s && !bad_f && !bad_q;
                if (!ok) {
                    ++fails;
                    printf("FAIL N%d C%d H%d W%d K%d R%d S%d s%d,%d p%d,%d zp%d ext%d %s nt%d relu%d: "
                           "s32 %ld f32 %ld u8 %ld mismatches (first at %ld: got %d want %d)\n",
                           d->N, d->C, d->H, d->W, d->K, d->R, d->S, d->stride_h, d->stride_w, d->pad_h, d->pad_w,
                           zp, extreme, ug_qalgo_name(ug_qconv_plan_algo(p)), nt, relu, bad_s, bad_f, bad_q, first,
                           ys[first], ref[first]);
                } else if (verbose) {
                    printf("ok   N%d C%d H%d W%d K%d R%d S%d s%d,%d p%d,%d zp%d ext%d %-15s nt%d relu%d\n",
                           d->N, d->C, d->H, d->W, d->K, d->R, d->S, d->stride_h, d->stride_w, d->pad_h, d->pad_w,
                           zp, extreme, ug_qalgo_name(ug_qconv_plan_algo(p)), nt, relu);
                }
                ug_qconv_plan_destroy(p);
            }
        }
    }
    free(x); free(w); free(ws); free(b); free(ref); free(ys); free(yf); free(yq);
}

int main(int argc, char **argv)
{
    int verbose = argc > 1 && !strcmp(argv[1], "-v");
    printf("VNNI backend: %s\n", ug_qvnni_backend());
    const ug_conv_desc fixed[] = {
        /* N  C   H   W   K  R S sh sw ph pw */
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},       /* smallest possible */
        {1, 3, 17, 19, 7, 3, 3, 1, 1, 1, 1},
        {2, 5, 23, 37, 13, 3, 3, 1, 1, 1, 1},
        {1, 16, 34, 70, 16, 3, 3, 1, 1, 1, 1},     /* 16-tile row groups: fast Winograd path */
        {1, 8, 66, 66, 6, 3, 3, 1, 1, 1, 1},
        {1, 32, 14, 14, 20, 3, 3, 1, 1, 1, 1},
        {1, 11, 7, 7, 9, 3, 3, 1, 1, 1, 1},
        {1, 130, 12, 13, 25, 3, 3, 1, 1, 1, 1},
        {1, 24, 20, 21, 200, 3, 3, 1, 1, 1, 1},    /* Kp > 96: Winograd K blocks */
        {2, 3, 31, 29, 10, 7, 7, 2, 2, 3, 3},      /* stem-like */
        {1, 7, 19, 22, 11, 5, 5, 1, 1, 2, 2},
        {1, 9, 25, 26, 14, 3, 3, 2, 2, 1, 1},
        {1, 6, 21, 20, 8, 3, 3, 3, 3, 1, 1},
        {1, 6, 21, 20, 8, 3, 1, 2, 1, 1, 0},
        {1, 6, 21, 40, 8, 1, 5, 1, 2, 0, 2},       /* different stride/pad per axis */
        {1, 40, 15, 17, 33, 1, 1, 1, 1, 0, 0},
        {2, 17, 16, 16, 30, 1, 1, 2, 2, 0, 0},
        {1, 300, 5, 6, 7, 1, 1, 1, 1, 0, 0},
        {1, 64, 18, 18, 64, 3, 3, 1, 1, 1, 1},
        {1, 4, 64, 64, 12, 3, 3, 2, 2, 1, 1},      /* stride-2 vector pack path (W >= 32) */
        {1, 2, 3, 3, 5, 3, 3, 1, 1, 0, 0},         /* 1x1 output */
        {1, 5, 9, 2, 3, 3, 2, 1, 1, 1, 0},
        {1, 64, 56, 56, 64, 1, 1, 1, 1, 0, 0},     /* ResNet 1x1 */
        {1, 33, 9, 9, 7, 3, 3, 1, 1, 2, 2},        /* pad larger than needed */
    };
    const int zps[] = {0, 128, 255, 77};
    for (unsigned i = 0; i < sizeof fixed / sizeof fixed[0]; ++i)
        for (int z = 0; z < 4; ++z) check_shape(&fixed[i], zps[z], 0, verbose);
    for (unsigned i = 0; i < sizeof fixed / sizeof fixed[0]; ++i) check_shape(&fixed[i], 0, 1, verbose);
    do_tune = 1; /* TUNE on the fixed shapes only (it times every algorithm) */
    for (unsigned i = 0; i < sizeof fixed / sizeof fixed[0]; ++i) check_shape(&fixed[i], 91, 0, verbose);
    do_tune = 0;
    printf("fixed shapes done: %d runs, %d fails\n", runs, fails);

    int nrand = 0;
    while (nrand < 120) {
        ug_conv_desc d;
        d.N = rint_(1, 2); d.C = rnd() % 4 == 0 ? rint_(65, 300) : rint_(1, 64); d.K = rint_(1, 70);
        d.R = rint_(1, 7); d.S = rnd() % 2 ? d.R : rint_(1, 7);
        d.H = rint_(1, 40); d.W = rint_(1, 40);
        d.stride_h = rint_(1, 3); d.stride_w = rnd() % 2 ? d.stride_h : rint_(1, 3);
        d.pad_h = rint_(0, d.R - 1); d.pad_w = rint_(0, d.S - 1);
        if (rnd() % 3 == 0) { d.R = d.S = 3; d.stride_h = d.stride_w = 1; d.pad_h = d.pad_w = 1; } /* Winograd-eligible */
        if (d.H + 2 * d.pad_h < d.R || d.W + 2 * d.pad_w < d.S) continue;
        if ((double)d.N * d.K * d.C * d.R * d.S * ug_out_h(&d) * ug_out_w(&d) > 4e7) continue;
        check_shape(&d, rint_(0, 255), rnd() % 8 == 0, verbose);
        ++nrand;
    }
    printf("random shapes done: %d runs, %d fails\n", runs, fails);

    /* Winograd range bound at its limit: C = 1800, x = 255, w = -128 everywhere */
    ug_conv_desc big = {1, 1800, 6, 6, 7, 3, 3, 1, 1, 1, 1};
    check_shape(&big, 0, 1, verbose);
    /* API checks: quantisation helpers against their definitions, names, invalid input */
    {
        int bad = 0;
        float wf[2 * 5] = {0.25f, -1.0f, 0.25f, 0.0f, 0.999f, 0, 0, 0, 0, 0};
        signed char wq[10];
        float sc[2];
        ug_quantize_weights_s8(wf, 2, 5, wq, sc);
        bad += sc[0] != 1.0f / 127.f || wq[1] != -127 || wq[0] != 32 || wq[3] != 0; /* 31.75 -> 32 */
        bad += sc[1] != 1.f || wq[5] != 0;                                          /* all-zero channel */
        float xf[5] = {-10.f, 0.f, 0.1f, 1.0f, 100.f};
        unsigned char xq[5];
        ug_quantize_u8(xf, 5, 0.1f, 3, xq);
        bad += xq[0] != 0 || xq[1] != 3 || xq[2] != 4 || xq[3] != 13 || xq[4] != 255;
        for (int a = 0; a <= 4; ++a) bad += ug_qalgo_name((ug_qalgo)a)[0] == '?';
        bad += ug_qalgo_name((ug_qalgo)42)[0] != '?';
        ug_conv_desc d = {1, 4, 8, 8, 4, 3, 3, 1, 1, 1, 1};
        signed char w[144] = {0};
        float s4[4] = {1, 1, 1, 1};
        bad += ug_qconv_plan_create(&d, w, s4, NULL, 0, 1.f, 256, UG_QALGO_AUTO, 1) != NULL; /* zp out of range */
        bad += ug_qconv_plan_create(&d, w, NULL, NULL, 0, 1.f, 0, UG_QALGO_AUTO, 1) != NULL;
        ug_conv_desc d5 = d; d5.R = d5.S = 5; d5.pad_h = d5.pad_w = 2;
        bad += ug_qconv_plan_create(&d5, w, s4, NULL, 0, 1.f, 0, UG_QALGO_WINOGRAD_F2, 1) != NULL; /* not eligible */
        ug_conv_desc dbig = d; dbig.C = UG_QWINO_MAX_C_TEST;
        bad += ug_qalgo_eligible(&dbig, UG_QALGO_WINOGRAD_F2);   /* beyond the exactness bound */
        bad += ug_qalgo_cost(&d5, UG_QALGO_WINOGRAD_F2, 1) >= 0;
        ug_qconv_plan *p = ug_qconv_plan_create(&d, w, s4, NULL, 0, 1.f, 0, UG_QALGO_DIRECT, 1);
        unsigned char xin[256] = {0}, yo[256];
        bad += ug_qconv_execute_u8(p, xin, yo, 0.f, 0) != -1;    /* non-positive out_scale */
        bad += ug_qconv_execute_s32(NULL, xin, (int *)yo) != -1;
        ug_qconv_plan_destroy(p);
        ++runs;
        if (bad) { ++fails; printf("FAIL api checks (%d)\n", bad); } else printf("ok   api checks\n");
    }
    printf("%d/%d passed\n", runs - fails, runs);
    return fails ? 1 : 0;
}
