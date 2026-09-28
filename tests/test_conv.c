/* Correctness: every algorithm vs an fp64 reference, on shapes that hit every tail
 * (K not a multiple of 6, widths not multiples of 8/16, tile rows not multiples of 8,
 * strides 1-3, padding 0-3, 1x1 / 3x3 / 5x5 / 7x7, relu on/off, N > 1).
 *
 * Error metric per output: |y - y_ref| / (sum |w*x| + |bias|), i.e. relative to the
 * magnitude the rounding error is proportional to. It is independent of cancellation.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "ugconv.h"

static unsigned long long rng = 88172645463325252ull;
static float frand(void)
{
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((rng >> 11) * (1.0 / 9007199254740992.0)) * 2.f - 1.f;
}

static void reference(const ug_conv_desc *d, const float *x, const float *w, const float *b, int relu,
                      double *y, double *mag)
{
    int P = ug_out_h(d), Q = ug_out_w(d);
    for (int n = 0; n < d->N; ++n)
        for (int k = 0; k < d->K; ++k)
            for (int oh = 0; oh < P; ++oh)
                for (int ow = 0; ow < Q; ++ow) {
                    double s = b ? b[k] : 0, m = b ? fabs(b[k]) : 0;
                    for (int c = 0; c < d->C; ++c)
                        for (int r = 0; r < d->R; ++r)
                            for (int q = 0; q < d->S; ++q) {
                                int ih = oh * d->stride_h - d->pad_h + r, iw = ow * d->stride_w - d->pad_w + q;
                                if (ih < 0 || ih >= d->H || iw < 0 || iw >= d->W) continue;
                                double t = (double)w[((k * d->C + c) * d->R + r) * d->S + q] *
                                           x[((n * d->C + c) * d->H + ih) * d->W + iw];
                                s += t; m += fabs(t);
                            }
                    size_t o = (((size_t)n * d->K + k) * P + oh) * Q + ow;
                    y[o] = relu && s < 0 ? 0 : s;
                    mag[o] = m + 1e-30;
                }
}

int main(void)
{
    const ug_conv_desc shapes[] = {
        /* N  C  H  W   K  R S sh sw ph pw */
        {1, 3, 17, 19, 7, 3, 3, 1, 1, 1, 1},
        {2, 5, 23, 37, 13, 3, 3, 1, 1, 1, 1},
        {1, 16, 34, 70, 16, 3, 3, 1, 1, 1, 1},
        {1, 8, 9, 9, 6, 3, 3, 1, 1, 0, 0},
        {1, 32, 14, 14, 20, 3, 3, 1, 1, 1, 1},
        {1, 11, 7, 7, 9, 3, 3, 1, 1, 1, 1},
        {1, 130, 12, 13, 25, 3, 3, 1, 1, 1, 1},    /* C > 128: Winograd channel chunks */
        {1, 24, 20, 21, 200, 3, 3, 1, 1, 1, 1},    /* Kp > 192: Winograd K blocks      */
        {2, 3, 31, 29, 10, 7, 7, 2, 2, 3, 3},      /* stem-like                         */
        {1, 7, 19, 22, 11, 5, 5, 1, 1, 2, 2},
        {1, 9, 25, 26, 14, 3, 3, 2, 2, 1, 1},
        {1, 6, 21, 20, 8, 3, 3, 3, 3, 1, 1},
        {1, 6, 21, 20, 8, 3, 1, 2, 1, 1, 0},
        {1, 40, 15, 17, 33, 1, 1, 1, 1, 0, 0},
        {2, 17, 16, 16, 30, 1, 1, 2, 2, 0, 0},
        {1, 300, 5, 6, 7, 1, 1, 1, 1, 0, 0},       /* CRS > KC                          */
        {1, 64, 18, 18, 64, 3, 3, 1, 1, 1, 1},
        {1, 4, 70, 70, 12, 3, 3, 2, 2, 1, 1},      /* stride-2 vector im2col path (row >= 32) */
        {1, 5, 40, 67, 9, 1, 1, 2, 2, 0, 0},       /* 1x1 stride 2, wide rows */
    };
    const ug_algo algos[] = {UG_ALGO_DIRECT, UG_ALGO_IM2COL_GEMM, UG_ALGO_WINOGRAD_F4, UG_ALGO_WINOGRAD_F2, UG_ALGO_AUTO, UG_ALGO_TUNE};
    const double tol[] = {1e-5, 1e-5, 2e-4, 3e-5, 2e-4, 2e-4};
    int fails = 0, runs = 0;

    for (unsigned si = 0; si < sizeof shapes / sizeof shapes[0]; ++si) {
        const ug_conv_desc *d = &shapes[si];
        int P = ug_out_h(d), Q = ug_out_w(d);
        size_t nx = (size_t)d->N * d->C * d->H * d->W, nw = (size_t)d->K * d->C * d->R * d->S;
        size_t ny = (size_t)d->N * d->K * P * Q;
        float *x = malloc(nx * 4), *w = malloc(nw * 4), *b = malloc(d->K * 4), *y = malloc(ny * 4);
        double *yr = malloc(ny * 8), *mag = malloc(ny * 8);
        for (size_t i = 0; i < nx; ++i) x[i] = frand();
        for (size_t i = 0; i < nw; ++i) w[i] = frand();
        for (int i = 0; i < d->K; ++i) b[i] = frand();
        for (int relu = 0; relu < 2; ++relu) {
            reference(d, x, w, relu ? b : NULL, relu, yr, mag);
            for (unsigned ai = 0; ai < sizeof algos / sizeof algos[0]; ++ai) {
                if (algos[ai] != UG_ALGO_AUTO && algos[ai] != UG_ALGO_TUNE && !ug_algo_eligible(d, algos[ai])) continue;
                for (int nt = 1; nt <= 3; nt += 2) {
                    ug_conv_plan *p = ug_conv_plan_create(d, w, relu ? b : NULL, relu, algos[ai], nt);
                    if (!p) { printf("FAIL plan_create shape %u algo %s\n", si, ug_algo_name(algos[ai])); ++fails; continue; }
                    for (size_t i = 0; i < ny; ++i) y[i] = NAN;
                    ug_conv_execute(p, x, y);
                    double worst = 0;
                    for (size_t i = 0; i < ny; ++i) {
                        double e = fabs(y[i] - yr[i]) / mag[i];
                        if (!(e <= worst)) worst = e; /* catches NaN (unwritten outputs) */
                    }
                    ++runs;
                    int ok = worst <= tol[ai];
                    if (!ok) ++fails;
                    printf("%s shape %2u relu %d nt %d %-16s -> %-16s err %.2e\n", ok ? "ok  " : "FAIL", si, relu, nt,
                           ug_algo_name(algos[ai]), ug_algo_name(ug_conv_plan_algo(p)), worst);
                    ug_conv_plan_destroy(p);
                }
            }
        }
        free(x); free(w); free(b); free(y); free(yr); free(mag);
    }
    /* API checks: names/groups, cost-model get/set and file loading, invalid input */
    {
        int bad = 0;
        const ug_algo all[] = {UG_ALGO_AUTO, UG_ALGO_DIRECT, UG_ALGO_IM2COL_GEMM, UG_ALGO_WINOGRAD_F4, UG_ALGO_WINOGRAD_F2, UG_ALGO_TUNE};
        const int grp[] = {0, 1, 2, 3, 3, 0};
        for (int i = 0; i < 6; ++i) bad += ug_algo_group(all[i]) != grp[i] || ug_algo_name(all[i])[0] == '?';
        bad += ug_algo_name((ug_algo)99)[0] != '?' || ug_algo_group((ug_algo)99) != 0;
        double c[UG_CM_NFEAT], c2[UG_CM_NFEAT];
        ug_costmodel_get(c);
        for (int i = 0; i < UG_CM_NFEAT; ++i) c2[i] = c[i] * 2;
        ug_costmodel_set(c2);
        ug_conv_desc dd = shapes[16];
        double t2 = ug_algo_cost(&dd, UG_ALGO_IM2COL_GEMM, 1);
        ug_costmodel_set(c);
        double t1 = ug_algo_cost(&dd, UG_ALGO_IM2COL_GEMM, 1);
        bad += !(t2 > t1 * 1.5);                         /* doubling every constant ~doubles compute time */
        ug_conv_desc inval = {1, 3, 2, 2, 4, 5, 5, 1, 1, 0, 0}; /* kernel larger than input */
        float wtmp[300] = {0};
        bad += ug_conv_plan_create(&inval, wtmp, NULL, 0, UG_ALGO_AUTO, 1) != NULL;
        bad += ug_algo_cost(&inval, UG_ALGO_DIRECT, 1) >= 0;
        bad += ug_algo_cost(&shapes[9], UG_ALGO_WINOGRAD_F4, 1) >= 0; /* 5x5 not Winograd-eligible */
        ++runs;
        if (bad) { ++fails; printf("FAIL api checks (%d)\n", bad); } else printf("ok   api checks\n");
    }
    printf("%d/%d passed\n", runs - fails, runs);
    return fails ? 1 : 0;
}
