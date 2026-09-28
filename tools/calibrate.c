/* Fit the cost-model constants on THIS machine (checklist D0: measured beats assumed).
 *
 *   ./build/calibrate [-o costmodel.txt]
 *
 * 1. Measures the core clock with a dependency-free FMA loop (2 FMA/cycle).
 * 2. Times every eligible algorithm, single-threaded, on a deterministic set of
 *    synthetic layers (the FIT set) and gets each run's feature vector f.
 * 3. Solves  min sum_rows ( (f . c) / cycles_measured - 1 )^2,  c >= 0
 *    (non-negative least squares on relative error, coordinate descent).
 * 4. Validates on the VGG-16 / ResNet-50 layers, which the fit never saw:
 *    prediction error per run and whether the model picks the measured-fastest
 *    algorithm per layer.
 * Writes the constants to a file that UGCONV_COSTMODEL=<file> loads at run time.
 */
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ugconv.h"
#include "../bench/layers.h"

#define NF 12 /* >= UG_CM_NFEAT and UG_QCM_NFEAT */
static int INT8 = 0, nf = UG_CM_NFEAT;
static const ug_algo algos[] = {UG_ALGO_DIRECT, UG_ALGO_IM2COL_GEMM, UG_ALGO_WINOGRAD_F4, UG_ALGO_WINOGRAD_F2};
static const ug_qalgo qalgos[] = {UG_QALGO_DIRECT, UG_QALGO_IM2COL_GEMM, UG_QALGO_WINOGRAD_F2};
#define NA 4
static int n_algos(void) { return INT8 ? 3 : 4; }
static int eligible(const ug_conv_desc *d, int a) { return INT8 ? ug_qalgo_eligible(d, qalgos[a]) : ug_algo_eligible(d, algos[a]); }
static const char *aname(int a) { return INT8 ? ug_qalgo_name(qalgos[a]) : ug_algo_name(algos[a]); }
static void features(const ug_conv_desc *d, int a, double *f)
{
    double bytes;
    memset(f, 0, NF * sizeof(double));
    if (INT8) ug_qcostmodel_features(d, qalgos[a], f);
    else ug_costmodel_features(d, algos[a], 1, f, &bytes);
}
static void cm_get(double *c) { if (INT8) ug_qcostmodel_get(c); else ug_costmodel_get(c); }
static void cm_set(const double *c) { if (INT8) ug_qcostmodel_set(c); else ug_costmodel_set(c); }
static const char *fname(int i) { return INT8 ? ug_qcm_feature_names[i] : ug_cm_feature_names[i]; }
static int pick(const ug_conv_desc *d)
{
    if (INT8) { ug_qalgo q = ug_qselect_algo(d, 1); for (int a = 0; a < 3; ++a) if (qalgos[a] == q) return a; return 0; }
    ug_algo p = ug_select_algo(d, 1);
    for (int a = 0; a < 4; ++a) if (algos[a] == p) return a;
    return 0;
}

static double measure_hz(void)
{
    long n = 200000000;
    double best = 0;
    for (int rep = 0; rep < 3; ++rep) {
        long it = n;
        double t = omp_get_wtime();
        __asm__ volatile(
            "vxorps %%ymm0,%%ymm0,%%ymm0\n vxorps %%ymm1,%%ymm1,%%ymm1\n vxorps %%ymm2,%%ymm2,%%ymm2\n"
            "vxorps %%ymm3,%%ymm3,%%ymm3\n vxorps %%ymm4,%%ymm4,%%ymm4\n vxorps %%ymm5,%%ymm5,%%ymm5\n"
            "vxorps %%ymm6,%%ymm6,%%ymm6\n vxorps %%ymm7,%%ymm7,%%ymm7\n vxorps %%ymm8,%%ymm8,%%ymm8\n"
            "vxorps %%ymm9,%%ymm9,%%ymm9\n"
            "1:\n"
            "vfmadd231ps %%ymm10,%%ymm11,%%ymm0\n vfmadd231ps %%ymm10,%%ymm11,%%ymm1\n"
            "vfmadd231ps %%ymm10,%%ymm11,%%ymm2\n vfmadd231ps %%ymm10,%%ymm11,%%ymm3\n"
            "vfmadd231ps %%ymm10,%%ymm11,%%ymm4\n vfmadd231ps %%ymm10,%%ymm11,%%ymm5\n"
            "vfmadd231ps %%ymm10,%%ymm11,%%ymm6\n vfmadd231ps %%ymm10,%%ymm11,%%ymm7\n"
            "vfmadd231ps %%ymm10,%%ymm11,%%ymm8\n vfmadd231ps %%ymm10,%%ymm11,%%ymm9\n"
            "dec %0\n jnz 1b\n"
            : "+r"(it)::"xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "cc");
        t = omp_get_wtime() - t;
        double hz = n * 10.0 / t / 2.0;
        if (hz > best) best = hz;
    }
    return best;
}

static double time_qalgo(const ug_conv_desc *d, ug_qalgo a)
{
    int P = ug_out_h(d), Q = ug_out_w(d);
    size_t nx = (size_t)d->N * d->C * d->H * d->W, nw = (size_t)d->K * d->C * d->R * d->S;
    size_t ny = (size_t)d->N * d->K * P * Q;
    unsigned char *x = malloc(nx);
    signed char *w = malloc(nw);
    float *s = malloc(d->K * 4), *y = malloc(ny * 4);
    for (size_t i = 0; i < nx; ++i) x[i] = (unsigned char)((i * 2654435761u) >> 24);
    for (size_t i = 0; i < nw; ++i) w[i] = (signed char)((i * 40503u) >> 5);
    for (int i = 0; i < d->K; ++i) s[i] = 1e-3f;
    ug_qconv_plan *p = ug_qconv_plan_create(d, w, s, NULL, 1, 1.f / 255, 128, a, 1);
    ug_qconv_execute_f32(p, x, y);
    double best = 1e300, tot = 0;
    for (int rep = 0; rep < 100 && (rep < 5 || tot < 0.2); ++rep) {
        double t = omp_get_wtime();
        ug_qconv_execute_f32(p, x, y);
        t = omp_get_wtime() - t;
        tot += t;
        if (t < best) best = t;
    }
    ug_qconv_plan_destroy(p);
    free(x); free(w); free(s); free(y);
    return best;
}

static double time_algo_f(const ug_conv_desc *d, ug_algo a)
{
    int P = ug_out_h(d), Q = ug_out_w(d);
    size_t nx = (size_t)d->N * d->C * d->H * d->W, nw = (size_t)d->K * d->C * d->R * d->S;
    size_t ny = (size_t)d->N * d->K * P * Q;
    float *x = malloc(nx * 4), *w = malloc(nw * 4), *b = malloc(d->K * 4), *y = malloc(ny * 4);
    for (size_t i = 0; i < nx; ++i) x[i] = (float)((i * 2654435761u) % 1000) * 1e-3f - 0.5f;
    for (size_t i = 0; i < nw; ++i) w[i] = (float)((i * 40503u) % 1000) * 1e-4f - 0.05f;
    for (int i = 0; i < d->K; ++i) b[i] = 0.01f;
    ug_conv_plan *p = ug_conv_plan_create(d, w, b, 1, a, 1);
    ug_conv_execute(p, x, y);
    double best = 1e300, tot = 0;
    for (int rep = 0; rep < 100 && (rep < 5 || tot < 0.2); ++rep) {
        double t = omp_get_wtime();
        ug_conv_execute(p, x, y);
        t = omp_get_wtime() - t;
        tot += t;
        if (t < best) best = t;
    }
    ug_conv_plan_destroy(p);
    free(x); free(w); free(b); free(y);
    return best;
}

static double time_algo(const ug_conv_desc *d, int a)
{
    return INT8 ? time_qalgo(d, qalgos[a]) : time_algo_f(d, algos[a]);
}

typedef struct { double f[NF]; double cyc; int layer; int algo; } row;

static void nnls(const row *r, int n, double c[NF])
{
    /* normal equations of the relative-error problem */
    double G[NF][NF] = {{0}}, h[NF] = {0};
    for (int i = 0; i < n; ++i)
        for (int a = 0; a < nf; ++a) {
            double xa = r[i].f[a] / r[i].cyc;
            h[a] += xa;
            for (int b = 0; b < nf; ++b) G[a][b] += xa * r[i].f[b] / r[i].cyc;
        }
    for (int it = 0; it < 20000; ++it)
        for (int a = 0; a < nf; ++a) {
            if (G[a][a] <= 0) continue; /* feature never exercised: keep default */
            double g = -h[a];
            for (int b = 0; b < nf; ++b) g += G[a][b] * c[b];
            double v = c[a] - g / G[a][a];
            c[a] = v > 0 ? v : 0;
        }
}

static double predict(const double c[NF], const double f[NF])
{
    double s = 0;
    for (int i = 0; i < nf; ++i) s += f[i] * c[i];
    return s;
}

int main(int argc, char **argv)
{
    const char *out = NULL;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "-int8")) { INT8 = 1; nf = UG_QCM_NFEAT; }
    }
    if (!out) out = INT8 ? "qcostmodel.txt" : "costmodel.txt";
    printf("calibrating the %s cost model\n", INT8 ? "INT8" : "FP32");

    const double hz = measure_hz();
    printf("measured core clock: %.2f GHz (FMA throughput / 2)\n", hz * 1e-9);

    /* FIT set: deterministic synthetic layers spanning every feature. */
    static const int Cs[] = {3, 8, 16, 32, 64, 128, 256, 512};
    static const int Ks[] = {16, 32, 64, 96, 128, 256, 512};
    static const int HWs[] = {7, 12, 14, 20, 28, 40, 56, 80, 112};
    static const int Rs[] = {1, 3, 3, 3, 5, 7};
    ug_conv_desc fitset[64];
    int nfit = 0;
    unsigned seed = 12345;
    while (nfit < 64) {
        seed = seed * 1103515245u + 12345u;
        unsigned v = seed >> 8;
        ug_conv_desc d = {1, Cs[v % 8], HWs[(v / 8) % 9], 0, Ks[(v / 72) % 7], Rs[(v / 504) % 6], 0, 1, 1, 0, 0};
        d.W = d.H; d.S = d.R;
        d.stride_h = d.stride_w = (v / 3024) % 3 == 0 ? 2 : 1;
        d.pad_h = d.pad_w = d.R / 2;
        double mac = (double)d.K * d.C * d.R * d.S * ug_out_h(&d) * ug_out_w(&d);
        if (mac > 2.5e8 || mac < 2e6) continue;
        fitset[nfit++] = d;
    }

    row *rows = malloc(sizeof(row) * 64 * NA);
    int nr = 0;
    printf("timing %d synthetic FIT layers x eligible algorithms (1 thread)...\n", nfit);
    for (int i = 0; i < nfit; ++i)
        for (int a = 0; a < n_algos(); ++a) {
            if (!eligible(&fitset[i], a)) continue;
            features(&fitset[i], a, rows[nr].f);
            rows[nr].cyc = time_algo(&fitset[i], a) * hz;
            rows[nr].layer = i;
            rows[nr].algo = a;
            ++nr;
        }

    double c0[NF], c[NF];
    cm_get(c0);
    memcpy(c, c0, sizeof c);
    nnls(rows, nr, c);

    printf("\n%-24s %12s %12s\n", "feature", "default", "fitted");
    for (int i = 0; i < nf; ++i) printf("%-24s %12.5f %12.5f\n", fname(i), c0[i], c[i]);
    double e0 = 0, e1 = 0;
    for (int i = 0; i < nr; ++i) {
        e0 += fabs(log(predict(c0, rows[i].f) / rows[i].cyc));
        e1 += fabs(log(predict(c, rows[i].f) / rows[i].cyc));
    }
    printf("FIT set (%d runs): mean |log(pred/meas)|  default %.3f  fitted %.3f\n", nr, e0 / nr, e1 / nr);

    /* VALIDATION on the network layers. */
    printf("\nVALIDATION on VGG-16 + ResNet-50 layers (never used in the fit), 1 thread:\n");
    printf("%-9s %-14s %-15s %-15s %8s %8s\n", "net", "layer", "measured best", "model pick", "loss", "|logerr|");
    int agree = 0, nl = 0;
    double vloss = 0, verr = 0, tbest = 0, tpick = 0;
    int nvr = 0;
    cm_set(c);
    for (int li = 0; li < (int)(sizeof layers / sizeof layers[0]); ++li) {
        const ug_conv_desc *d = &layers[li].d;
        double t[NA];
        int bi = -1;
        double lerr = 0;
        int ln = 0;
        for (int a = 0; a < n_algos(); ++a) {
            t[a] = -1;
            if (!eligible(d, a)) continue;
            t[a] = time_algo(d, a);
            double f[NF];
            features(d, a, f);
            lerr += fabs(log(predict(c, f) / (t[a] * hz)));
            ++ln;
            if (bi < 0 || t[a] < t[bi]) bi = a;
        }
        const int pi = pick(d);
        agree += pi == bi;
        ++nl;
        vloss += t[pi] / t[bi];
        tbest += layers[li].mult * t[bi];
        tpick += layers[li].mult * t[pi];
        verr += lerr;
        nvr += ln;
        printf("%-9s %-14s %-15s %-15s %8.3f %8.3f\n", layers[li].net, layers[li].name, aname(bi),
               aname(pi), t[pi] / t[bi], lerr / ln);
    }
    printf("model picked the measured-fastest algorithm on %d/%d layers; network time with model picks = "
           "%.3fx the per-layer optimum; mean |log(pred/meas)| %.3f over %d runs\n",
           agree, nl, tpick / tbest, verr / nvr, nvr);

    FILE *fo = fopen(out, "w");
    if (fo) {
        for (int i = 0; i < nf; ++i) fprintf(fo, "%.8g  # %s\n", c[i], fname(i));
        fclose(fo);
        printf("\nwrote %s  (use: export %s=%s)\n", out, INT8 ? "UGCONV_QCOSTMODEL" : "UGCONV_COSTMODEL", out);
    }
    free(rows);
    return 0;
}
