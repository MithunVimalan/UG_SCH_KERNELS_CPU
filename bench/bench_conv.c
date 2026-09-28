/* Benchmark: every algorithm group on VGG-16 and ResNet-50 (batch 1) conv layers.
 *
 *   ./build/bench_conv [-t threads] [-n net] [-naive] [-csv file]
 *     net: vgg16 | resnet50 | all (default all)
 *
 * Baselines: OpenBLAS explicit im2col + cblas_sgemm + bias/ReLU pass, and a naive
 * 7-deep loop (-naive, single thread). Every output is checked against the
 * OpenBLAS result. Times are min over >= 5 runs and >= 0.25 s.
 */
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ugconv.h"
#ifndef NO_OPENBLAS
#include <cblas.h>
void openblas_set_num_threads(int);
#endif

#include "layers.h"
#define NL (int)(sizeof layers / sizeof layers[0])
static const ug_algo algos[] = {UG_ALGO_DIRECT, UG_ALGO_IM2COL_GEMM, UG_ALGO_WINOGRAD_F4, UG_ALGO_WINOGRAD_F2};
#define NA 4

typedef void (*run_fn)(void *);
static double time_it(run_fn f, void *arg)
{
    f(arg); /* warm-up */
    double best = 1e300, total = 0;
    for (int rep = 0; rep < 200 && (rep < 5 || total < 0.25); ++rep) {
        double t0 = omp_get_wtime();
        f(arg);
        double dt = omp_get_wtime() - t0;
        total += dt;
        if (dt < best) best = dt;
    }
    return best;
}

typedef struct { const ug_conv_plan *p; const float *x; float *y; } ug_arg;
static void run_ug(void *a) { ug_arg *u = a; ug_conv_execute(u->p, u->x, u->y); }

typedef struct {
    const ug_conv_desc *d; const float *x, *w, *b; float *y, *col; int nt;
} ref_arg;

#ifndef NO_OPENBLAS
static void run_openblas(void *a)
{
    ref_arg *r = a;
    const ug_conv_desc *d = r->d;
    int P = ug_out_h(d), Q = ug_out_w(d), PQ = P * Q, CRS = d->C * d->R * d->S;
    int pointwise = d->R == 1 && d->S == 1 && d->stride_h == 1 && d->stride_w == 1 && d->pad_h == 0 && d->pad_w == 0;
    const float *col = r->x;
    if (!pointwise) {
#pragma omp parallel for num_threads(r->nt) schedule(static)
        for (int row = 0; row < CRS; ++row) {
            int c = row / (d->R * d->S), rr = (row / d->S) % d->R, s = row % d->S;
            float *dst = r->col + (size_t)row * PQ;
            for (int oh = 0; oh < P; ++oh) {
                int ih = oh * d->stride_h - d->pad_h + rr;
                for (int ow = 0; ow < Q; ++ow) {
                    int iw = ow * d->stride_w - d->pad_w + s;
                    dst[oh * Q + ow] = (ih >= 0 && ih < d->H && iw >= 0 && iw < d->W)
                                           ? r->x[((size_t)c * d->H + ih) * d->W + iw] : 0.f;
                }
            }
        }
        col = r->col;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, d->K, PQ, CRS, 1.f, r->w, CRS, col, PQ, 0.f, r->y, PQ);
#pragma omp parallel for num_threads(r->nt) schedule(static)
    for (int k = 0; k < d->K; ++k)
        for (int i = 0; i < PQ; ++i) {
            float v = r->y[(size_t)k * PQ + i] + r->b[k];
            r->y[(size_t)k * PQ + i] = v > 0 ? v : 0;
        }
}
#endif

static void run_naive(void *a)
{
    ref_arg *r = a;
    const ug_conv_desc *d = r->d;
    int P = ug_out_h(d), Q = ug_out_w(d);
    for (int k = 0; k < d->K; ++k)
        for (int oh = 0; oh < P; ++oh)
            for (int ow = 0; ow < Q; ++ow) {
                float s = r->b[k];
                for (int c = 0; c < d->C; ++c)
                    for (int rr = 0; rr < d->R; ++rr)
                        for (int q = 0; q < d->S; ++q) {
                            int ih = oh * d->stride_h - d->pad_h + rr, iw = ow * d->stride_w - d->pad_w + q;
                            if (ih >= 0 && ih < d->H && iw >= 0 && iw < d->W)
                                s += r->w[((k * d->C + c) * d->R + rr) * d->S + q] * r->x[((size_t)c * d->H + ih) * d->W + iw];
                        }
                r->y[((size_t)k * P + oh) * Q + ow] = s > 0 ? s : 0;
            }
}

/* End-to-end: every layer instance of a network (multiplicity expanded) with its own
 * weights and activation buffers, run back to back, so weights are cold as in real
 * inference. strategy: 0 auto, 1 G1 only, 2 G2 only, 3 F4 else G2, 4 F2 else G2, 5 OpenBLAS. */
static double run_network(const char *net, int nt, int strategy)
{
    int n = 0;
    for (int li = 0; li < NL; ++li) if (!strcmp(layers[li].net, net)) n += layers[li].mult;
    ug_conv_plan **plans = calloc(n, sizeof *plans);
    ref_arg *ra = calloc(n, sizeof *ra);
    float **xs = calloc(n, sizeof *xs), **ys = calloc(n, sizeof *ys), **ws = calloc(n, sizeof *ws), **bs = calloc(n, sizeof *bs);
    float *col = NULL;
    size_t colmax = 0;
    int i = 0;
    for (int li = 0; li < NL; ++li) {
        if (strcmp(layers[li].net, net)) continue;
        const ug_conv_desc *d = &layers[li].d;
        int P = ug_out_h(d), Q = ug_out_w(d);
        size_t nx = (size_t)d->C * d->H * d->W, nw = (size_t)d->K * d->C * d->R * d->S, ny = (size_t)d->K * P * Q;
        size_t nc = (size_t)d->C * d->R * d->S * P * Q;
        if (nc > colmax) colmax = nc;
        for (int m = 0; m < layers[li].mult; ++m, ++i) {
            xs[i] = malloc(nx * 4); ys[i] = malloc(ny * 4); ws[i] = malloc(nw * 4); bs[i] = malloc(d->K * 4);
            for (size_t j = 0; j < nx; ++j) xs[i][j] = rand() / (float)RAND_MAX - 0.5f;
            for (size_t j = 0; j < nw; ++j) ws[i][j] = (rand() / (float)RAND_MAX - 0.5f) * 0.05f;
            for (int j = 0; j < d->K; ++j) bs[i][j] = 0.01f;
            ug_algo a = UG_ALGO_AUTO;
            if (strategy == 1) a = UG_ALGO_DIRECT;
            if (strategy == 2) a = UG_ALGO_IM2COL_GEMM;
            if (strategy == 3) a = ug_algo_eligible(d, UG_ALGO_WINOGRAD_F4) ? UG_ALGO_WINOGRAD_F4 : UG_ALGO_IM2COL_GEMM;
            if (strategy == 4) a = ug_algo_eligible(d, UG_ALGO_WINOGRAD_F2) ? UG_ALGO_WINOGRAD_F2 : UG_ALGO_IM2COL_GEMM;
            if (strategy != 5) plans[i] = ug_conv_plan_create(d, ws[i], bs[i], 1, a, nt);
            ra[i] = (ref_arg){d, xs[i], ws[i], bs[i], ys[i], NULL, nt};
        }
    }
    if (strategy == 5) { col = malloc(colmax * 4); for (int j = 0; j < n; ++j) ra[j].col = col; }
    double best = 1e300;
    for (int rep = 0; rep < 4; ++rep) {
        double t0 = omp_get_wtime();
        for (int j = 0; j < n; ++j) {
            if (strategy == 5) {
#ifndef NO_OPENBLAS
                run_openblas(&ra[j]);
#endif
            } else ug_conv_execute(plans[j], xs[j], ys[j]);
        }
        double dt = omp_get_wtime() - t0;
        if (rep > 0 && dt < best) best = dt;  /* rep 0 = warm-up */
    }
    for (int j = 0; j < n; ++j) { ug_conv_plan_destroy(plans[j]); free(xs[j]); free(ys[j]); free(ws[j]); free(bs[j]); }
    free(plans); free(ra); free(xs); free(ys); free(ws); free(bs); free(col);
    return best;
}

/* ---------------- INT8 ---------------- */
typedef struct { const ug_qconv_plan *p; const unsigned char *x; float *y; } q_arg;
static void run_q(void *a) { q_arg *u = a; ug_qconv_execute_f32(u->p, u->x, u->y); }

static const ug_qalgo qalgos[] = {UG_QALGO_DIRECT, UG_QALGO_IM2COL_GEMM, UG_QALGO_WINOGRAD_F2};
#define NQ 3

/* whole network in int8 (auto), same structure as run_network */
static double run_network_q(const char *net, int nt)
{
    int n = 0;
    for (int li = 0; li < NL; ++li) if (!strcmp(layers[li].net, net)) n += layers[li].mult;
    ug_qconv_plan **plans = calloc(n, sizeof *plans);
    unsigned char **xs = calloc(n, sizeof *xs);
    float **ys = calloc(n, sizeof *ys);
    int i = 0;
    for (int li = 0; li < NL; ++li) {
        if (strcmp(layers[li].net, net)) continue;
        const ug_conv_desc *d = &layers[li].d;
        size_t nx = (size_t)d->C * d->H * d->W, nw = (size_t)d->K * d->C * d->R * d->S;
        size_t ny = (size_t)d->K * ug_out_h(d) * ug_out_w(d);
        for (int m = 0; m < layers[li].mult; ++m, ++i) {
            signed char *w = malloc(nw);
            float *sc = malloc(d->K * 4), *b = malloc(d->K * 4);
            xs[i] = malloc(nx); ys[i] = malloc(ny * 4);
            for (size_t j = 0; j < nx; ++j) xs[i][j] = (unsigned char)rand();
            for (size_t j = 0; j < nw; ++j) w[j] = (signed char)(rand() % 255 - 127);
            for (int j = 0; j < d->K; ++j) { sc[j] = 1e-3f; b[j] = 0.01f; }
            plans[i] = ug_qconv_plan_create(d, w, sc, b, 1, 1.f / 255, 128, UG_QALGO_AUTO, nt);
            free(w); free(sc); free(b);
        }
    }
    double best = 1e300;
    for (int rep = 0; rep < 4; ++rep) {
        double t0 = omp_get_wtime();
        for (int j = 0; j < n; ++j) ug_qconv_execute_f32(plans[j], xs[j], ys[j]);
        double dt = omp_get_wtime() - t0;
        if (rep > 0 && dt < best) best = dt;
    }
    for (int j = 0; j < n; ++j) { ug_qconv_plan_destroy(plans[j]); free(xs[j]); free(ys[j]); }
    free(plans); free(xs); free(ys);
    return best;
}

static int bench_int8(const char *net, int nt, const char *csv)
{
    printf("INT8 VNNI backend: %s\n", ug_qvnni_backend());
    printf("activations: fp32 in [-0.5,0.5) -> u8 (scale 1/255, zp 128); weights: per-channel symmetric s8\n");
    printf("err = max|y_int8 - y_fp32| / max|y_fp32| against the fp32 convolution of the unquantised data\n\n");
    printf("%-9s %-14s %3s %8s %8s %8s %8s | %8s %8s | %-15s %6s %6s %8s\n", "net", "layer", "x", "Q1 ms", "Q2 ms",
           "Q3 ms", "qauto", "f32auto", "OBLAS", "int8 pick", "vsF32", "vsOB", "err");
    FILE *fc = csv ? fopen(csv, "w") : NULL;
    if (fc) fprintf(fc, "net,layer,mult,gmac,Q1_ms,Q2_ms,Q3_ms,qauto_algo,qauto_ms,qbest_ms,f32auto_ms,openblas_ms,err_Q1,err_Q2,err_Q3\n");
    double tq = 0, tqb = 0, tf = 0, tob = 0, tmac = 0;
    int agree = 0, cnt = 0;
    for (int li = 0; li < NL; ++li) {
        const layer *l = &layers[li];
        if (strcmp(net, "all") && strcmp(net, l->net)) continue;
        const ug_conv_desc *d = &l->d;
        int P = ug_out_h(d), Q = ug_out_w(d);
        size_t nx = (size_t)d->C * d->H * d->W, nw = (size_t)d->K * d->C * d->R * d->S, ny = (size_t)d->K * P * Q;
        float *x = malloc(nx * 4), *w = malloc(nw * 4), *b = malloc(d->K * 4), *yf = malloc(ny * 4), *yq = malloc(ny * 4);
        float *col = malloc((size_t)d->C * d->R * d->S * P * Q * 4), *yob = malloc(ny * 4);
        unsigned char *xq = malloc(nx);
        signed char *wq = malloc(nw);
        float *ws = malloc(d->K * 4);
        srand(li + 1);
        for (size_t i = 0; i < nx; ++i) x[i] = rand() / (float)RAND_MAX - 0.5f;
        float wsc = 1.f / sqrtf((float)d->C * d->R * d->S);
        for (size_t i = 0; i < nw; ++i) w[i] = (rand() / (float)RAND_MAX - 0.5f) * wsc;
        for (int i = 0; i < d->K; ++i) b[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
        const float xs = 1.f / 255;
        const int zp = 128;
        ug_quantize_u8(x, (long)nx, xs, zp, xq);
        ug_quantize_weights_s8(w, d->K, d->C * d->R * d->S, wq, ws);
        double gmac = (double)d->K * d->C * d->R * d->S * P * Q * 1e-9;

        ug_conv_plan *pf = ug_conv_plan_create(d, w, b, 1, UG_ALGO_AUTO, nt);
        ug_arg ua = {pf, x, yf};
        double t_f = time_it(run_ug, &ua);
        ug_conv_plan_destroy(pf);
        double t_ob = -1;
#ifndef NO_OPENBLAS
        ref_arg ra = {d, x, w, b, yob, col, nt};
        t_ob = time_it(run_openblas, &ra);
#endif
        double fmax_ = 0;
        for (size_t i = 0; i < ny; ++i) fmax_ = fmax(fmax_, fabs(yf[i]));
        double t[NQ], err[NQ];
        int bi = -1;
        for (int a = 0; a < NQ; ++a) {
            t[a] = -1; err[a] = -1;
            if (!ug_qalgo_eligible(d, qalgos[a])) continue;
            ug_qconv_plan *p = ug_qconv_plan_create(d, wq, ws, b, 1, xs, zp, qalgos[a], nt);
            q_arg qa = {p, xq, yq};
            t[a] = time_it(run_q, &qa);
            double e = 0;
            for (size_t i = 0; i < ny; ++i) e = fmax(e, fabs(yq[i] - yf[i]));
            err[a] = e / fmax_;
            ug_qconv_plan_destroy(p);
            if (bi < 0 || t[a] < t[bi]) bi = a;
        }
        ug_qalgo pick = ug_qselect_algo(d, nt);
        int pi = 0;
        for (int a = 0; a < NQ; ++a) if (qalgos[a] == pick) pi = a;
        printf("%-9s %-14s %3d", l->net, l->name, l->mult);
        for (int a = 0; a < NQ; ++a) { if (t[a] > 0) printf(" %8.3f", t[a] * 1e3); else printf(" %8s", "-"); }
        printf(" %8.3f | %8.3f %8.3f | %-15s %5.2fx %5.2fx %8.2e\n", t[pi] * 1e3, t_f * 1e3, t_ob * 1e3,
               ug_qalgo_name(pick), t_f / t[pi], t_ob / t[pi], err[pi]);
        if (fc) fprintf(fc, "%s,%s,%d,%.6f,%.6f,%.6f,%.6f,%s,%.6f,%.6f,%.6f,%.6f,%.3e,%.3e,%.3e\n", l->net, l->name, l->mult, gmac,
                        t[0] * 1e3, t[1] * 1e3, t[2] * 1e3, ug_qalgo_name(pick), t[pi] * 1e3, t[bi] * 1e3, t_f * 1e3,
                        t_ob * 1e3, err[0], err[1], err[2]);
        tq += l->mult * t[pi]; tqb += l->mult * t[bi]; tf += l->mult * t_f; tob += l->mult * t_ob; tmac += l->mult * gmac;
        agree += pi == bi; ++cnt;
        free(x); free(w); free(b); free(yf); free(yq); free(col); free(yob); free(xq); free(wq); free(ws);
    }
    printf("\nTotals (%.2f GMAC): int8 auto %.2f ms | int8 best-per-layer %.2f ms | fp32 auto %.2f ms | OpenBLAS fp32 %.2f ms\n",
           tmac, tq * 1e3, tqb * 1e3, tf * 1e3, tob * 1e3);
    printf("  int8 auto vs fp32 auto %.2fx, vs OpenBLAS fp32 %.2fx; int8 selector picked the fastest on %d/%d layers; "
           "%.1f effective GOP/s\n", tf / tq, tob / tq, agree, cnt, 2 * tmac / tq);
    if (fc) fclose(fc);
    return 0;
}

static double max_rel_diff(const float *a, const float *b, size_t n)
{
    double m = 0, s = 0;
    for (size_t i = 0; i < n; ++i) { double e = fabs(a[i] - b[i]); if (!(e <= m)) m = e; if (fabs(b[i]) > s) s = fabs(b[i]); }
    return m / (s + 1e-30);
}

int main(int argc, char **argv)
{
    int nt = omp_get_max_threads(), naive = 0, e2e = 0, int8 = 0;
    const char *net = "all", *csv = NULL;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-t") && i + 1 < argc) nt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) net = argv[++i];
        else if (!strcmp(argv[i], "-naive")) naive = 1;
        else if (!strcmp(argv[i], "-e2e")) e2e = 1;
        else if (!strcmp(argv[i], "-int8")) int8 = 1;
        else if (!strcmp(argv[i], "-csv") && i + 1 < argc) csv = argv[++i];
    }
    ug_cpu_info ci;
    ug_cpu_detect(&ci);
    printf("CPU: %s | family %d model 0x%X | AVX2 %d FMA %d AVX-VNNI %d VNNI-INT8 %d AVX512F %d hybrid %d | Arrow Lake-S: %s\n",
           ci.brand, ci.family, ci.model, ci.avx2, ci.fma, ci.avx_vnni, ci.avx_vnni_int8, ci.avx512f, ci.hybrid,
           ci.is_arrow_lake_s ? "yes" : "NO (numbers below are not target numbers)");
    printf("L1d %ld KB  L2 %ld KB  L3 %ld KB | threads %d | fp32 kernels: AVX2+FMA | int8 kernels: %s\n\n",
           ci.l1d_bytes >> 10, ci.l2_bytes >> 10, ci.l3_bytes >> 10, nt, ug_qvnni_backend());
#ifndef NO_OPENBLAS
    openblas_set_num_threads(nt);
#endif
    if (int8 && !e2e) return bench_int8(net, nt, csv);
    if (e2e) {
        static const char *nets[] = {"vgg16", "resnet50"};
        static const char *sn[] = {"auto", "G1 only", "G2 only", "G3-F4 else G2", "G3-F2 else G2", "OpenBLAS"};
        printf("End-to-end conv stack, cold weights, min of 3 runs:\n");
        for (int k = 0; k < 2; ++k) {
            if (strcmp(net, "all") && strcmp(net, nets[k])) continue;
            double t[6];
            for (int st = 0; st < 6; ++st) t[st] = run_network(nets[k], nt, st);
            printf("  %-9s", nets[k]);
            for (int st = 0; st < 6; ++st) printf(" | %s %.2f ms", sn[st], t[st] * 1e3);
            printf(" | auto vs OpenBLAS %.2fx, vs best single group %.2fx\n", t[5] / t[0],
                   fmin(fmin(t[1], t[2]), fmin(t[3], t[4])) / t[0]);
            if (int8) {
                double tq = run_network_q(nets[k], nt);
                printf("  %-9s | INT8 auto %.2f ms | vs fp32 auto %.2fx | vs OpenBLAS fp32 %.2fx\n", nets[k], tq * 1e3,
                       t[0] / tq, t[5] / tq);
            }
        }
        return 0;
    }
    FILE *fc = csv ? fopen(csv, "w") : NULL;
    if (fc) fprintf(fc, "net,layer,mult,C,H,K,R,stride,gmac,G1_ms,G2_ms,G3F4_ms,G3F2_ms,openblas_ms,naive_ms,auto_algo,best_algo,auto_ms,best_ms,errG1,errG2,errF4,errF2\n");

    printf("%-9s %-14s %3s %9s %9s %9s %9s %9s %9s | %-15s %-15s %6s\n", "net", "layer", "x", "GMAC",
           "G1 ms", "G2 ms", "G3F4 ms", "G3F2 ms", "OBLAS ms", "auto", "measured best", "auto/best");
    double tot_auto = 0, tot_best = 0, tot_ob = 0, tot_naive = 0, tot_g[NA] = {0}, tot_mac = 0;
    int agree = 0, count = 0;
    for (int li = 0; li < NL; ++li) {
        const layer *l = &layers[li];
        if (strcmp(net, "all") && strcmp(net, l->net)) continue;
        const ug_conv_desc *d = &l->d;
        int P = ug_out_h(d), Q = ug_out_w(d);
        size_t nx = (size_t)d->C * d->H * d->W, nw = (size_t)d->K * d->C * d->R * d->S, ny = (size_t)d->K * P * Q;
        float *x = malloc(nx * 4), *w = malloc(nw * 4), *b = malloc(d->K * 4), *y = malloc(ny * 4), *yr = malloc(ny * 4);
        float *col = malloc((size_t)d->C * d->R * d->S * P * Q * 4);
        srand(li + 1);
        for (size_t i = 0; i < nx; ++i) x[i] = rand() / (float)RAND_MAX - 0.5f;
        float ws = 1.f / sqrtf((float)d->C * d->R * d->S);
        for (size_t i = 0; i < nw; ++i) w[i] = (rand() / (float)RAND_MAX - 0.5f) * ws;
        for (int i = 0; i < d->K; ++i) b[i] = (rand() / (float)RAND_MAX - 0.5f) * 0.1f;
        double gmac = (double)d->K * d->C * d->R * d->S * P * Q * 1e-9;

        ref_arg ra = {d, x, w, b, yr, col, nt};
        double t_ob = -1, t_nv = -1;
#ifndef NO_OPENBLAS
        t_ob = time_it(run_openblas, &ra);
#else
        run_naive(&ra);
#endif
        if (naive) { ref_arg rn = ra; float *yn = malloc(ny * 4); rn.y = yn; double t0 = omp_get_wtime(); run_naive(&rn); t_nv = omp_get_wtime() - t0; free(yn); }

        double t[NA], err[NA];
        for (int a = 0; a < NA; ++a) {
            t[a] = -1; err[a] = -1;
            if (!ug_algo_eligible(d, algos[a])) continue;
            ug_conv_plan *p = ug_conv_plan_create(d, w, b, 1, algos[a], nt);
            ug_arg ua = {p, x, y};
            t[a] = time_it(run_ug, &ua);
            err[a] = max_rel_diff(y, yr, ny);
            ug_conv_plan_destroy(p);
        }
        ug_algo aa = ug_select_algo(d, nt);
        int ai = 0, bi = -1;
        for (int a = 0; a < NA; ++a) {
            if (algos[a] == aa) ai = a;
            if (t[a] > 0 && (bi < 0 || t[a] < t[bi])) bi = a;
        }
        printf("%-9s %-14s %3d %9.3f", l->net, l->name, l->mult, gmac);
        for (int a = 0; a < NA; ++a) { if (t[a] > 0) printf(" %9.3f", t[a] * 1e3); else printf(" %9s", "-"); }
        printf(" %9.3f | %-15s %-15s %6.2f\n", t_ob * 1e3, ug_algo_name(aa), ug_algo_name(algos[bi]), t[ai] / t[bi]);
        if (fc) fprintf(fc, "%s,%s,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%s,%.6f,%.6f,%.3e,%.3e,%.3e,%.3e\n",
                        l->net, l->name, l->mult, d->C, d->H, d->K, d->R, d->stride_h, gmac,
                        t[0] * 1e3, t[1] * 1e3, t[2] * 1e3, t[3] * 1e3, t_ob * 1e3, t_nv * 1e3,
                        ug_algo_name(aa), ug_algo_name(algos[bi]), t[ai] * 1e3, t[bi] * 1e3, err[0], err[1], err[2], err[3]);
        tot_auto += l->mult * t[ai]; tot_best += l->mult * t[bi]; tot_ob += l->mult * t_ob; tot_naive += l->mult * t_nv;
        tot_mac += l->mult * gmac;
        for (int a = 0; a < NA; ++a) tot_g[a] += l->mult * (t[a] > 0 ? t[a] : t[1]); /* G3 falls back to G2 */
        agree += ai == bi; ++count;
        free(x); free(w); free(b); free(y); free(yr); free(col);
    }
    printf("\nTotals over the selected network(s), weighted by layer multiplicity (%.2f GMAC):\n", tot_mac);
    printf("  G1 only %.2f ms | G2 only %.2f ms | G3-F4 (else G2) %.2f ms | G3-F2 (else G2) %.2f ms\n",
           tot_g[0] * 1e3, tot_g[1] * 1e3, tot_g[2] * 1e3, tot_g[3] * 1e3);
    printf("  auto (cost model) %.2f ms | measured best per layer %.2f ms | OpenBLAS im2col+sgemm %.2f ms\n",
           tot_auto * 1e3, tot_best * 1e3, tot_ob * 1e3);
    if (naive) printf("  naive loop (1 thread) %.2f ms\n", tot_naive * 1e3);
    printf("  auto picked the measured-fastest algorithm on %d/%d layers; auto vs OpenBLAS speed-up %.2fx; "
           "effective %.1f GFLOP/s (direct-conv FLOPs)\n", agree, count, tot_ob / tot_auto, 2 * tot_mac / tot_auto);
    if (fc) fclose(fc);
    return 0;
}
