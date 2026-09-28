/* Stress / property tests (not covered by the per-shape suites):
 *
 *  D1  determinism across thread counts: every fp32 algorithm gives BITWISE identical
 *      output for 1, 2, 3, 4, 7 and 16 threads (the per-output summation order does not
 *      depend on the work split), and repeated executes are identical.
 *  D2  int8: identical s32 output for every thread count (exact arithmetic).
 *  C1  one plan executed concurrently from 4 pthreads (each with 2 OpenMP threads)
 *      gives the same bits as a serial execute: execute() is reentrant.
 */
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ugconv.h"

static int fails = 0, total = 0;
static void check(const char *id, int ok, const char *what)
{
    ++total;
    if (!ok) ++fails;
    printf("%s %s %s\n", ok ? "PASS" : "FAIL", id, what);
}

static const ug_conv_desc SHAPES[] = {
    {2, 17, 29, 45, 23, 3, 3, 1, 1, 1, 1},
    {1, 150, 14, 14, 200, 3, 3, 1, 1, 1, 1},   /* C > 128 chunks, Kp > 192 K-blocks */
    {1, 9, 57, 61, 13, 5, 5, 2, 2, 2, 2},
    {1, 300, 7, 7, 70, 1, 1, 1, 1, 0, 0},
};

typedef struct { const ug_conv_plan *p; const float *x; float *y; int rc; } job;
static void *run_job(void *a)
{
    job *j = a;
    for (int r = 0; r < 5 && j->rc == 0; ++r) j->rc = ug_conv_execute(j->p, j->x, j->y);
    return NULL;
}

int main(void)
{
    char msg[256];
    static const ug_algo A[] = {UG_ALGO_DIRECT, UG_ALGO_IM2COL_GEMM, UG_ALGO_WINOGRAD_F4, UG_ALGO_WINOGRAD_F2};
    static const int NT[] = {1, 2, 3, 4, 7, 16};
    for (unsigned si = 0; si < sizeof SHAPES / sizeof SHAPES[0]; ++si) {
        const ug_conv_desc *d = &SHAPES[si];
        const size_t nx = (size_t)d->N * d->C * d->H * d->W, nw = (size_t)d->K * d->C * d->R * d->S;
        const size_t ny = (size_t)d->N * d->K * ug_out_h(d) * ug_out_w(d);
        float *x = malloc(nx * 4), *w = malloc(nw * 4), *b = malloc(d->K * 4), *y0 = malloc(ny * 4), *y = malloc(ny * 4);
        srand(si + 7);
        for (size_t i = 0; i < nx; ++i) x[i] = rand() / (float)RAND_MAX - 0.5f;
        for (size_t i = 0; i < nw; ++i) w[i] = rand() / (float)RAND_MAX - 0.5f;
        for (int i = 0; i < d->K; ++i) b[i] = rand() / (float)RAND_MAX - 0.5f;

        for (unsigned a = 0; a < 4; ++a) {
            if (!ug_algo_eligible(d, A[a])) continue;
            ug_conv_plan *p1 = ug_conv_plan_create(d, w, b, 1, A[a], 1);
            ug_conv_execute(p1, x, y0);
            int same = 1;
            for (unsigned t = 1; t < sizeof NT / sizeof NT[0]; ++t) {
                ug_conv_plan *pt = ug_conv_plan_create(d, w, b, 1, A[a], NT[t]);
                for (int rep = 0; rep < 2; ++rep) {
                    memset(y, 0xFF, ny * 4);
                    ug_conv_execute(pt, x, y);
                    same &= !memcmp(y, y0, ny * 4);
                }
                ug_conv_plan_destroy(pt);
            }
            snprintf(msg, sizeof msg, "shape %u %-15s bitwise identical for 1/2/3/4/7/16 threads, repeated", si, ug_algo_name(A[a]));
            check("D1", same, msg);

            /* C1: 4 pthreads share one plan (2 OpenMP threads each) */
            ug_conv_plan *pc = ug_conv_plan_create(d, w, b, 1, A[a], 2);
            pthread_t th[4];
            job jobs[4];
            float *ys[4];
            for (int i = 0; i < 4; ++i) {
                ys[i] = malloc(ny * 4);
                jobs[i] = (job){pc, x, ys[i], 0};
                pthread_create(&th[i], NULL, run_job, &jobs[i]);
            }
            int ok = 1;
            for (int i = 0; i < 4; ++i) {
                pthread_join(th[i], NULL);
                ok &= jobs[i].rc == 0 && !memcmp(ys[i], y0, ny * 4);
                free(ys[i]);
            }
            snprintf(msg, sizeof msg, "shape %u %-15s concurrent execute from 4 threads matches serial", si, ug_algo_name(A[a]));
            check("C1", ok, msg);
            ug_conv_plan_destroy(pc);
            ug_conv_plan_destroy(p1);
        }

        /* D2: int8 */
        if ((long long)d->C * d->R * d->S <= UG_Q_MAX_CRS) {
            static const ug_qalgo QA[] = {UG_QALGO_DIRECT, UG_QALGO_IM2COL_GEMM, UG_QALGO_WINOGRAD_F2};
            unsigned char *xq = malloc(nx);
            signed char *wq = malloc(nw);
            float *sc = malloc(d->K * 4);
            int32_t *q0 = malloc(ny * 4), *q = malloc(ny * 4);
            for (size_t i = 0; i < nx; ++i) xq[i] = (unsigned char)rand();
            for (size_t i = 0; i < nw; ++i) wq[i] = (signed char)rand();
            for (int i = 0; i < d->K; ++i) sc[i] = 1.f;
            for (unsigned a = 0; a < 3; ++a) {
                if (!ug_qalgo_eligible(d, QA[a])) continue;
                ug_qconv_plan *p1 = ug_qconv_plan_create(d, wq, sc, NULL, 0, 1.f, 100, QA[a], 1);
                ug_qconv_execute_s32(p1, xq, q0);
                int same = 1;
                for (unsigned t = 1; t < sizeof NT / sizeof NT[0]; ++t) {
                    ug_qconv_plan *pt = ug_qconv_plan_create(d, wq, sc, NULL, 0, 1.f, 100, QA[a], NT[t]);
                    ug_qconv_execute_s32(pt, xq, q);
                    same &= !memcmp(q, q0, ny * 4);
                    ug_qconv_plan_destroy(pt);
                }
                snprintf(msg, sizeof msg, "shape %u %-15s identical for 1/2/3/4/7/16 threads", si, ug_qalgo_name(QA[a]));
                check("D2", same, msg);
                ug_qconv_plan_destroy(p1);
            }
            free(xq); free(wq); free(sc); free(q0); free(q);
        }
        free(x); free(w); free(b); free(y0); free(y);
    }
    printf("%d/%d stress checks passed\n", total - fails, total);
    return fails ? 1 : 0;
}
