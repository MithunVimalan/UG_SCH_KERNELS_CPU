/* Group 1: direct convolution.
 *
 * Register tile: 6 output channels x 16 output columns (2 ymm), 12 accumulators.
 * Vector loop = output width (unit stride in NKPQ output, so stores are contiguous);
 * weights are the broadcast operand. Per (c,r,s): 2 input loads + 6 broadcasts, 12 FMAs.
 *
 * Strided convs would make the input vector load strided (a gather). Instead the
 * padded input copy is phase-split by column (checklist D3 rule V3):
 *   X[c][ih][phase][j] = in[c][ih-pad_h][j*stride_w + phase - pad_w]
 * so input column ow*stride_w + s lives at X[..][s % stride_w][ow + s / stride_w],
 * contiguous in ow for every stride. No lowering buffer, no gather.
 */
#include <immintrin.h>
#include <string.h>
#include "ug_internal.h"

int ug_direct_prepare(ug_conv_plan *p, const float *w)
{
    const ug_conv_desc *d = &p->d;
    const int C = d->C, RS = d->R * d->S;
    p->wdir = ug_malloc((size_t)p->Kp * C * RS * sizeof(float));
    if (!p->wdir) return -1;
    for (int k = 0; k < p->Kp; ++k)
        for (int c = 0; c < C; ++c)
            for (int q = 0; q < RS; ++q)
                p->wdir[(((size_t)(k / UG_MR) * C + c) * RS + q) * UG_MR + k % UG_MR] =
                    k < d->K ? w[((size_t)k * C + c) * RS + q] : 0.f;
    return 0;
}

/* off[q] = input offset of tap q = (c,r,s), precomputed so the hot loop is a single
 * flat loop over C*R*S with no index arithmetic (no div/mod by the stride). */
static inline void direct_tile(const float *X, const float *wp, const long *off, int CRS,
                               __m256 acc[12])
{
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();
#pragma GCC unroll 4
    for (int q = 0; q < CRS; ++q) {
        const float *src = X + off[q];
        __m256 b0 = _mm256_loadu_ps(src);
        __m256 b1 = _mm256_loadu_ps(src + 8);
        __m256 a;
        a = _mm256_broadcast_ss(wp + 0); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(wp + 1); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(wp + 2); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(wp + 3); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(wp + 4); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(wp + 5); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
        wp += UG_MR;
    }
    acc[0] = c00; acc[1] = c01; acc[2] = c10; acc[3] = c11; acc[4] = c20; acc[5] = c21;
    acc[6] = c30; acc[7] = c31; acc[8] = c40; acc[9] = c41; acc[10] = c50; acc[11] = c51;
}

int ug_direct_execute(const ug_conv_plan *p, const float *in, float *out)
{
    const ug_conv_desc *d = &p->d;
    const int N = d->N, C = d->C, H = d->H, W = d->W, K = d->K, R = d->R, S = d->S;
    const int sh = d->stride_h, sw = d->stride_w, ph = d->pad_h, pw = d->pad_w;
    const int P = p->P, Q = p->Q, PQ = P * Q;
    const int nt = p->nthreads;
    const int kpan = ug_ceil_div(K, UG_MR);

    const int Hp = H + 2 * ph;
    const int Q16 = ug_round_up(Q, 16);
    const int Wq = ug_round_up(Q16 + (S - 1) / sw + 1, 8);
    const long rowstride = (long)sw * Wq;           /* one input row, all phases */
    const long planestride = (long)Hp * rowstride;  /* one channel */
    float *X = ug_malloc((size_t)C * planestride * sizeof(float));
    long *off = ug_malloc((size_t)C * R * S * sizeof(long));
    if (!X || !off) { ug_free(X); ug_free(off); return -1; }
    for (int c = 0, q = 0; c < C; ++c)
        for (int r = 0; r < R; ++r)
            for (int s = 0; s < S; ++s, ++q)
                off[q] = c * planestride + r * rowstride + (long)(s % sw) * Wq + s / sw;

    /* Rows of output per item: keep the input band (C x (rows*sh+R) rows) <= ~256 KB. */
    const long band_row_bytes = (long)C * rowstride * 4;
    int rows = (int)ug_max(1, (int)((256L * 1024 / band_row_bytes - R) / sh + 1));
    rows = ug_min(rows, P);
    const int want = nt > 1 ? 3 * nt : 1; /* only split for parallelism */
    while (rows > 1 && (long)ug_ceil_div(P, rows) * kpan < want) rows = (rows + 1) / 2;
    const int rblk = ug_ceil_div(P, rows);
    int kgrp = 1;
    if (rblk < want) kgrp = ug_min(kpan, ug_ceil_div(want, rblk));
    const int pan_per_grp = ug_ceil_div(kpan, kgrp);
    kgrp = ug_ceil_div(kpan, pan_per_grp);
    const int items = rblk * kgrp;

    for (int n = 0; n < N; ++n) {
        const float *x = in + (size_t)n * C * H * W;
        float *y = out + (size_t)n * K * PQ;
#pragma omp parallel num_threads(nt)
        {
            /* Build the padded, column-phase-split copy (zeros in the halo). */
#pragma omp for schedule(static)
            for (int cr = 0; cr < C * Hp; ++cr) {
                const int c = cr / Hp, ihp = cr % Hp, ih = ihp - ph;
                float *dst = X + c * planestride + (long)ihp * rowstride;
                if (ih < 0 || ih >= H) { memset(dst, 0, rowstride * sizeof(float)); continue; }
                const float *src = x + ((size_t)c * H + ih) * W;
                if (sw == 1) {
                    const int ncp = ug_min(W, Wq - pw);
                    memset(dst, 0, pw * sizeof(float));
                    memcpy(dst + pw, src, ncp * sizeof(float));
                    memset(dst + pw + ncp, 0, (Wq - pw - ncp) * sizeof(float));
                    continue;
                }
                for (int f = 0; f < sw; ++f) {
                    float *df = dst + (long)f * Wq;
                    for (int j = 0; j < Wq; ++j) {
                        int iw = j * sw + f - pw;
                        df[j] = (iw >= 0 && iw < W) ? src[iw] : 0.f;
                    }
                }
            }

#pragma omp for schedule(dynamic, 1)
            for (int it = 0; it < items; ++it) {
                const int rb = it / kgrp, kg = it % kgrp;
                const int oh0 = rb * rows, oh1 = ug_min(P, oh0 + rows);
                const int ip0 = kg * pan_per_grp, ip1 = ug_min(kpan, ip0 + pan_per_grp);
                for (int ip = ip0; ip < ip1; ++ip) {
                    const float *wp = p->wdir + (size_t)ip * C * R * S * UG_MR;
                    const int mr = ug_min(UG_MR, K - ip * UG_MR);
                    const float *bias = p->bias + ip * UG_MR;
                    for (int oh = oh0; oh < oh1; ++oh) {
                        const float *xrow = X + (long)oh * sh * rowstride;
                        for (int ow0 = 0; ow0 < Q; ow0 += 16) {
                            __m256 acc[12];
                            direct_tile(xrow + ow0, wp, off, C * R * S, acc);
                            const int nv = ug_min(16, Q - ow0);
                            const __m256 zero = _mm256_setzero_ps();
                            for (int i = 0; i < mr; ++i) {
                                __m256 b = _mm256_broadcast_ss(bias + i);
                                __m256 v0 = _mm256_add_ps(acc[2 * i], b);
                                __m256 v1 = _mm256_add_ps(acc[2 * i + 1], b);
                                if (p->relu) { v0 = _mm256_max_ps(zero, v0); v1 = _mm256_max_ps(zero, v1); } /* keeps NaN */
                                float *dst = y + (size_t)(ip * UG_MR + i) * PQ + (size_t)oh * Q + ow0;
                                if (nv == 16) {
                                    _mm256_storeu_ps(dst, v0);
                                    _mm256_storeu_ps(dst + 8, v1);
                                } else {
                                    float tmp[16] __attribute__((aligned(32)));
                                    _mm256_store_ps(tmp, v0);
                                    _mm256_store_ps(tmp + 8, v1);
                                    for (int j = 0; j < nv; ++j) dst[j] = tmp[j];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    ug_free(X);
    ug_free(off);
    return 0;
}
