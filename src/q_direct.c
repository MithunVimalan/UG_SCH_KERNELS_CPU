/* INT8 group 1: direct convolution with VNNI.
 *
 * vpdpbusd reduces 4 bytes per 32-bit lane; lanes = output pixels along the row.
 * The 4 bytes are 4 input channels of the same pixel, so the input is repacked once
 * per image into  X[C4/4][Hp][stride_w phases][Wq][4]  (u8, halo = in_zp):
 *   X[c4][ih][f][j][b] = x[4*c4 + b][ih - pad_h][j*stride_w + f - pad_w]
 * Tap (c4, r, s) for 16 consecutive output columns is then 64 contiguous bytes
 * (column phase split, as in the fp32 direct kernel, removes the stride gather).
 * Weights: [Kp/6][C4/4][R][S][6][4] s8, one dword broadcast per output channel.
 * Tile: 6 output channels x 16 output columns, 12 accumulators, one flat loop over
 * the (C4/4)*R*S taps through a precomputed offset table.
 */
#include <string.h>
#include "q_internal.h"

int ug_qdirect_prepare(ug_qconv_plan *p, const int8_t *w)
{
    const ug_conv_desc *d = &p->d;
    const int C = d->C, R = d->R, S = d->S;
    p->C4 = ug_round_up(C, 4);
    const int c4n = p->C4 / 4;
    p->wd = ug_malloc((size_t)p->Kp * p->C4 * R * S);
    if (!p->wd) return -1;
    for (int k = 0; k < p->Kp; ++k)
        for (int c = 0; c < p->C4; ++c)
            for (int r = 0; r < R; ++r)
                for (int s = 0; s < S; ++s)
                    p->wd[((((size_t)(k / UG_MR) * c4n + c / 4) * R + r) * S + s) * UG_MR * 4 + (k % UG_MR) * 4 + c % 4] =
                        (k < d->K && c < C) ? w[(((size_t)k * C + c) * R + r) * S + s] : 0;
    return 0;
}

static inline void qdirect_tile(const uint8_t *X, const int8_t *wp, const long *off, int ntaps, __m256i acc[12])
{
    __m256i c00 = _mm256_setzero_si256(), c01 = _mm256_setzero_si256();
    __m256i c10 = _mm256_setzero_si256(), c11 = _mm256_setzero_si256();
    __m256i c20 = _mm256_setzero_si256(), c21 = _mm256_setzero_si256();
    __m256i c30 = _mm256_setzero_si256(), c31 = _mm256_setzero_si256();
    __m256i c40 = _mm256_setzero_si256(), c41 = _mm256_setzero_si256();
    __m256i c50 = _mm256_setzero_si256(), c51 = _mm256_setzero_si256();
    const ug_i32_alias *a32 = (const ug_i32_alias *)wp;
#pragma GCC unroll 4
    for (int q = 0; q < ntaps; ++q) {
        const uint8_t *src = X + off[q];
        __m256i b0 = _mm256_loadu_si256((const __m256i *)src);
        __m256i b1 = _mm256_loadu_si256((const __m256i *)(src + 32));
        __m256i a;
        a = _mm256_set1_epi32(a32[0]); c00 = ug_dpbusd(c00, b0, a); c01 = ug_dpbusd(c01, b1, a);
        a = _mm256_set1_epi32(a32[1]); c10 = ug_dpbusd(c10, b0, a); c11 = ug_dpbusd(c11, b1, a);
        a = _mm256_set1_epi32(a32[2]); c20 = ug_dpbusd(c20, b0, a); c21 = ug_dpbusd(c21, b1, a);
        a = _mm256_set1_epi32(a32[3]); c30 = ug_dpbusd(c30, b0, a); c31 = ug_dpbusd(c31, b1, a);
        a = _mm256_set1_epi32(a32[4]); c40 = ug_dpbusd(c40, b0, a); c41 = ug_dpbusd(c41, b1, a);
        a = _mm256_set1_epi32(a32[5]); c50 = ug_dpbusd(c50, b0, a); c51 = ug_dpbusd(c51, b1, a);
        a32 += UG_MR;
    }
    acc[0] = c00; acc[1] = c01; acc[2] = c10; acc[3] = c11; acc[4] = c20; acc[5] = c21;
    acc[6] = c30; acc[7] = c31; acc[8] = c40; acc[9] = c41; acc[10] = c50; acc[11] = c51;
}

int ug_qdirect_execute(const ug_qconv_plan *p, const uint8_t *in, const ug_qout *o, long img_elems)
{
    const ug_conv_desc *d = &p->d;
    const int N = d->N, C = d->C, H = d->H, W = d->W, K = d->K, R = d->R, S = d->S;
    const int sh = d->stride_h, sw = d->stride_w, ph = d->pad_h, pw = d->pad_w;
    const int P = p->P, Q = p->Q, PQ = P * Q, nt = p->nthreads;
    const int kpan = p->Kp / UG_MR, c4n = p->C4 / 4, ntaps = c4n * R * S;
    const uint8_t zp = (uint8_t)p->in_zp;

    const int Hp = H + 2 * ph;
    const int Q16 = ug_round_up(Q, 16);
    const int Wq = ug_round_up(Q16 + (S - 1) / sw + 1, 8);   /* pixels per phase row */
    const long rowstride = (long)sw * Wq * 4;                 /* bytes: one input row, all phases */
    const long planestride = (long)Hp * rowstride;            /* bytes: one group of 4 channels */
    uint8_t *X = ug_malloc((size_t)c4n * planestride);
    long *off = ug_malloc((size_t)ntaps * sizeof(long));
    if (!X || !off) { ug_free(X); ug_free(off); return -1; }
    for (int c4 = 0, q = 0; c4 < c4n; ++c4)
        for (int r = 0; r < R; ++r)
            for (int s = 0; s < S; ++s, ++q)
                off[q] = c4 * planestride + r * rowstride + (long)(s % sw) * Wq * 4 + (long)(s / sw) * 4;

    /* rows per item: input band (c4n x (rows*sh+R) rows) <= ~256 KB */
    int rows = ug_max(1, (int)((256L * 1024 / (c4n * rowstride) - R) / sh + 1));
    rows = ug_min(rows, P);
    const int want = nt > 1 ? 3 * nt : 1;
    while (rows > 1 && (long)ug_ceil_div(P, rows) * kpan < want) rows = (rows + 1) / 2;
    const int rblk = ug_ceil_div(P, rows);
    int kgrp = 1;
    if (rblk < want) kgrp = ug_min(kpan, ug_ceil_div(want, rblk));
    const int ppg = ug_ceil_div(kpan, kgrp);
    kgrp = ug_ceil_div(kpan, ppg);
    const int items = rblk * kgrp;

    for (int n = 0; n < N; ++n) {
        const uint8_t *x = in + (size_t)n * C * H * W;
        const ug_qout oi = ug_qout_image(o, n, img_elems);
#pragma omp parallel num_threads(nt)
        {
#pragma omp for schedule(static)
            for (int cr = 0; cr < c4n * Hp; ++cr) {
                const int c4 = cr / Hp, ihp = cr % Hp, ih = ihp - ph;
                uint8_t *dst = X + c4 * planestride + (long)ihp * rowstride;
                if (ih < 0 || ih >= H) { memset(dst, zp, rowstride); continue; }
                for (int f = 0; f < sw; ++f) {
                    uint8_t *df = dst + (long)f * Wq * 4;
                    for (int j = 0; j < Wq; ++j) {
                        const int iw = j * sw + f - pw;
                        for (int b = 0; b < 4; ++b) {
                            const int c = 4 * c4 + b;
                            df[j * 4 + b] = (iw >= 0 && iw < W && c < C) ? x[((size_t)c * H + ih) * W + iw] : zp;
                        }
                    }
                }
            }
#pragma omp for schedule(dynamic, 1)
            for (int it = 0; it < items; ++it) {
                const int rb = it / kgrp, kg = it % kgrp;
                const int oh0 = rb * rows, oh1 = ug_min(P, oh0 + rows);
                const int ip0 = kg * ppg, ip1 = ug_min(kpan, ip0 + ppg);
                for (int ip = ip0; ip < ip1; ++ip) {
                    const int8_t *wp = p->wd + (size_t)ip * ntaps * UG_MR * 4;
                    const int mr = ug_min(UG_MR, K - ip * UG_MR);
                    for (int oh = oh0; oh < oh1; ++oh) {
                        const uint8_t *xrow = X + (long)oh * sh * rowstride;
                        for (int ow0 = 0; ow0 < Q; ow0 += 16) {
                            __m256i acc[12];
                            qdirect_tile(xrow + (long)ow0 * 4, wp, off, ntaps, acc);
                            const int nv = ug_min(16, Q - ow0);
                            for (int i = 0; i < mr; ++i) {
                                const int k = ip * UG_MR + i;
                                ug_q_epilogue16(p, &oi, k, (long)k * PQ + (long)oh * Q + ow0,
                                                acc[2 * i], acc[2 * i + 1], nv, 0);
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
