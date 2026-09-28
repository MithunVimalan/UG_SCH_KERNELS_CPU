/* INT8 group 2: im2col + GEMM with VNNI.
 *
 * out[K][PQ] = W[K][CRS] x col[CRS][PQ], reduction padded to CRS4 = round_up(CRS, 4).
 * vpdpbusd multiplies 4 consecutive reduction elements per 32-bit lane, so:
 *   B (activations, u8, the unsigned operand) panel: [CRS4/4][16 pixels][4 bytes]
 *     -> one k4 step = 64 contiguous bytes = 2 ymm (8 pixels each)
 *   A (weights, s8) panel: [CRS4/4][6 rows][4 bytes] -> one k4 step = 6 dword broadcasts
 * Register tile 6x16: 12 int32 accumulators, 12 vpdpbusd per k4 step (= 384 MACs).
 * No KC split: int32 accumulators live in registers over the whole reduction, so
 * the epilogue runs once; the per-item B block is sized to stay in L2 instead.
 */
#include <string.h>
#include "q_internal.h"

#define QG_B_BUDGET (384 * 1024) /* bytes of packed B per work item */
#define QG_A_BUDGET (384 * 1024) /* bytes of weight block per ic step */

int ug_qgemm_prepare(ug_qconv_plan *p, const int8_t *w)
{
    const ug_conv_desc *d = &p->d;
    const int CRS = d->C * d->R * d->S;
    p->CRS4 = ug_round_up(CRS, 4);
    const int k4n = p->CRS4 / 4;
    p->wg = ug_malloc((size_t)p->Kp * p->CRS4);
    if (!p->wg) return -1;
    for (int k = 0; k < p->Kp; ++k)
        for (int q = 0; q < p->CRS4; ++q)
            p->wg[(((size_t)(k / UG_MR) * k4n + q / 4) * UG_MR + k % UG_MR) * 4 + q % 4] =
                (k < d->K && q < CRS) ? w[(size_t)k * CRS + q] : 0;
    return 0;
}

/* acc[12] = A(k4n x 6 x 4) . B(k4n x 16 x 4) */
static inline void qkernel(int k4n, const int8_t *A, const uint8_t *B, int half, __m256i acc[12])
{
    __m256i c00 = _mm256_setzero_si256(), c01 = _mm256_setzero_si256();
    __m256i c10 = _mm256_setzero_si256(), c11 = _mm256_setzero_si256();
    __m256i c20 = _mm256_setzero_si256(), c21 = _mm256_setzero_si256();
    __m256i c30 = _mm256_setzero_si256(), c31 = _mm256_setzero_si256();
    __m256i c40 = _mm256_setzero_si256(), c41 = _mm256_setzero_si256();
    __m256i c50 = _mm256_setzero_si256(), c51 = _mm256_setzero_si256();
    const int32_t *a32 = (const int32_t *)A;
    if (!half) {
#pragma GCC unroll 4
        for (int s = 0; s < k4n; ++s) {
            __m256i b0 = _mm256_load_si256((const __m256i *)B);
            __m256i b1 = _mm256_load_si256((const __m256i *)(B + 32));
            __m256i a;
            a = _mm256_set1_epi32(a32[0]); c00 = ug_dpbusd(c00, b0, a); c01 = ug_dpbusd(c01, b1, a);
            a = _mm256_set1_epi32(a32[1]); c10 = ug_dpbusd(c10, b0, a); c11 = ug_dpbusd(c11, b1, a);
            a = _mm256_set1_epi32(a32[2]); c20 = ug_dpbusd(c20, b0, a); c21 = ug_dpbusd(c21, b1, a);
            a = _mm256_set1_epi32(a32[3]); c30 = ug_dpbusd(c30, b0, a); c31 = ug_dpbusd(c31, b1, a);
            a = _mm256_set1_epi32(a32[4]); c40 = ug_dpbusd(c40, b0, a); c41 = ug_dpbusd(c41, b1, a);
            a = _mm256_set1_epi32(a32[5]); c50 = ug_dpbusd(c50, b0, a); c51 = ug_dpbusd(c51, b1, a);
            a32 += UG_MR;
            B += 64;
        }
    } else { /* <= 8 live pixels: skip the second vector */
#pragma GCC unroll 4
        for (int s = 0; s < k4n; ++s) {
            __m256i b0 = _mm256_load_si256((const __m256i *)B);
            c00 = ug_dpbusd(c00, b0, _mm256_set1_epi32(a32[0]));
            c10 = ug_dpbusd(c10, b0, _mm256_set1_epi32(a32[1]));
            c20 = ug_dpbusd(c20, b0, _mm256_set1_epi32(a32[2]));
            c30 = ug_dpbusd(c30, b0, _mm256_set1_epi32(a32[3]));
            c40 = ug_dpbusd(c40, b0, _mm256_set1_epi32(a32[4]));
            c50 = ug_dpbusd(c50, b0, _mm256_set1_epi32(a32[5]));
            a32 += UG_MR;
            B += 64;
        }
    }
    acc[0] = c00; acc[1] = c01; acc[2] = c10; acc[3] = c11; acc[4] = c20; acc[5] = c21;
    acc[6] = c30; acc[7] = c31; acc[8] = c40; acc[9] = c41; acc[10] = c50; acc[11] = c51;
}

/* 16 bytes of im2col row q = (c,r,s) for pixels p0..p0+15. */
static inline __m128i im2col_row16(const ug_conv_desc *d, const uint8_t *x, int q, int CRS, int zp,
                                   const int *oh, const int *ow, int nv, int p0, int same_row, int pointwise)
{
    const int H = d->H, W = d->W;
    if (q >= CRS) return _mm_set1_epi8((char)zp); /* weight is 0 there: any value works */
    const int c = q / (d->R * d->S), r = (q / d->S) % d->R, s = q % d->S;
    const uint8_t *xc = x + (size_t)c * H * W;
    if (pointwise && nv == 16) return _mm_loadu_si128((const __m128i *)(xc + p0));
    if (same_row) {
        const int ih = oh[0] * d->stride_h - d->pad_h + r;
        const int iw0 = ow[0] * d->stride_w - d->pad_w + s;
        if (ih < 0 || ih >= H) return _mm_set1_epi8((char)zp);
        const uint8_t *row = xc + (size_t)ih * W;
        if (d->stride_w == 1 && iw0 >= 0 && iw0 + 16 <= W) return _mm_loadu_si128((const __m128i *)(row + iw0));
        if (d->stride_w == 2 && iw0 >= 0 && iw0 + 32 <= W) {
            const __m128i ev = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
            __m128i lo = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(row + iw0)), ev);
            __m128i hi = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(row + iw0 + 16)), ev);
            return _mm_unpacklo_epi64(lo, hi);
        }
    }
    uint8_t t[16];
    for (int j = 0; j < 16; ++j) {
        uint8_t v = (uint8_t)zp;
        if (j < nv) {
            int ih = oh[j] * d->stride_h - d->pad_h + r, iw = ow[j] * d->stride_w - d->pad_w + s;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) v = xc[(size_t)ih * W + iw];
        }
        t[j] = v;
    }
    return _mm_loadu_si128((const __m128i *)t);
}

/* Pack pixels [n0, n0+ncols) x all CRS4 rows into panels [jp][k4][16][4]. */
static void qpack_im2col(const ug_qconv_plan *p, const uint8_t *x, int n0, int ncols, uint8_t *buf)
{
    const ug_conv_desc *d = &p->d;
    const int Q = p->Q, PQ = p->P * p->Q, CRS = d->C * d->R * d->S, k4n = p->CRS4 / 4;
    const int pointwise = d->R == 1 && d->S == 1 && d->stride_h == 1 && d->stride_w == 1 &&
                          d->pad_h == 0 && d->pad_w == 0;
    const int npan = ug_ceil_div(ncols, 16);
    for (int jp = 0; jp < npan; ++jp) {
        const int p0 = n0 + jp * 16;
        const int nv = ug_min(16, ug_min(n0 + ncols, PQ) - p0);
        int oh[16], ow[16];
        for (int j = 0; j < 16; ++j) {
            int pp = p0 + (j < nv ? j : 0);
            oh[j] = pp / Q;
            ow[j] = pp % Q;
        }
        const int same_row = nv == 16 && oh[0] == oh[15];
        uint8_t *dst = buf + (size_t)jp * k4n * 64;
        for (int k4 = 0; k4 < k4n; ++k4, dst += 64) {
            __m128i r0 = im2col_row16(d, x, 4 * k4 + 0, CRS, p->in_zp, oh, ow, nv, p0, same_row, pointwise);
            __m128i r1 = im2col_row16(d, x, 4 * k4 + 1, CRS, p->in_zp, oh, ow, nv, p0, same_row, pointwise);
            __m128i r2 = im2col_row16(d, x, 4 * k4 + 2, CRS, p->in_zp, oh, ow, nv, p0, same_row, pointwise);
            __m128i r3 = im2col_row16(d, x, 4 * k4 + 3, CRS, p->in_zp, oh, ow, nv, p0, same_row, pointwise);
            /* transpose 4 rows x 16 pixels -> 16 pixels x 4 bytes */
            __m128i a = _mm_unpacklo_epi8(r0, r1), b = _mm_unpackhi_epi8(r0, r1);
            __m128i c = _mm_unpacklo_epi8(r2, r3), e = _mm_unpackhi_epi8(r2, r3);
            _mm_store_si128((__m128i *)(dst + 0), _mm_unpacklo_epi16(a, c));
            _mm_store_si128((__m128i *)(dst + 16), _mm_unpackhi_epi16(a, c));
            _mm_store_si128((__m128i *)(dst + 32), _mm_unpacklo_epi16(b, e));
            _mm_store_si128((__m128i *)(dst + 48), _mm_unpackhi_epi16(b, e));
        }
    }
}

int ug_qgemm_execute(const ug_qconv_plan *p, const uint8_t *in, const ug_qout *o, long img_elems)
{
    const ug_conv_desc *d = &p->d;
    const int N = d->N, K = d->K, PQ = p->P * p->Q, nt = p->nthreads;
    const int k4n = p->CRS4 / 4, kpan = p->Kp / UG_MR;

    /* NCb: packed B block (CRS4 x NCb bytes) within budget; MC: weight block within budget. */
    int NCb = ug_max(16, ug_min(512, QG_B_BUDGET / p->CRS4 / 16 * 16));
    NCb = ug_min(NCb, ug_round_up(PQ, 16));
    const int want = nt > 1 ? 3 * nt : 1;
    if (N * ug_ceil_div(PQ, NCb) < want) {
        int per_img = ug_ceil_div(want, N);
        NCb = ug_min(NCb, ug_max(32, ug_round_up(ug_ceil_div(PQ, per_img), 16)));
    }
    const int nblk = ug_ceil_div(PQ, NCb);
    int mblk = 1;
    if (N * nblk < want) mblk = ug_min(kpan, ug_ceil_div(want, N * nblk));
    const int ppm = ug_ceil_div(kpan, mblk);
    mblk = ug_ceil_div(kpan, ppm);
    const int mc_pan = ug_max(1, QG_A_BUDGET / (p->CRS4 * UG_MR));
    const int items = N * nblk * mblk;

#pragma omp parallel num_threads(nt)
    {
        uint8_t *buf = ug_malloc((size_t)p->CRS4 * NCb);
#pragma omp for schedule(dynamic, 1)
        for (int it = 0; it < items; ++it) {
            const int n = it / (nblk * mblk), rem = it % (nblk * mblk);
            const int nb = rem / mblk, mb = rem % mblk;
            const int n0 = nb * NCb, ncols = ug_min(NCb, PQ - n0);
            const int ip0 = mb * ppm, ip1 = ug_min(kpan, ip0 + ppm);
            const uint8_t *x = in + (size_t)n * d->C * d->H * d->W;
            const ug_qout oi = ug_qout_image(o, n, img_elems);
            const int npan = ug_ceil_div(ncols, 16);
            qpack_im2col(p, x, n0, ncols, buf);
            for (int ic = ip0; ic < ip1; ic += mc_pan) {
                const int ic1 = ug_min(ip1, ic + mc_pan);
                for (int jp = 0; jp < npan; ++jp) {
                    const int nr = ug_min(16, ncols - jp * 16);
                    const uint8_t *B = buf + (size_t)jp * k4n * 64;
                    for (int ip = ic; ip < ic1; ++ip) {
                        __m256i acc[12];
                        qkernel(k4n, p->wg + (size_t)ip * k4n * UG_MR * 4, B, nr <= 8, acc);
                        const int mr = ug_min(UG_MR, K - ip * UG_MR);
                        for (int i = 0; i < mr; ++i) {
                            const int k = ip * UG_MR + i;
                            ug_q_epilogue16(p, &oi, k, (long)k * PQ + n0 + jp * 16, acc[2 * i], acc[2 * i + 1], nr, 0);
                        }
                    }
                }
            }
        }
        ug_free(buf);
    }
    return 0;
}
