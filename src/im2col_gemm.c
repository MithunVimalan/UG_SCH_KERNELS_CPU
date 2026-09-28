/* Group 2: im2col + GEMM.
 *
 * out[K][P*Q] = W[K][C*R*S] x col[C*R*S][P*Q]
 * The im2col matrix is never materialised in full: each KC x NCb block of it is
 * written straight into the GEMM's packed B panels (KC x 16), which is the only
 * place the GEMM reads it from. Memory for "col" drops from C*R*S*P*Q*4 bytes to
 * KC*NCb*4 per thread; the arithmetic is identical to explicit im2col + SGEMM.
 *
 * Loop order per work item (BLIS):  kc -> pack B -> ic (MC) -> jr (16) -> ir (6).
 * B micro-panel (kc*64 B) stays in L1 across the ir loop, the A block (MC x KC,
 * pre-packed weights) in L2.
 */
#include <immintrin.h>
#include <string.h>
#include "ug_internal.h"

#define GEMM_KC 256
#define GEMM_MC 240   /* multiple of 6; fits Skymont's 1 MB/core L2 share with B */
#define GEMM_NC 384   /* multiple of 16 */

int ug_gemm_prepare(ug_conv_plan *p, const float *w)
{
    const ug_conv_desc *d = &p->d;
    int CRS = d->C * d->R * d->S;
    p->wgemm = ug_malloc((size_t)p->Kp * CRS * sizeof(float));
    if (!p->wgemm) return -1;
    for (int k = 0; k < p->Kp; ++k)
        for (int q = 0; q < CRS; ++q)
            p->wgemm[((size_t)(k / UG_MR) * CRS + q) * UG_MR + k % UG_MR] =
                k < d->K ? w[(size_t)k * CRS + q] : 0.f;
    return 0;
}

/* Pack rows [k0, k0+kc) and pixels [n0, n0+ncols) of the implicit im2col matrix. */
static void pack_im2col(const ug_conv_desc *d, int P, int Q, const float *x,
                        int k0, int kc, int n0, int ncols, float *buf)
{
    const int H = d->H, W = d->W, R = d->R, S = d->S;
    const int sh = d->stride_h, sw = d->stride_w, ph = d->pad_h, pw = d->pad_w;
    const int PQ = P * Q;
    const int pointwise = (R == 1 && S == 1 && sh == 1 && sw == 1 && ph == 0 && pw == 0);
    const int npan = ug_ceil_div(ncols, UG_NR);

    for (int jp = 0; jp < npan; ++jp) {
        const int p0 = n0 + jp * UG_NR;
        const int nv = ug_min(UG_NR, ug_min(n0 + ncols, PQ) - p0);
        float *dst = buf + (size_t)jp * kc * UG_NR;
        int oh[UG_NR], ow[UG_NR];
        for (int j = 0; j < UG_NR; ++j) {
            int pp = p0 + (j < nv ? j : 0);
            oh[j] = pp / Q;
            ow[j] = pp % Q;
        }
        const int same_row = (nv == UG_NR) && (oh[0] == oh[UG_NR - 1]) && (sw == 1 || sw == 2);

        int c = k0 / (R * S), r = (k0 / S) % R, s = k0 % S;
        for (int kk = 0; kk < kc; ++kk, dst += UG_NR) {
            const float *xc = x + (size_t)c * H * W;
            if (pointwise) {
                if (nv == UG_NR) {
                    _mm256_store_ps(dst, _mm256_loadu_ps(xc + p0));
                    _mm256_store_ps(dst + 8, _mm256_loadu_ps(xc + p0 + 8));
                } else {
                    for (int j = 0; j < UG_NR; ++j) dst[j] = j < nv ? xc[p0 + j] : 0.f;
                }
            } else if (same_row) {
                const int ih = oh[0] * sh - ph + r;
                const int iw0 = ow[0] * sw - pw + s;
                if (ih < 0 || ih >= H) {
                    _mm256_store_ps(dst, _mm256_setzero_ps());
                    _mm256_store_ps(dst + 8, _mm256_setzero_ps());
                } else if (sw == 1 && iw0 >= 0 && iw0 + UG_NR <= W) {
                    const float *src = xc + (size_t)ih * W + iw0;
                    _mm256_store_ps(dst, _mm256_loadu_ps(src));
                    _mm256_store_ps(dst + 8, _mm256_loadu_ps(src + 8));
                } else if (sw == 2 && iw0 >= 0 && iw0 + 2 * UG_NR <= W) {
                    /* even elements of 32 contiguous floats, all inside the row */
                    const float *src = xc + (size_t)ih * W + iw0;
                    for (int h = 0; h < 2; ++h) {
                        __m256 a = _mm256_loadu_ps(src + 16 * h), b = _mm256_loadu_ps(src + 16 * h + 8);
                        __m256 e = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
                        e = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(e), 0xD8));
                        _mm256_store_ps(dst + 8 * h, e);
                    }
                } else {
                    const float *src = xc + (size_t)ih * W;
                    for (int j = 0; j < UG_NR; ++j) {
                        int iw = iw0 + j * sw;
                        dst[j] = (iw >= 0 && iw < W) ? src[iw] : 0.f;
                    }
                }
            } else {
                for (int j = 0; j < UG_NR; ++j) {
                    float v = 0.f;
                    if (j < nv) {
                        int ih = oh[j] * sh - ph + r, iw = ow[j] * sw - pw + s;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) v = xc[(size_t)ih * W + iw];
                    }
                    dst[j] = v;
                }
            }
            if (++s == S) { s = 0; if (++r == R) { r = 0; ++c; } }
        }
    }
}

void ug_gemm_blocking(const ug_conv_desc *d, int nt, ug_gemm_blk *b)
{
    const int PQ = ug_out_h(d) * ug_out_w(d), CRS = d->C * d->R * d->S;
    const int kpan = ug_ceil_div(d->K, UG_MR);
    /* KC: split CRS into equal chunks of <= GEMM_KC. */
    const int nkc = ug_ceil_div(CRS, GEMM_KC);
    b->KC = ug_ceil_div(CRS, nkc);
    /* Work decomposition: >= 3 items per thread for hybrid (P/E) load balance. */
    const int want = nt > 1 ? 3 * nt : 1; /* only split for parallelism */
    int NCb = ug_min(ug_round_up(PQ, UG_NR), GEMM_NC);
    if (d->N * ug_ceil_div(PQ, NCb) < want) {
        int per_img = ug_ceil_div(want, d->N);
        NCb = ug_max(48, ug_round_up(ug_ceil_div(PQ, per_img), UG_NR));
        NCb = ug_min(NCb, ug_round_up(PQ, UG_NR));
    }
    b->NCb = NCb;
    b->nblk = ug_ceil_div(PQ, NCb);
    int mblk = 1;
    if (d->N * b->nblk < want) mblk = ug_min(kpan, ug_ceil_div(want, d->N * b->nblk));
    b->pan_per_mblk = ug_ceil_div(kpan, mblk);
    b->mblk = ug_ceil_div(kpan, b->pan_per_mblk);
}

int ug_gemm_execute(const ug_conv_plan *p, const float *in, float *out)
{
    const ug_conv_desc *d = &p->d;
    const int N = d->N, C = d->C, K = d->K;
    const int P = p->P, Q = p->Q, PQ = P * Q;
    const int CRS = C * d->R * d->S;
    const int nt = p->nthreads;
    const int kpan = ug_ceil_div(K, UG_MR);

    ug_gemm_blk bk;
    ug_gemm_blocking(d, nt, &bk);
    const int KC = bk.KC, NCb = bk.NCb, nblk = bk.nblk, mblk = bk.mblk, pan_per_mblk = bk.pan_per_mblk;
    const int items = N * nblk * mblk;

#pragma omp parallel num_threads(nt)
    {
        float *buf = ug_malloc((size_t)KC * NCb * sizeof(float));
#pragma omp for schedule(dynamic, 1)
        for (int it = 0; it < items; ++it) {
            const int n = it / (nblk * mblk);
            const int rem = it % (nblk * mblk);
            const int nb = rem / mblk, mb = rem % mblk;
            const int n0 = nb * NCb, ncols = ug_min(NCb, PQ - n0);
            const int ip0 = mb * pan_per_mblk, ip1 = ug_min(kpan, ip0 + pan_per_mblk);
            const float *x = in + (size_t)n * C * d->H * d->W;
            float *y = out + (size_t)n * K * PQ;
            const int npan = ug_ceil_div(ncols, UG_NR);

            for (int k0 = 0; k0 < CRS; k0 += KC) {
                const int kc = ug_min(KC, CRS - k0);
                const int last = (k0 + kc == CRS);
                pack_im2col(d, P, Q, x, k0, kc, n0, ncols, buf);
                for (int ic = ip0; ic < ip1; ic += GEMM_MC / UG_MR) {
                    const int ic1 = ug_min(ip1, ic + GEMM_MC / UG_MR);
                    for (int jp = 0; jp < npan; ++jp) {
                        const int nr = ug_min(UG_NR, ncols - jp * UG_NR);
                        const float *B = buf + (size_t)jp * kc * UG_NR;
                        for (int ip = ic; ip < ic1; ++ip) {
                            const int mr = ug_min(UG_MR, K - ip * UG_MR);
                            ug_kernel_6x16(kc, p->wgemm + ((size_t)ip * CRS + k0) * UG_MR, B,
                                           y + (size_t)ip * UG_MR * PQ + n0 + jp * UG_NR, PQ,
                                           mr, nr, k0 > 0, last ? p->bias + ip * UG_MR : NULL, p->relu);
                        }
                    }
                }
            }
        }
        ug_free(buf);
    }
    return 0;
}
