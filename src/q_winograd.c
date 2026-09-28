/* INT8 group 3: exact integer Winograd F(2x2,3x3) on int16 VNNI (vpdpwssd).
 *
 * Integer form (all arithmetic exact):
 *   d  = x_u8 - in_zp                     |d|  <= 255
 *   V  = B^T d B,   B^T = [1 0 -1 0; 0 1 1 0; 0 -1 1 0; 0 1 0 -1]
 *                                         |V|  <= 4*255 = 1020        -> int16
 *   U' = G' g G'^T, G' = 2G = [2 0 0; 1 1 1; 1 -1 1; 0 0 2]  (so U' = 4U)
 *                                         |U'| <= 3*3*128 = 1152      -> int16
 *   M  = sum_c U' (.) V                   int32, vpdpwssd (2 channels per lane)
 *   4Y = A^T M A,   A^T = [1 1 1 0; 0 1 -1 -1]
 *   Y  = (A^T M A) >> 2  == sum (x - zp) * w   exactly, the same integer as direct.
 * Intermediates may wrap mod 2^32; the final 4Y is exact as long as
 * |4Y| <= 4*C*9*255*128 < 2^31, i.e. C <= UG_QWINO_MAX_C (1800).
 *
 * Cost: 16 products per 2x2 outputs (4/output vs 9 for direct), but int16 VNNI does
 * 16 MAC/instruction vs 32 for int8 -> at best 9/4/2 = 1.125x the direct MAC rate,
 * before transforms. Included because it is exact; the selector decides.
 */
#include <string.h>
#include "q_internal.h"

#define QW_BUDGET (768 * 1024)

int ug_qwino_prepare(ug_qconv_plan *p, const int8_t *w)
{
    static const int Gp[4][3] = {{2, 0, 0}, {1, 1, 1}, {1, -1, 1}, {0, 0, 2}};
    const ug_conv_desc *d = &p->d;
    const int C = d->C, K = d->K, kpan = p->Kp / UG_MR;
    p->C2 = ug_round_up(C, 2);
    const int c2n = p->C2 / 2;
    const size_t n = (size_t)16 * p->Kp * p->C2;
    p->ww = ug_malloc(n * sizeof(int16_t));
    if (!p->ww) return -1;
    memset(p->ww, 0, n * sizeof(int16_t));
    for (int k = 0; k < K; ++k)
        for (int c = 0; c < C; ++c) {
            const int8_t *g = w + ((size_t)k * C + c) * 9;
            int t[4][3];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 3; ++j)
                    t[i][j] = Gp[i][0] * g[j] + Gp[i][1] * g[3 + j] + Gp[i][2] * g[6 + j];
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) {
                    int u = t[i][0] * Gp[j][0] + t[i][1] * Gp[j][1] + t[i][2] * Gp[j][2];
                    const int xi = i * 4 + j;
                    p->ww[((((size_t)xi * kpan + k / UG_MR) * c2n + c / 2) * UG_MR + k % UG_MR) * 2 + c % 2] = (int16_t)u;
                }
        }
    return 0;
}

/* 1-D B^T on 4 int16 vectors */
static inline void bt2_16(const __m256i d[4], __m256i r[4])
{
    r[0] = _mm256_sub_epi16(d[0], d[2]);
    r[1] = _mm256_add_epi16(d[1], d[2]);
    r[2] = _mm256_sub_epi16(d[2], d[1]);
    r[3] = _mm256_sub_epi16(d[1], d[3]);
}

/* 1-D A^T on 4 int32 vectors */
static inline void at2_32(const __m256i m[4], __m256i y[2])
{
    y[0] = _mm256_add_epi32(_mm256_add_epi32(m[0], m[1]), m[2]);
    y[1] = _mm256_sub_epi32(_mm256_sub_epi32(m[1], m[2]), m[3]);
}

/* 32 int16 of a row -> even-index and odd-index 16-vectors (natural tile order) */
static inline void deint16(const int16_t *src, __m256i *ev, __m256i *od)
{
    const __m256i sh = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
                                        0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
    __m256i a = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i *)src), sh);
    __m256i b = _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i *)(src + 16)), sh);
    a = _mm256_permute4x64_epi64(a, 0xD8); /* [ev 0-7 | od 0-7] of the first 16 */
    b = _mm256_permute4x64_epi64(b, 0xD8);
    *ev = _mm256_permute2x128_si256(a, b, 0x20);
    *od = _mm256_permute2x128_si256(a, b, 0x31);
}

typedef struct { int Th, Tw, T, Hx, Wx; } qwgeo;

/* V (16 vectors of 16 int16: one per xi, lanes = tiles tg..tg+15) of channel plane xc */
static inline void qw_input16(const qwgeo *g, const int16_t *xc, int tg, int nv, __m256i V[16])
{
    __m256i d[16];
    const int ty = tg / g->Tw, tx = tg % g->Tw;
    if (xc == NULL) {
        for (int q = 0; q < 16; ++q) V[q] = _mm256_setzero_si256();
        return;
    }
    if (nv == 16 && tx + 15 < g->Tw) {
        const int16_t *base = xc + (long)ty * 2 * g->Wx + tx * 2;
        for (int i = 0; i < 4; ++i) {
            deint16(base + (long)i * g->Wx, &d[i * 4 + 0], &d[i * 4 + 1]);
            deint16(base + (long)i * g->Wx + 2, &d[i * 4 + 2], &d[i * 4 + 3]);
        }
    } else {
        int16_t st[16][16] __attribute__((aligned(32)));
        for (int t = 0; t < 16; ++t) {
            if (t < nv) {
                const int tt = tg + t, y0 = (tt / g->Tw) * 2, x0 = (tt % g->Tw) * 2;
                for (int i = 0; i < 4; ++i)
                    for (int j = 0; j < 4; ++j) st[i * 4 + j][t] = xc[(long)(y0 + i) * g->Wx + x0 + j];
            } else {
                for (int q = 0; q < 16; ++q) st[q][t] = 0;
            }
        }
        for (int q = 0; q < 16; ++q) d[q] = _mm256_load_si256((const __m256i *)st[q]);
    }
    __m256i e[16], col[4], r[4];
    for (int i = 0; i < 4; ++i) bt2_16(d + i * 4, e + i * 4);
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) col[i] = e[i * 4 + j];
        bt2_16(col, r);
        for (int i = 0; i < 4; ++i) V[i * 4 + j] = r[i];
    }
}

/* acc[12] = sum_{c2} A(6 rows x 2 int16) . B(16 tiles x 2 int16) */
static inline void qw_kernel(int c2n, const int16_t *A, const int16_t *B, __m256i acc[12])
{
    __m256i c00 = _mm256_setzero_si256(), c01 = _mm256_setzero_si256();
    __m256i c10 = _mm256_setzero_si256(), c11 = _mm256_setzero_si256();
    __m256i c20 = _mm256_setzero_si256(), c21 = _mm256_setzero_si256();
    __m256i c30 = _mm256_setzero_si256(), c31 = _mm256_setzero_si256();
    __m256i c40 = _mm256_setzero_si256(), c41 = _mm256_setzero_si256();
    __m256i c50 = _mm256_setzero_si256(), c51 = _mm256_setzero_si256();
    const ug_i32_alias *a32 = (const ug_i32_alias *)A;
#pragma GCC unroll 4
    for (int s = 0; s < c2n; ++s) {
        __m256i b0 = _mm256_load_si256((const __m256i *)B);
        __m256i b1 = _mm256_load_si256((const __m256i *)(B + 16));
        __m256i a;
        a = _mm256_set1_epi32(a32[0]); c00 = ug_dpwssd(c00, b0, a); c01 = ug_dpwssd(c01, b1, a);
        a = _mm256_set1_epi32(a32[1]); c10 = ug_dpwssd(c10, b0, a); c11 = ug_dpwssd(c11, b1, a);
        a = _mm256_set1_epi32(a32[2]); c20 = ug_dpwssd(c20, b0, a); c21 = ug_dpwssd(c21, b1, a);
        a = _mm256_set1_epi32(a32[3]); c30 = ug_dpwssd(c30, b0, a); c31 = ug_dpwssd(c31, b1, a);
        a = _mm256_set1_epi32(a32[4]); c40 = ug_dpwssd(c40, b0, a); c41 = ug_dpwssd(c41, b1, a);
        a = _mm256_set1_epi32(a32[5]); c50 = ug_dpwssd(c50, b0, a); c51 = ug_dpwssd(c51, b1, a);
        a32 += UG_MR;
        B += 32;
    }
    acc[0] = c00; acc[1] = c01; acc[2] = c10; acc[3] = c11; acc[4] = c20; acc[5] = c21;
    acc[6] = c30; acc[7] = c31; acc[8] = c40; acc[9] = c41; acc[10] = c50; acc[11] = c51;
}

/* Output transform + epilogue for 8 tiles of channel k (M: 16 planes, stride ms). */
static inline void qw_output8(const ug_qconv_plan *p, const qwgeo *g, const ug_qout *o, int k,
                              const int32_t *M, long ms, int tg, int nv)
{
    const int P = p->P, Q = p->Q;
    __m256i m[16], e[8], col[4], y[2], Y[4];
    for (int q = 0; q < 16; ++q) m[q] = _mm256_load_si256((const __m256i *)(M + q * ms));
    for (int i = 0; i < 4; ++i) at2_32(m + i * 4, e + i * 2);
    for (int x = 0; x < 2; ++x) {
        for (int i = 0; i < 4; ++i) col[i] = e[i * 2 + x];
        at2_32(col, y);
        Y[0 * 2 + x] = _mm256_srai_epi32(y[0], 2);  /* exact: 4Y is a multiple of 4 */
        Y[1 * 2 + x] = _mm256_srai_epi32(y[1], 2);
    }
    const int ty = tg / g->Tw, tx = tg % g->Tw;
    const long kbase = (long)k * P * Q;
    if (nv == 8 && tx + 7 < g->Tw && (tx + 8) * 2 <= Q && ty * 2 + 2 <= P) {
        for (int yy = 0; yy < 2; ++yy) {
            __m256i lo = _mm256_unpacklo_epi32(Y[yy * 2], Y[yy * 2 + 1]);
            __m256i hi = _mm256_unpackhi_epi32(Y[yy * 2], Y[yy * 2 + 1]);
            __m256i r0 = _mm256_permute2x128_si256(lo, hi, 0x20);
            __m256i r1 = _mm256_permute2x128_si256(lo, hi, 0x31);
            ug_q_epilogue16(p, o, k, kbase + (long)(ty * 2 + yy) * Q + tx * 2, r0, r1, 16, 1);
        }
        return;
    }
    int32_t st[4][8] __attribute__((aligned(32)));
    for (int q = 0; q < 4; ++q) _mm256_store_si256((__m256i *)st[q], Y[q]);
    for (int t = 0; t < nv; ++t) {
        const int tt = tg + t, y0 = (tt / g->Tw) * 2, x0 = (tt % g->Tw) * 2;
        for (int yy = 0; yy < 2 && y0 + yy < P; ++yy) {
            int nx = ug_min(2, Q - x0);
            __m256i v = _mm256_setr_epi32(st[yy * 2][t], st[yy * 2 + 1][t], 0, 0, 0, 0, 0, 0);
            ug_q_epilogue16(p, o, k, kbase + (long)(y0 + yy) * Q + x0, v, _mm256_setzero_si256(), nx, 1);
        }
    }
}

int ug_qwino_execute(const ug_qconv_plan *p, const uint8_t *in, const ug_qout *o, long img_elems)
{
    const ug_conv_desc *d = &p->d;
    const int N = d->N, C = d->C, H = d->H, W = d->W, K = d->K;
    const int P = p->P, Q = p->Q, nt = p->nthreads;
    const int kpan = p->Kp / UG_MR, c2n = p->C2 / 2;
    qwgeo g;
    g.Th = ug_ceil_div(P, 2); g.Tw = ug_ceil_div(Q, 2); g.T = g.Th * g.Tw;
    g.Hx = g.Th * 2 + 2; g.Wx = g.Tw * 2 + 2 + 16;

    /* blocking: V = 16*C2*TB*2 B, M = 16*KB*TB*4 B, together <= budget */
    int KB = p->Kp <= 96 ? p->Kp : 48;
    int TB = 128;
    while (TB > 16 && 16L * TB * (p->C2 * 2 + KB * 4) > QW_BUDGET) TB -= 16;
    const int want = nt > 1 ? 3 * nt : 1;
    while (ug_ceil_div(g.T, TB) * ug_ceil_div(p->Kp, KB) < want && TB > 16) TB -= 16;
    while (ug_ceil_div(g.T, TB) * ug_ceil_div(p->Kp, KB) < want && KB > 12) KB = ug_round_up(KB / 2, UG_MR);
    const int tblk = ug_ceil_div(g.T, TB), kblk = ug_ceil_div(p->Kp, KB);
    const int items = tblk * kblk;
    const long vs = (long)TB * p->C2 + 32;   /* int16 per xi plane (+1 line: checklist D6) */
    const long ms = (long)KB * TB + 16;      /* int32 per xi plane (+1 line) */

    const long plane = (long)g.Hx * g.Wx;
    int16_t *X = ug_malloc((size_t)C * plane * sizeof(int16_t));
    if (!X) return -1;
    int err = 0;

    for (int n = 0; n < N && !err; ++n) {
        const uint8_t *x = in + (size_t)n * C * H * W;
        const ug_qout oi = ug_qout_image(o, n, img_elems);
#pragma omp parallel num_threads(nt)
        {
#pragma omp for schedule(static)
            for (int cr = 0; cr < C * g.Hx; ++cr) {
                const int c = cr / g.Hx, yy = cr % g.Hx, ih = yy - d->pad_h;
                int16_t *dst = X + c * plane + (long)yy * g.Wx;
                for (int j = 0; j < g.Wx; ++j) {
                    const int iw = j - d->pad_w;
                    dst[j] = (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                 ? (int16_t)(x[((size_t)c * H + ih) * W + iw] - p->in_zp) : 0;
                }
            }
            int16_t *V = ug_malloc((size_t)16 * vs * sizeof(int16_t));
            int32_t *M = ug_malloc((size_t)16 * ms * sizeof(int32_t));
            if (!V || !M) {
#pragma omp atomic write
                err = 1;
            }
#pragma omp for schedule(dynamic, 1)
            for (int it = 0; it < items; ++it) {
                if (!V || !M) continue; /* F5 */
                const int tb = it / kblk, kb = it % kblk;
                const int t0 = tb * TB, tcount = ug_min(TB, g.T - t0), npan = ug_ceil_div(tcount, 16);
                const int k0 = kb * KB, kcount = ug_min(KB, p->Kp - k0);
                /* input transform: channel pairs, 16 tiles at a time, interleaved (c, c+1) */
                for (int c2 = 0; c2 < c2n; ++c2)
                    for (int jp = 0; jp < npan; ++jp) {
                        __m256i va[16], vb[16];
                        const int nv = ug_min(16, tcount - jp * 16);
                        qw_input16(&g, X + (long)(2 * c2) * plane, t0 + jp * 16, nv, va);
                        qw_input16(&g, 2 * c2 + 1 < C ? X + (long)(2 * c2 + 1) * plane : NULL, t0 + jp * 16, nv, vb);
                        for (int xi = 0; xi < 16; ++xi) {
                            int16_t *dst = V + xi * vs + ((long)jp * c2n + c2) * 32;
                            __m256i lo = _mm256_unpacklo_epi16(va[xi], vb[xi]);
                            __m256i hi = _mm256_unpackhi_epi16(va[xi], vb[xi]);
                            _mm256_store_si256((__m256i *)dst, _mm256_permute2x128_si256(lo, hi, 0x20));
                            _mm256_store_si256((__m256i *)(dst + 16), _mm256_permute2x128_si256(lo, hi, 0x31));
                        }
                    }
                /* 16 GEMMs */
                for (int xi = 0; xi < 16; ++xi)
                    for (int jp = 0; jp < npan; ++jp)
                        for (int ip = k0 / UG_MR; ip < (k0 + kcount) / UG_MR; ++ip) {
                            __m256i acc[12];
                            qw_kernel(c2n, p->ww + (((size_t)xi * kpan + ip) * c2n) * UG_MR * 2,
                                      V + xi * vs + (long)jp * c2n * 32, acc);
                            int32_t *dst = M + xi * ms + (long)(ip * UG_MR - k0) * TB + jp * 16;
                            for (int i = 0; i < UG_MR; ++i) {
                                _mm256_store_si256((__m256i *)(dst + (long)i * TB), acc[2 * i]);
                                _mm256_store_si256((__m256i *)(dst + (long)i * TB + 8), acc[2 * i + 1]);
                            }
                        }
                /* output transform + epilogue */
                const int kreal = ug_min(k0 + kcount, K) - k0;
                for (int k = 0; k < kreal; ++k)
                    for (int t8 = 0; t8 * 8 < tcount; ++t8)
                        qw_output8(p, &g, &oi, k0 + k, M + (long)k * TB + t8 * 8, ms, t0 + t8 * 8,
                                   ug_min(8, tcount - t8 * 8));
            }
            ug_free(V);
            ug_free(M);
        }
    }
    ug_free(X);
    return err ? -1 : 0;
}
