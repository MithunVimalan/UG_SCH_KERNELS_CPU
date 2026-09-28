/* Group 3: Winograd F(4x4,3x3) and F(2x2,3x3), stride 1.
 *
 *   Y = A^T [ U (.) V ] A,   U = G g G^T (offline),   V = B^T d B (runtime)
 *
 * Summed over input channels, the elementwise product becomes alpha^2 independent
 * GEMMs  M_xi[K][T] = U_xi[K][C] x V_xi[C][T]  (alpha = m+2), run with the same
 * 6x16 micro-kernel as group 2. Everything is fused per work item
 * (image, tile block, output-channel block) so V and M live in L2 and never go
 * to DRAM: unfused Winograd drops to ~3 MAC/byte and loses the speed-up.
 *
 * Transforms are vectorised over 8 tiles. When 8 tiles share a tile row the
 * 4x8 / 2x8 transposes are done in registers; other groups go through a small
 * staging buffer.
 */
#include <immintrin.h>
#include <string.h>
#include "ug_internal.h"

#define WINO_L2_BUDGET (768 * 1024) /* per thread: V + M; Skymont has 1 MB L2 per core */

/* ---------------- filter transform (offline, fp64) ---------------- */

static const double G4[6][3] = {
    {1.0 / 4, 0, 0},
    {-1.0 / 6, -1.0 / 6, -1.0 / 6},
    {-1.0 / 6, 1.0 / 6, -1.0 / 6},
    {1.0 / 24, 1.0 / 12, 1.0 / 6},
    {1.0 / 24, -1.0 / 12, 1.0 / 6},
    {0, 0, 1},
};
static const double G2[4][3] = {
    {1, 0, 0},
    {0.5, 0.5, 0.5},
    {0.5, -0.5, 0.5},
    {0, 0, 1},
};

int ug_wino_prepare(ug_conv_plan *p, const float *w)
{
    const ug_conv_desc *d = &p->d;
    const int m = p->wino_m, a = m + 2, C = d->C, K = d->K;
    const int kpan = p->Kp / UG_MR;
    p->wwino = ug_malloc((size_t)a * a * p->Kp * C * sizeof(float));
    if (!p->wwino) return -1;
    memset(p->wwino, 0, (size_t)a * a * p->Kp * C * sizeof(float));
    for (int k = 0; k < K; ++k)
        for (int c = 0; c < C; ++c) {
            const float *g = w + ((size_t)k * C + c) * 9;
            double t[6][3], u;
            for (int i = 0; i < a; ++i)
                for (int j = 0; j < 3; ++j) {
                    double s = 0;
                    for (int l = 0; l < 3; ++l) s += (m == 4 ? G4[i][l] : G2[i][l]) * g[l * 3 + j];
                    t[i][j] = s;
                }
            for (int i = 0; i < a; ++i)
                for (int j = 0; j < a; ++j) {
                    u = 0;
                    for (int l = 0; l < 3; ++l) u += t[i][l] * (m == 4 ? G4[j][l] : G2[j][l]);
                    int xi = i * a + j;
                    p->wwino[(((size_t)xi * kpan + k / UG_MR) * C + c) * UG_MR + k % UG_MR] = (float)u;
                }
        }
    return 0;
}

/* ---------------- 1-D transforms on vectors ---------------- */

static inline void bt4(const __m256 d[6], __m256 r[6])
{
    const __m256 two = _mm256_set1_ps(2.f), four = _mm256_set1_ps(4.f), five = _mm256_set1_ps(5.f);
    __m256 t0 = _mm256_fnmadd_ps(four, d[2], d[4]); /* d4 - 4 d2 */
    __m256 t1 = _mm256_fnmadd_ps(four, d[1], d[3]); /* d3 - 4 d1 */
    __m256 t2 = _mm256_sub_ps(d[4], d[2]);
    __m256 t3 = _mm256_mul_ps(two, _mm256_sub_ps(d[3], d[1]));
    r[0] = _mm256_add_ps(_mm256_fnmadd_ps(five, d[2], _mm256_mul_ps(four, d[0])), d[4]);
    r[1] = _mm256_add_ps(t0, t1);
    r[2] = _mm256_sub_ps(t0, t1);
    r[3] = _mm256_add_ps(t2, t3);
    r[4] = _mm256_sub_ps(t2, t3);
    r[5] = _mm256_add_ps(_mm256_fnmadd_ps(five, d[3], _mm256_mul_ps(four, d[1])), d[5]);
}

static inline void at4(const __m256 m[6], __m256 y[4])
{
    __m256 a = _mm256_add_ps(m[1], m[2]), b = _mm256_sub_ps(m[1], m[2]);
    __m256 c = _mm256_add_ps(m[3], m[4]), e = _mm256_sub_ps(m[3], m[4]);
    y[0] = _mm256_add_ps(_mm256_add_ps(m[0], a), c);
    y[1] = _mm256_fmadd_ps(_mm256_set1_ps(2.f), e, b);
    y[2] = _mm256_fmadd_ps(_mm256_set1_ps(4.f), c, a);
    y[3] = _mm256_add_ps(_mm256_fmadd_ps(_mm256_set1_ps(8.f), e, b), m[5]);
}

static inline void bt2(const __m256 d[4], __m256 r[4])
{
    r[0] = _mm256_sub_ps(d[0], d[2]);
    r[1] = _mm256_add_ps(d[1], d[2]);
    r[2] = _mm256_sub_ps(d[2], d[1]);
    r[3] = _mm256_sub_ps(d[1], d[3]);
}

static inline void at2(const __m256 m[4], __m256 y[2])
{
    y[0] = _mm256_add_ps(_mm256_add_ps(m[0], m[1]), m[2]);
    y[1] = _mm256_sub_ps(_mm256_sub_ps(m[1], m[2]), m[3]);
}

/* 32 contiguous floats = 8 tiles x 4 columns  ->  4 vectors (column x, lanes = tiles). */
static inline void tr_in4(const float *src, __m256 x[4])
{
    __m256 r0 = _mm256_loadu_ps(src), r1 = _mm256_loadu_ps(src + 8);
    __m256 r2 = _mm256_loadu_ps(src + 16), r3 = _mm256_loadu_ps(src + 24);
    __m256 e0 = _mm256_permute2f128_ps(r0, r2, 0x20), e1 = _mm256_permute2f128_ps(r0, r2, 0x31);
    __m256 e2 = _mm256_permute2f128_ps(r1, r3, 0x20), e3 = _mm256_permute2f128_ps(r1, r3, 0x31);
    __m256 a = _mm256_unpacklo_ps(e0, e1), b = _mm256_unpackhi_ps(e0, e1);
    __m256 c = _mm256_unpacklo_ps(e2, e3), dd = _mm256_unpackhi_ps(e2, e3);
    x[0] = _mm256_shuffle_ps(a, c, 0x44);
    x[1] = _mm256_shuffle_ps(a, c, 0xEE);
    x[2] = _mm256_shuffle_ps(b, dd, 0x44);
    x[3] = _mm256_shuffle_ps(b, dd, 0xEE);
}

/* inverse of tr_in4: 4 vectors (column x, lanes = tiles) -> 32 contiguous floats */
static inline void tr_out4(float *dst, const __m256 x[4])
{
    __m256 a = _mm256_unpacklo_ps(x[0], x[1]), b = _mm256_unpackhi_ps(x[0], x[1]);
    __m256 c = _mm256_unpacklo_ps(x[2], x[3]), dd = _mm256_unpackhi_ps(x[2], x[3]);
    __m256 e0 = _mm256_shuffle_ps(a, c, 0x44), e1 = _mm256_shuffle_ps(a, c, 0xEE);
    __m256 e2 = _mm256_shuffle_ps(b, dd, 0x44), e3 = _mm256_shuffle_ps(b, dd, 0xEE);
    _mm256_storeu_ps(dst, _mm256_permute2f128_ps(e0, e1, 0x20));
    _mm256_storeu_ps(dst + 8, _mm256_permute2f128_ps(e2, e3, 0x20));
    _mm256_storeu_ps(dst + 16, _mm256_permute2f128_ps(e0, e1, 0x31));
    _mm256_storeu_ps(dst + 24, _mm256_permute2f128_ps(e2, e3, 0x31));
}

/* 16 contiguous floats = 8 tiles x 2 columns -> even / odd vectors */
static inline void tr_in2(const float *src, __m256 *ev, __m256 *od)
{
    __m256 a = _mm256_loadu_ps(src), b = _mm256_loadu_ps(src + 8);
    __m256 e = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(2, 0, 2, 0));
    __m256 o = _mm256_shuffle_ps(a, b, _MM_SHUFFLE(3, 1, 3, 1));
    *ev = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(e), 0xD8));
    *od = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(o), 0xD8));
}

static inline void tr_out2(float *dst, __m256 x0, __m256 x1)
{
    __m256 lo = _mm256_unpacklo_ps(x0, x1), hi = _mm256_unpackhi_ps(x0, x1);
    _mm256_storeu_ps(dst, _mm256_permute2f128_ps(lo, hi, 0x20));
    _mm256_storeu_ps(dst + 8, _mm256_permute2f128_ps(lo, hi, 0x31));
}

/* ---------------- execution ---------------- */

typedef struct {
    int m, a, Th, Tw, T, Hx, Wx;
} wgeo;

/* Load the alpha x alpha patch of 8 tiles (tg..tg+nv-1) of channel plane xc into d[i*a+j]. */
static inline __attribute__((always_inline)) void load_patch(const wgeo *g, const int m, const float *xc, int tg, int nv, __m256 *d)
{
    const int a = m + 2;
    const int ty = tg / g->Tw, tx = tg % g->Tw;
    if (nv == 8 && tx + 7 < g->Tw) {
        const float *base = xc + (long)ty * m * g->Wx + tx * m;
        for (int i = 0; i < a; ++i) {
            const float *row = base + (long)i * g->Wx;
            if (m == 4) {
                __m256 lo[4], hi[4];
                tr_in4(row, lo);
                tr_in4(row + 4, hi);
                d[i * 6 + 0] = lo[0]; d[i * 6 + 1] = lo[1]; d[i * 6 + 2] = lo[2];
                d[i * 6 + 3] = lo[3]; d[i * 6 + 4] = hi[0]; d[i * 6 + 5] = hi[1];
            } else {
                tr_in2(row, &d[i * 4 + 0], &d[i * 4 + 1]);
                tr_in2(row + 2, &d[i * 4 + 2], &d[i * 4 + 3]);
            }
        }
        return;
    }
    float st[36 * 8] __attribute__((aligned(32)));
    for (int t = 0; t < 8; ++t) {
        if (t < nv) {
            const int tt = tg + t, y0 = (tt / g->Tw) * m, x0 = (tt % g->Tw) * m;
            for (int i = 0; i < a; ++i)
                for (int j = 0; j < a; ++j) st[(i * a + j) * 8 + t] = xc[(long)(y0 + i) * g->Wx + x0 + j];
        } else {
            for (int q = 0; q < a * a; ++q) st[q * 8 + t] = 0.f;
        }
    }
    for (int q = 0; q < a * a; ++q) d[q] = _mm256_load_ps(st + q * 8);
}

/* V = B^T d B, written to V[xi*xstride + lane] for xi in 0..a*a-1 */
static inline __attribute__((always_inline)) void input_transform(const int m, const __m256 *d, float *V, long xstride)
{
    if (m == 4) {
        __m256 e[36], col[6], r[6];
        for (int i = 0; i < 6; ++i) bt4(d + i * 6, e + i * 6);
        for (int j = 0; j < 6; ++j) {
            for (int i = 0; i < 6; ++i) col[i] = e[i * 6 + j];
            bt4(col, r);
            for (int i = 0; i < 6; ++i) _mm256_store_ps(V + (i * 6 + j) * xstride, r[i]);
        }
    } else {
        __m256 e[16], col[4], r[4];
        for (int i = 0; i < 4; ++i) bt2(d + i * 4, e + i * 4);
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) col[i] = e[i * 4 + j];
            bt2(col, r);
            for (int i = 0; i < 4; ++i) _mm256_store_ps(V + (i * 4 + j) * xstride, r[i]);
        }
    }
}

/* Output transform of 8 tiles of one output channel, fused bias + ReLU + store. */
static inline __attribute__((always_inline)) void output_tiles(const wgeo *g, const int m, int P, int Q, const float *M, long xstride,
                                int tg, int nv, float bias, int relu, float *yk)
{
    const int a = m + 2;
    __m256 mv[36], e[24], col[6], y[4], Y[16];
    for (int q = 0; q < a * a; ++q) mv[q] = _mm256_load_ps(M + q * xstride);
    /* rows first: e[i][x] = sum_j m[i][j] A[j][x] */
    for (int i = 0; i < a; ++i) {
        if (m == 4) at4(mv + i * 6, e + i * 4); else at2(mv + i * 4, e + i * 2);
    }
    const __m256 b = _mm256_set1_ps(bias), zero = _mm256_setzero_ps();
    for (int x = 0; x < m; ++x) {
        for (int i = 0; i < a; ++i) col[i] = e[i * m + x];
        if (m == 4) at4(col, y); else at2(col, y);
        for (int yy = 0; yy < m; ++yy) {
            __m256 v = _mm256_add_ps(y[yy], b);
            if (relu) v = _mm256_max_ps(zero, v); /* keeps NaN */
            Y[yy * m + x] = v;
        }
    }
    const int ty = tg / g->Tw, tx = tg % g->Tw;
    if (nv == 8 && tx + 7 < g->Tw && (tx + 8) * m <= Q && ty * m + m <= P) {
        float *base = yk + (long)ty * m * Q + tx * m;
        for (int yy = 0; yy < m; ++yy) {
            if (m == 4) tr_out4(base + (long)yy * Q, Y + yy * 4);
            else tr_out2(base + (long)yy * Q, Y[yy * 2], Y[yy * 2 + 1]);
        }
        return;
    }
    float st[16 * 8] __attribute__((aligned(32)));
    for (int q = 0; q < m * m; ++q) _mm256_store_ps(st + q * 8, Y[q]);
    for (int t = 0; t < nv; ++t) {
        const int tt = tg + t, y0 = (tt / g->Tw) * m, x0 = (tt % g->Tw) * m;
        for (int yy = 0; yy < m && y0 + yy < P; ++yy)
            for (int x = 0; x < m && x0 + x < Q; ++x)
                yk[(long)(y0 + yy) * Q + x0 + x] = st[(yy * m + x) * 8 + t];
    }
}

void ug_wino_blocking(const ug_conv_desc *d, int m, int nt, ug_wino_blk *b)
{
    const int P = ug_out_h(d), Q = ug_out_w(d), Kp = ug_round_up(d->K, UG_MR), a2 = (m + 2) * (m + 2);
    b->m = m;
    b->Th = ug_ceil_div(P, m); b->Tw = ug_ceil_div(Q, m); b->T = b->Th * b->Tw;
    int KB = Kp <= 192 ? Kp : 96;          /* rows of M per item, multiple of 6 */
    b->CC = d->C <= 128 ? d->C : 128;      /* channel chunk of V */
    int TB = 64;                           /* tiles per item, multiple of 16 */
    while (TB > 16 && (long)a2 * TB * (b->CC + KB) * 4 > WINO_L2_BUDGET) TB -= 16;
    /* items are scheduled per image: split tiles, then output channels, for parallelism */
    const int want = nt > 1 ? 3 * nt : 1;
    while (ug_ceil_div(b->T, TB) * ug_ceil_div(Kp, KB) < want && TB > 16) TB -= 16;
    while (ug_ceil_div(b->T, TB) * ug_ceil_div(Kp, KB) < want && KB > 24) KB = ug_round_up(KB / 2, UG_MR);
    b->TB = TB; b->KB = KB;
    b->tblk = ug_ceil_div(b->T, TB); b->kblk = ug_ceil_div(Kp, KB);
}

/* Constant-m entry points: every transform loop above unrolls fully. */
static void in_group4(const wgeo *g, const float *xc, int tg, int nv, float *V, long xs)
{
    __m256 dv[36];
    load_patch(g, 4, xc, tg, nv, dv);
    input_transform(4, dv, V, xs);
}
static void in_group2(const wgeo *g, const float *xc, int tg, int nv, float *V, long xs)
{
    __m256 dv[16];
    load_patch(g, 2, xc, tg, nv, dv);
    input_transform(2, dv, V, xs);
}
static void out_group4(const wgeo *g, int P, int Q, const float *M, long xs, int tg, int nv, float b, int relu, float *yk)
{
    output_tiles(g, 4, P, Q, M, xs, tg, nv, b, relu, yk);
}
static void out_group2(const wgeo *g, int P, int Q, const float *M, long xs, int tg, int nv, float b, int relu, float *yk)
{
    output_tiles(g, 2, P, Q, M, xs, tg, nv, b, relu, yk);
}

int ug_wino_execute(const ug_conv_plan *p, const float *in, float *out)
{
    const ug_conv_desc *d = &p->d;
    const int N = d->N, C = d->C, H = d->H, W = d->W, K = d->K;
    const int P = p->P, Q = p->Q, PQ = P * Q;
    const int nt = p->nthreads;
    wgeo g;
    g.m = p->wino_m; g.a = g.m + 2;
    g.Th = ug_ceil_div(P, g.m); g.Tw = ug_ceil_div(Q, g.m); g.T = g.Th * g.Tw;
    g.Hx = g.Th * g.m + 2; g.Wx = g.Tw * g.m + 8;
    const int a2 = g.a * g.a;
    const int kpan = p->Kp / UG_MR;

    ug_wino_blk bk;
    ug_wino_blocking(d, g.m, nt, &bk);
    const int KB = bk.KB, CC = bk.CC, TB = bk.TB, kblk = bk.kblk;
    const int items_per_img = bk.tblk * bk.kblk;

    /* Checklist D6: V and M hold alpha^2 planes that are touched together (36
     * stores / loads per 8-tile group). A plane pitch that is a multiple of 4 KiB
     * puts all of them in ONE L1 set; one extra cache line per plane spreads them
     * over consecutive sets. */
    const long vstride = (long)TB * CC + 16;
    const long mstride = (long)KB * TB + 16;
    const long plane = (long)g.Hx * g.Wx;
    float *X = ug_malloc((size_t)C * plane * sizeof(float));
    if (!X) return -1;
    int err = 0;

    for (int n = 0; n < N && !err; ++n) {
        const float *x = in + (size_t)n * C * H * W;
        float *yout = out + (size_t)n * K * PQ;
#pragma omp parallel num_threads(nt)
        {
#pragma omp for schedule(static)
            for (int cr = 0; cr < C * g.Hx; ++cr) {
                const int c = cr / g.Hx, yy = cr % g.Hx, ih = yy - d->pad_h;
                float *dst = X + c * plane + (long)yy * g.Wx;
                if (ih < 0 || ih >= H) { memset(dst, 0, g.Wx * sizeof(float)); continue; }
                const float *src = x + ((size_t)c * H + ih) * W;
                const int ncp = ug_min(W, g.Wx - d->pad_w);
                memset(dst, 0, d->pad_w * sizeof(float));
                memcpy(dst + d->pad_w, src, ncp * sizeof(float));
                memset(dst + d->pad_w + ncp, 0, (g.Wx - d->pad_w - ncp) * sizeof(float));
            }

            float *V = ug_malloc((size_t)a2 * vstride * sizeof(float));
            float *M = ug_malloc((size_t)a2 * mstride * sizeof(float));
            if (!V || !M) {
#pragma omp atomic write
                err = 1;
            }
#pragma omp for schedule(dynamic, 1)
            for (int it = 0; it < items_per_img; ++it) {
                if (!V || !M) continue; /* F5 */
                const int tb = it / kblk, kb = it % kblk;
                const int t0 = tb * TB, tcount = ug_min(TB, g.T - t0);
                const int npan = ug_ceil_div(tcount, UG_NR);
                const int k0 = kb * KB, kcount = ug_min(KB, p->Kp - k0);

                for (int c0 = 0; c0 < C; c0 += CC) {
                    const int cc = ug_min(CC, C - c0);
                    const long vx = vstride;
                    for (int c = 0; c < cc; ++c) {
                        const float *xc = X + (long)(c0 + c) * plane;
                        for (int t8 = 0; t8 < npan * 2; ++t8) {
                            const int nv = ug_max(0, ug_min(8, tcount - t8 * 8));
                            float *vp = V + ((long)(t8 / 2) * cc + c) * UG_NR + (t8 % 2) * 8;
                            if (nv == 0)
                                for (int q = 0; q < a2; ++q) _mm256_store_ps(vp + q * vx, _mm256_setzero_ps());
                            else if (g.m == 4) in_group4(&g, xc, t0 + t8 * 8, nv, vp, vx);
                            else in_group2(&g, xc, t0 + t8 * 8, nv, vp, vx);
                        }
                    }
                    for (int xi = 0; xi < a2; ++xi) {
                        const float *Vx = V + xi * vx;
                        float *Mx = M + xi * mstride;
                        const float *Ux = p->wwino + (((size_t)xi * kpan) * C + c0) * UG_MR;
                        for (int jp = 0; jp < npan; ++jp)
                            for (int ip = k0 / UG_MR; ip < (k0 + kcount) / UG_MR; ++ip)
                                ug_kernel_6x16(cc, Ux + (size_t)ip * C * UG_MR, Vx + (long)jp * cc * UG_NR,
                                               Mx + (long)(ip * UG_MR - k0) * TB + jp * UG_NR, TB,
                                               UG_MR, UG_NR, c0 > 0, NULL, 0);
                    }
                }
                const int kreal = ug_min(k0 + kcount, K) - k0;
                for (int k = 0; k < kreal; ++k) {
                    float *yk = yout + (size_t)(k0 + k) * PQ;
                    for (int t8 = 0; t8 * 8 < tcount; ++t8)
                        (g.m == 4 ? out_group4 : out_group2)(&g, P, Q, M + (long)k * TB + t8 * 8, mstride,
                                                             t0 + t8 * 8, ug_min(8, tcount - t8 * 8),
                                                             p->bias[k0 + k], p->relu, yk);
                }
            }
            ug_free(V);
            ug_free(M);
        }
    }
    ug_free(X);
    return err ? -1 : 0;
}
