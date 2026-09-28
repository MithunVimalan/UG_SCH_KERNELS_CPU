/* 6x16 FP32 GEMM micro-kernel, AVX2 + FMA.
 *
 * Per k step: 2 ymm loads of B, 6 broadcasts of A, 12 FMAs.
 * Lion Cove: 12 FMA / 2 pipes = 6 cycles vs 8 loads / 3 ports = 2.7 cycles -> FMA-bound.
 * Skymont:   same 12 FMA at 2/cycle; loads 2x2 + 6 half-width slots / 3 = 3.3 cycles -> FMA-bound.
 */
#include <immintrin.h>
#include <string.h>
#include "ug_internal.h"

static inline void kernel_full(int kc, const float *A, const float *B, __m256 acc[12])
{
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();
#pragma GCC unroll 4
    for (int k = 0; k < kc; ++k) {
        __m256 b0 = _mm256_load_ps(B);
        __m256 b1 = _mm256_load_ps(B + 8);
        __m256 a;
        a = _mm256_broadcast_ss(A + 0); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
        a = _mm256_broadcast_ss(A + 1); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
        a = _mm256_broadcast_ss(A + 2); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
        a = _mm256_broadcast_ss(A + 3); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
        a = _mm256_broadcast_ss(A + 4); c40 = _mm256_fmadd_ps(a, b0, c40); c41 = _mm256_fmadd_ps(a, b1, c41);
        a = _mm256_broadcast_ss(A + 5); c50 = _mm256_fmadd_ps(a, b0, c50); c51 = _mm256_fmadd_ps(a, b1, c51);
        A += UG_MR;
        B += UG_NR;
    }
    acc[0] = c00; acc[1] = c01; acc[2] = c10; acc[3] = c11; acc[4] = c20; acc[5] = c21;
    acc[6] = c30; acc[7] = c31; acc[8] = c40; acc[9] = c41; acc[10] = c50; acc[11] = c51;
}

/* Right-edge variant for nr <= 8: only the first B vector is live, so skip the
 * second half (6 chains: latency-bound at 6/8 of peak, still 1.5x the full tile's
 * useful rate when nr <= 8). */
static inline void kernel_half(int kc, const float *A, const float *B, __m256 acc[12])
{
    __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps(), c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps(), c4 = _mm256_setzero_ps(), c5 = _mm256_setzero_ps();
#pragma GCC unroll 4
    for (int k = 0; k < kc; ++k) {
        __m256 b0 = _mm256_load_ps(B);
        c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 0), b0, c0);
        c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 1), b0, c1);
        c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 2), b0, c2);
        c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 3), b0, c3);
        c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 4), b0, c4);
        c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(A + 5), b0, c5);
        A += UG_MR;
        B += UG_NR;
    }
    const __m256 z = _mm256_setzero_ps();
    acc[0] = c0; acc[2] = c1; acc[4] = c2; acc[6] = c3; acc[8] = c4; acc[10] = c5;
    acc[1] = acc[3] = acc[5] = acc[7] = acc[9] = acc[11] = z;
}

void ug_kernel_6x16(int kc, const float *A, const float *B, float *C, long ldc,
                    int mr, int nr, int accumulate, const float *bias, int relu)
{
    __m256 acc[12];
    if (nr <= 8) kernel_half(kc, A, B, acc);
    else kernel_full(kc, A, B, acc);

    if (mr == UG_MR && nr == UG_NR) {
        const __m256 zero = _mm256_setzero_ps();
        for (int i = 0; i < UG_MR; ++i) {
            float *c = C + i * ldc;
            __m256 v0 = acc[2 * i], v1 = acc[2 * i + 1];
            if (accumulate) {
                v0 = _mm256_add_ps(v0, _mm256_loadu_ps(c));
                v1 = _mm256_add_ps(v1, _mm256_loadu_ps(c + 8));
            }
            if (bias) {
                __m256 b = _mm256_broadcast_ss(bias + i);
                v0 = _mm256_add_ps(v0, b);
                v1 = _mm256_add_ps(v1, b);
                if (relu) { v0 = _mm256_max_ps(v0, zero); v1 = _mm256_max_ps(v1, zero); }
            }
            _mm256_storeu_ps(c, v0);
            _mm256_storeu_ps(c + 8, v1);
        }
        return;
    }

    /* Edge tile: spill to a stack buffer and copy the valid part. */
    float tmp[UG_MR * UG_NR] __attribute__((aligned(32)));
    for (int i = 0; i < UG_MR; ++i) {
        _mm256_store_ps(tmp + i * UG_NR, acc[2 * i]);
        _mm256_store_ps(tmp + i * UG_NR + 8, acc[2 * i + 1]);
    }
    for (int i = 0; i < mr; ++i) {
        float *c = C + i * ldc;
        for (int j = 0; j < nr; ++j) {
            float v = tmp[i * UG_NR + j];
            if (accumulate) v += c[j];
            if (bias) {
                v += bias[i];
                if (relu && v < 0.f) v = 0.f;
            }
            c[j] = v;
        }
    }
}
