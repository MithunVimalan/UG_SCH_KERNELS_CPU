/* 256-bit VNNI primitives, three interchangeable backends with identical results.
 *
 *   dpbusd(acc, a, b): acc.i32[l] += sum_{j<4} u8(a.byte[4l+j]) * s8(b.byte[4l+j])
 *   dpwssd(acc, a, b): acc.i32[l] += s16(a.w[2l]) * s16(b.w[2l]) + s16(a.w[2l+1]) * s16(b.w[2l+1])
 * Both are non-saturating: int32 wrap-around, i.e. exact arithmetic mod 2^32.
 *
 *   UG_VNNI_VEX   AVX-VNNI (VEX, Arrow Lake / Alder Lake+)       -> build ARCH=arrowlake
 *   UG_VNNI_EVEX  AVX512-VNNI + AVX512VL on ymm (Ice Lake+, Xeon) -> dev-VM build
 *   UG_VNNI_EMU   AVX2 only, exact emulation (4-5 instructions)   -> build VNNI=emu
 */
#ifndef UG_VNNI_H
#define UG_VNNI_H
#include <immintrin.h>

#if defined(UG_FORCE_VNNI_EMU)
#define UG_VNNI_EMU 1
#elif defined(__AVXVNNI__)
#define UG_VNNI_VEX 1
#elif defined(__AVX512VNNI__) && defined(__AVX512VL__)
#define UG_VNNI_EVEX 1
#else
#define UG_VNNI_EMU 1
#endif

static inline __m256i ug_dpbusd(__m256i acc, __m256i a_u8, __m256i b_s8)
{
#if defined(UG_VNNI_VEX)
    return _mm256_dpbusd_avx_epi32(acc, a_u8, b_s8);
#elif defined(UG_VNNI_EVEX)
    return _mm256_dpbusd_epi32(acc, a_u8, b_s8);
#else
    /* bytes 0,2 and 1,3 of every dword widened to 16 bit, then pairwise madd.
     * |u8*s8 + u8*s8| <= 2*255*128 = 65280 fits int32; no saturation anywhere. */
    const __m256i lo = _mm256_set1_epi16(0x00FF);
    __m256i a_e = _mm256_and_si256(a_u8, lo);
    __m256i a_o = _mm256_srli_epi16(a_u8, 8);
    __m256i b_e = _mm256_srai_epi16(_mm256_slli_epi16(b_s8, 8), 8);
    __m256i b_o = _mm256_srai_epi16(b_s8, 8);
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_e, b_e));
    return _mm256_add_epi32(acc, _mm256_madd_epi16(a_o, b_o));
#endif
}

static inline __m256i ug_dpwssd(__m256i acc, __m256i a_s16, __m256i b_s16)
{
#if defined(UG_VNNI_VEX)
    return _mm256_dpwssd_avx_epi32(acc, a_s16, b_s16);
#elif defined(UG_VNNI_EVEX)
    return _mm256_dpwssd_epi32(acc, a_s16, b_s16);
#else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(a_s16, b_s16));
#endif
}

static inline const char *ug_vnni_backend(void)
{
#if defined(UG_VNNI_VEX)
    return "AVX-VNNI (VEX)";
#elif defined(UG_VNNI_EVEX)
    return "AVX512-VNNI (EVEX, ymm) [dev-VM stand-in for AVX-VNNI]";
#else
    return "AVX2 emulation (exact)";
#endif
}
#endif
