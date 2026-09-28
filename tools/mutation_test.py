#!/usr/bin/env python3
"""Mutation test: inject one bug at a time; the test suites must catch every one.
Run from the repo root:  python3 tools/mutation_test.py [extra make args]"""
import os, shutil, subprocess, sys, tempfile

MUTANTS = [
    # (file, original text, mutated text, test binary, what the bug is)
    ("src/q_winograd.c", "Y[0 * 2 + x] = _mm256_srai_epi32(y[0], 2);", "Y[0 * 2 + x] = _mm256_srai_epi32(y[0], 1);", "test_qconv", "Winograd output scale"),
    ("src/q_winograd.c", "r[3] = _mm256_sub_epi16(d[1], d[3]);", "r[3] = _mm256_add_epi16(d[1], d[3]);", "test_qconv", "B^T row 3"),
    ("src/q_winograd.c", "a = _mm256_permute4x64_epi64(a, 0xD8);", "a = _mm256_permute4x64_epi64(a, 0x8D);", "test_qconv", "fast-path deinterleave"),
    ("src/q_winograd.c", "y[1] = _mm256_sub_epi32(_mm256_sub_epi32(m[1], m[2]), m[3]);", "y[1] = _mm256_sub_epi32(_mm256_add_epi32(m[1], m[2]), m[3]);", "test_qconv", "A^T row 1"),
    ("src/q_gemm.c", "_mm_store_si128((__m128i *)(dst + 32), _mm_unpacklo_epi16(b, e));", "_mm_store_si128((__m128i *)(dst + 32), _mm_unpackhi_epi16(b, e));", "test_qconv", "im2col byte transpose"),
    ("src/q_gemm.c", "const __m128i ev = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14,", "const __m128i ev = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, 15,", "test_qconv", "stride-2 even bytes"),
    ("src/q_gemm.c", "nr <= 8, acc);", "nr <= 9, acc);", "test_qconv", "half-kernel threshold"),
    ("src/q_direct.c", "(long)(s % sw) * Wq * 4 + (long)(s / sw) * 4;", "(long)(s % sw) * Wq * 4 + (long)(s / sw) * 4 + 4 * (s > 5);", "test_qconv", "direct tap offset (only S>6)"),
    ("src/q_plan.c", "v0 = _mm256_sub_epi32(v0, c);", "v0 = _mm256_add_epi32(v0, c);", "test_qconv", "zero-point correction"),
    ("src/q_plan.c", "q0 = _mm256_min_epi32(_mm256_max_epi32(q0, _mm256_setzero_si256()), _mm256_set1_epi32(255));", "q0 = _mm256_min_epi32(q0, _mm256_set1_epi32(255));", "test_qconv", "u8 clamp at 0"),
    ("src/winograd.c", "x[1] = _mm256_shuffle_ps(a, c, 0xEE);", "x[1] = _mm256_shuffle_ps(a, c, 0x44);", "test_conv", "fp32 F4 input transpose"),
    ("src/im2col_gemm.c", "e = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(e), 0xD8));", "e = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(e), 0x8D));", "test_conv", "fp32 stride-2 pack"),
    ("src/kernel_6x16.c", "if (nr <= 8) kernel_half", "if (nr <= 9) kernel_half", "test_conv", "fp32 half-kernel threshold"),
    # each fix from the line-by-line review, reverted: its regression test must fail
    ("src/q_plan.c", "if ((long long)d->C * d->R * d->S > UG_Q_MAX_CRS) return 0;", "", "test_regression", "F1 revert: no int8 CRS bound"),
    ("include/ugconv.h", "return (e < 0 || d->stride_h <= 0) ? 0 : (int)(e / d->stride_h + 1);", "return (int)(e / d->stride_h + 1);", "test_regression", "F2 revert: truncating out_h"),
    ("src/q_plan.c", "__m256 s0 = _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(f0, inv), lo), hi);", "__m256 s0 = _mm256_mul_ps(f0, inv);", "test_regression", "F3 revert: no float clamp"),
    ("src/kernel_6x16.c", "v0 = _mm256_max_ps(zero, v0); v1 = _mm256_max_ps(zero, v1); } /* NaN-propagating", "v0 = _mm256_max_ps(v0, zero); v1 = _mm256_max_ps(v1, zero); } /* NaN-propagating", "test_regression", "F4 revert: NaN-dropping ReLU"),
    ("src/im2col_gemm.c", "if (!buf) continue; /* F5", "/* F5", "test_regression", "F5 revert: unchecked scratch"),
    ("src/plan.c", "if (bytes > (size_t)PTRDIFF_MAX - UG_ALIGN) return NULL;", "", "test_regression", "F8 revert: wrapping malloc"),
    ("src/plan.c", "if ((long long)d->C * d->R * d->S > INT_MAX) return 0;", "", "test_regression", "F9 revert: no C*R*S limit"),
    # F11: restore the exact pre-fix loop (time every candidate whatever execute returned).
    # Deleting only the final `if (rc) continue` is an equivalent mutant: the rc == 0 loop
    # guard alone already keeps a failing candidate's time at 1e300.
    ("src/plan.c", ["""        int rc = ug_conv_execute(p, x, y); /* warm-up */
        for (int rep = 0; rep < 3 && rc == 0; ++rep) {""", "if (rc) { ug_conv_plan_destroy(p); continue; } /* F11"],
                   ["""        int rc = ug_conv_execute(p, x, y); /* warm-up */
        for (int rep = 0; rep < 3; ++rep) {""", "/* F11"], "test_regression", "F11 revert: TUNE times failures"),
    ("src/plan.c", "&& isfinite(c[n]) && c[n] >= 0) ++n;", ") ++n;", "test_regression", "B2 revert: NaN cost file accepted"),
]

def main():
    root = os.getcwd()
    extra = sys.argv[1:]
    caught = 0
    for f, a, b, test, what in MUTANTS:
        tmp = tempfile.mkdtemp()
        dst = os.path.join(tmp, "repo")
        shutil.copytree(root, dst, ignore=shutil.ignore_patterns("build", ".git", "results"))
        path = os.path.join(dst, f)
        src = open(path).read()
        for aa, bb in (zip(a, b) if isinstance(a, list) else [(a, b)]):  # a mutant may need several edits
            assert src.count(aa) == 1, f"mutation anchor not unique/found in {f}: {aa}"
            src = src.replace(aa, bb)
        open(path, "w").write(src)
        subprocess.run(["make", "-s", "-j4", f"build/{test}"] + extra, cwd=dst, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        r = subprocess.run([f"./build/{test}"], cwd=dst, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        killed = r.returncode != 0
        caught += killed
        last = r.stdout.strip().splitlines()[-1]
        print(f"{'CAUGHT ' if killed else 'MISSED '} {what:32s} ({f}) -> {last}")
        shutil.rmtree(tmp)
    print(f"{caught}/{len(MUTANTS)} mutants caught")
    sys.exit(0 if caught == len(MUTANTS) else 1)

main()
