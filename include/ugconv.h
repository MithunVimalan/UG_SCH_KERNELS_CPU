/*
 * ugconv - FP32 2-D convolution for Intel Core Ultra 7 270K Plus (Arrow Lake-S).
 *
 * ISA contract: AVX2 + FMA only (Arrow Lake has no AVX-512 / AMX).
 * Layouts: input NCHW, weights KCRS (OIHW), output NKPQ. groups = 1, dilation = 1.
 *
 * Three algorithm groups:
 *   Group 1  UG_ALGO_DIRECT        direct convolution, no lowering buffer
 *   Group 2  UG_ALGO_IM2COL_GEMM   im2col fused into GEMM B-panel packing + 6x16 GEMM
 *   Group 3  UG_ALGO_WINOGRAD_F4   Winograd F(4x4,3x3)  (3x3, stride 1 only)
 *            UG_ALGO_WINOGRAD_F2   Winograd F(2x2,3x3)  (3x3, stride 1 only)
 */
#ifndef UGCONV_H
#define UGCONV_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int N, C, H, W;      /* input  */
    int K, R, S;         /* filters */
    int stride_h, stride_w;
    int pad_h, pad_w;
} ug_conv_desc;

typedef enum {
    UG_ALGO_AUTO = 0,        /* analytic cost model picks */
    UG_ALGO_DIRECT = 1,      /* group 1 */
    UG_ALGO_IM2COL_GEMM = 2, /* group 2 */
    UG_ALGO_WINOGRAD_F4 = 3, /* group 3 */
    UG_ALGO_WINOGRAD_F2 = 4, /* group 3 */
    UG_ALGO_TUNE = 5         /* time every eligible algorithm once, keep the fastest */
} ug_algo;

typedef struct ug_conv_plan ug_conv_plan;

static inline int ug_out_h(const ug_conv_desc *d) { return (d->H + 2 * d->pad_h - d->R) / d->stride_h + 1; }
static inline int ug_out_w(const ug_conv_desc *d) { return (d->W + 2 * d->pad_w - d->S) / d->stride_w + 1; }

/* Group number (1,2,3) of an algorithm, 0 for AUTO/TUNE. */
int ug_algo_group(ug_algo a);
const char *ug_algo_name(ug_algo a);
int ug_algo_eligible(const ug_conv_desc *d, ug_algo a);

/* Cost-model estimate in seconds for one execute() on the target machine model.
 * Returns a negative value if the algorithm is not eligible. */
double ug_algo_cost(const ug_conv_desc *d, ug_algo a, int nthreads);
/* Pick the algorithm the cost model predicts fastest. */
ug_algo ug_select_algo(const ug_conv_desc *d, int nthreads);

/* Cost-model calibration (tools/calibrate.c). The model is
 *   t = max( sum_i feature_i * cycles_i / core_capacity(nthreads),  bytes / bandwidth )
 * Features are unit counts (MACs, kernel calls, im2col elements, tiles, ...) of the
 * schedule the executor will run. Constants can be replaced by a fitted set:
 * ug_costmodel_set(), or a file named by env UGCONV_COSTMODEL (one number per line). */
#define UG_CM_NFEAT 12
extern const char *const ug_cm_feature_names[UG_CM_NFEAT];
int ug_costmodel_features(const ug_conv_desc *d, ug_algo a, int nthreads, double f[UG_CM_NFEAT], double *bytes);
void ug_costmodel_get(double c[UG_CM_NFEAT]);
void ug_costmodel_set(const double c[UG_CM_NFEAT]);

/* weights: K*C*R*S floats (KCRS). bias: K floats or NULL. relu: fuse max(0,x).
 * nthreads <= 0 means omp_get_max_threads(). Weights are packed once here. */
ug_conv_plan *ug_conv_plan_create(const ug_conv_desc *d, const float *weights,
                                  const float *bias, int relu, ug_algo algo, int nthreads);
ug_algo ug_conv_plan_algo(const ug_conv_plan *p);
int ug_conv_execute(const ug_conv_plan *p, const float *in, float *out);
void ug_conv_plan_destroy(ug_conv_plan *p);

/* ------------------------------------------------------------------------------
 * INT8 convolution (AVX-VNNI).
 *
 * Quantisation: activations u8 asymmetric  x = in_scale * (q - in_zp),
 *               weights    s8 symmetric per output channel  w = w_scale[k] * q.
 * Accumulation is exact int32:  acc[k,p] = sum (qx - in_zp) * qw  (no rounding at all).
 * Epilogues: s32 (acc itself, for verification), f32 (acc*in_scale*w_scale[k] + bias,
 * ReLU), u8 (the f32 value requantised: clamp(nearbyint(y / out_scale) + out_zp)).
 *
 *   Group 1  UG_QALGO_DIRECT       direct, input interleaved 4 channels per pixel (u8 x s8)
 *   Group 2  UG_QALGO_IM2COL_GEMM  im2col fused into 4-byte-interleaved GEMM panels (u8 x s8)
 *   Group 3  UG_QALGO_WINOGRAD_F2  exact integer Winograd F(2x2,3x3), int16 x int16 VNNI
 * ------------------------------------------------------------------------------ */
typedef enum {
    UG_QALGO_AUTO = 0,
    UG_QALGO_DIRECT = 1,
    UG_QALGO_IM2COL_GEMM = 2,
    UG_QALGO_WINOGRAD_F2 = 3,
    UG_QALGO_TUNE = 4
} ug_qalgo;

typedef struct ug_qconv_plan ug_qconv_plan;

const char *ug_qalgo_name(ug_qalgo a);
int ug_qalgo_eligible(const ug_conv_desc *d, ug_qalgo a);
ug_qalgo ug_qselect_algo(const ug_conv_desc *d, int nthreads);
double ug_qalgo_cost(const ug_conv_desc *d, ug_qalgo a, int nthreads);
const char *ug_qvnni_backend(void);

/* INT8 cost-model calibration (tools/calibrate.c -int8); env UGCONV_QCOSTMODEL=<file>. */
#define UG_QCM_NFEAT 10
extern const char *const ug_qcm_feature_names[UG_QCM_NFEAT];
int ug_qcostmodel_features(const ug_conv_desc *d, ug_qalgo a, double f[UG_QCM_NFEAT]);
void ug_qcostmodel_get(double c[UG_QCM_NFEAT]);
void ug_qcostmodel_set(const double c[UG_QCM_NFEAT]);

/* w: K*C*R*S int8 (KCRS). w_scale: K floats. bias: K floats or NULL. in_zp in [0,255]. */
ug_qconv_plan *ug_qconv_plan_create(const ug_conv_desc *d, const signed char *w, const float *w_scale,
                                    const float *bias, int relu, float in_scale, int in_zp,
                                    ug_qalgo algo, int nthreads);
ug_qalgo ug_qconv_plan_algo(const ug_qconv_plan *p);
int ug_qconv_execute_s32(const ug_qconv_plan *p, const unsigned char *in, int *out);
int ug_qconv_execute_f32(const ug_qconv_plan *p, const unsigned char *in, float *out);
int ug_qconv_execute_u8(const ug_qconv_plan *p, const unsigned char *in, unsigned char *out,
                        float out_scale, int out_zp);
void ug_qconv_plan_destroy(ug_qconv_plan *p);

/* Helpers: symmetric per-output-channel weight quantisation (scale = max|w| / 127),
 * and u8 activation quantisation q = clamp(nearbyint(x / scale) + zp, 0, 255). */
void ug_quantize_weights_s8(const float *w, int K, int CRS, signed char *q, float *scale);
void ug_quantize_u8(const float *x, long n, float scale, int zp, unsigned char *q);

/* CPU facts detected at runtime (CPUID). */
typedef struct {
    char brand[49];
    int family, model, stepping;
    int avx2, fma, f16c, avx_vnni, avx_vnni_int8, avx_ne_convert, avx512f;
    int hybrid;               /* CPUID.07H:EDX[15] */
    int is_arrow_lake_s;      /* family 6, model 0xC6 */
    long l1d_bytes, l2_bytes, l3_bytes; /* sysfs, cpu0 */
} ug_cpu_info;

void ug_cpu_detect(ug_cpu_info *info);
/* Core type of the calling thread: 'P' (Lion Cove), 'E' (Skymont), '?' non-hybrid. */
char ug_cpu_core_type(void);

#ifdef __cplusplus
}
#endif
#endif
