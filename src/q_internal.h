#ifndef UG_Q_INTERNAL_H
#define UG_Q_INTERNAL_H
#include <stdint.h>
#include "ug_internal.h"
#include "vnni.h"

/* Output of one execute(): which epilogue, and the per-image base pointer. */
enum { UG_QOUT_S32 = 0, UG_QOUT_F32 = 1, UG_QOUT_U8 = 2 };
typedef struct {
    int kind;
    void *base;          /* image base: K*P*Q elements of the kind's type */
    float inv_out_scale; /* u8 only */
    int out_zp;          /* u8 only */
} ug_qout;

struct ug_qconv_plan {
    ug_conv_desc d;
    ug_qalgo algo;
    int relu, nthreads, P, Q, Kp;
    float in_scale;
    int in_zp;
    float *oscale;   /* [Kp] in_scale * w_scale[k] (0 for padded rows) */
    float *bias;     /* [Kp] */
    int32_t *zcorr;  /* [Kp] in_zp * sum_taps w[k]  (subtracted from u8 x s8 sums) */
    /* group 1: [Kp/6][C4/4][R][S][6][4] */
    int C4;
    int8_t *wd;
    /* group 2: [Kp/6][CRS4/4][6][4] */
    int CRS4;
    int8_t *wg;
    /* group 3: U' = 4 G g G^T, [16][Kp/6][C2/2][6][2] int16 */
    int C2;
    int16_t *ww;
};

static inline ug_qout ug_qout_image(const ug_qout *o, long n, long img_elems)
{
    ug_qout r = *o;
    long esz = o->kind == UG_QOUT_U8 ? 1 : 4;
    r.base = (char *)o->base + n * img_elems * esz;
    return r;
}

/* Epilogue for n <= 16 consecutive outputs of channel k starting at element
 * offset `off` of the image plane. `centered` = acc already excludes the zero point. */
void ug_q_epilogue16(const ug_qconv_plan *p, const ug_qout *o, int k, long off,
                     __m256i v0, __m256i v1, int n, int centered);

int ug_qdirect_prepare(ug_qconv_plan *p, const int8_t *w);
int ug_qdirect_execute(const ug_qconv_plan *p, const uint8_t *in, const ug_qout *o, long out_img_elems);
int ug_qgemm_prepare(ug_qconv_plan *p, const int8_t *w);
int ug_qgemm_execute(const ug_qconv_plan *p, const uint8_t *in, const ug_qout *o, long out_img_elems);
int ug_qwino_prepare(ug_qconv_plan *p, const int8_t *w);
int ug_qwino_execute(const ug_qconv_plan *p, const uint8_t *in, const ug_qout *o, long out_img_elems);

/* Winograd int8 exactness bound: |4*C*9*255*128| < 2^31 */
#define UG_QWINO_MAX_C 1800
#endif
