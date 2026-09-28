/* F6: concurrent first use of the library from several threads. Build with
 * -fsanitize=thread (make tsan); ThreadSanitizer reports any unsynchronised
 * access to the lazily-initialised globals (machine model, ISA check, cost-model
 * file). No OpenMP parallel region runs here, so there is no libgomp noise. */
#include <pthread.h>
#include <stdio.h>
#include "ugconv.h"

static void *worker(void *arg)
{
    (void)arg;
    ug_conv_desc d = {1, 8, 12, 12, 8, 3, 3, 1, 1, 1, 1};
    static const float w[8 * 8 * 9];
    static const signed char wq[8 * 8 * 9];
    static const float s[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    for (int i = 0; i < 20; ++i) {
        (void)ug_select_algo(&d, 1);
        (void)ug_qselect_algo(&d, 1);
        ug_conv_plan_destroy(ug_conv_plan_create(&d, w, NULL, 0, UG_ALGO_IM2COL_GEMM, 1));
        ug_qconv_plan_destroy(ug_qconv_plan_create(&d, wq, s, NULL, 0, 1.f, 0, UG_QALGO_IM2COL_GEMM, 1));
    }
    return NULL;
}

int main(void)
{
    pthread_t t[8];
    for (int i = 0; i < 8; ++i) pthread_create(&t[i], NULL, worker, NULL);
    for (int i = 0; i < 8; ++i) pthread_join(t[i], NULL);
    printf("race test finished\n");
    return 0;
}
