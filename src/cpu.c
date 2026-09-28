/* Runtime CPU facts (CPUID + sysfs). Checklist step D0: measured values beat spec sheets. */
#include <cpuid.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ug_internal.h"

static long read_cache_size(int index)
{
    char path[128], buf[64];
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/size", index);
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    long v = 0;
    if (fgets(buf, sizeof buf, f)) {
        char *end;
        v = strtol(buf, &end, 10);
        if (*end == 'K') v *= 1024;
        else if (*end == 'M') v *= 1024 * 1024;
    }
    fclose(f);
    return v;
}

static unsigned long long xgetbv0(void)
{
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return ((unsigned long long)hi << 32) | lo;
}

void ug_cpu_detect(ug_cpu_info *info)
{
    unsigned a, b, c, d;
    memset(info, 0, sizeof *info);
    const unsigned max_leaf = __get_cpuid_max(0, NULL);
    const unsigned max_ext = __get_cpuid_max(0x80000000u, NULL);
    if (max_leaf < 1) return;

    __cpuid(1, a, b, c, d);
    info->stepping = a & 0xF;
    int fam = (a >> 8) & 0xF, model = (a >> 4) & 0xF;
    if (fam == 6 || fam == 15) model |= ((a >> 16) & 0xF) << 4;
    if (fam == 15) fam += (a >> 20) & 0xFF;
    info->family = fam;
    info->model = model;
    /* AVX state must be enabled by the OS (OSXSAVE + XCR0 bits 1,2), otherwise any
     * VEX instruction faults even though CPUID advertises it (F7). */
    const int os_ymm = ((c >> 27) & 1) && (xgetbv0() & 0x6) == 0x6;
    const int os_zmm = os_ymm && (xgetbv0() & 0xE6) == 0xE6;
    info->fma = os_ymm && ((c >> 12) & 1);
    info->f16c = os_ymm && ((c >> 29) & 1);

    if (max_leaf >= 7) {
        __cpuid_count(7, 0, a, b, c, d);
        const unsigned max_sub7 = a;
        info->avx2 = os_ymm && ((b >> 5) & 1);
        info->avx512f = os_zmm && ((b >> 16) & 1);
        info->hybrid = (d >> 15) & 1;
        if (max_sub7 >= 1) {
            __cpuid_count(7, 1, a, b, c, d);
            info->avx_vnni = os_ymm && ((a >> 4) & 1);          /* CPUID.07H.01H:EAX[4]  */
            info->avx_vnni_int8 = os_ymm && ((d >> 4) & 1);     /* CPUID.07H.01H:EDX[4]  */
            info->avx_ne_convert = os_ymm && ((d >> 5) & 1);    /* CPUID.07H.01H:EDX[5]  */
        }
    }
    if (max_ext >= 0x80000004u) {
        unsigned brand[12];
        for (unsigned i = 0; i < 3; ++i)
            __cpuid(0x80000002u + i, brand[4 * i], brand[4 * i + 1], brand[4 * i + 2], brand[4 * i + 3]);
        memcpy(info->brand, brand, 48);
    }
    info->brand[48] = 0;

    info->is_arrow_lake_s = (info->family == 6 && info->model == 0xC6);
    info->l1d_bytes = read_cache_size(0);
    info->l2_bytes = read_cache_size(2);
    info->l3_bytes = read_cache_size(3);
}

char ug_cpu_core_type(void)
{
    unsigned a, b, c, d;
    if (__get_cpuid_max(0, NULL) < 0x1A) return '?';
    __cpuid_count(7, 0, a, b, c, d);
    if (!((d >> 15) & 1)) return '?';
    __cpuid_count(0x1A, 0, a, b, c, d);
    switch (a >> 24) {
    case 0x40: return 'P';
    case 0x20: return 'E';
    default: return '?';
    }
}
