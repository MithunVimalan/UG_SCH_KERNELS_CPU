/* Benchmark layer table: VGG-16 and ResNet-50 (v1.5) conv layers, batch 1.
 * mult = how many times the shape occurs in the network. */
#ifndef UG_BENCH_LAYERS_H
#define UG_BENCH_LAYERS_H
#include "ugconv.h"

typedef struct { const char *net, *name; int mult; ug_conv_desc d; } layer;

#define L(net, name, mult, C, HW, K, R, s, p) {net, name, mult, {1, C, HW, HW, K, R, R, s, s, p, p}}
static const layer layers[] __attribute__((unused)) = {
    L("vgg16", "conv1_1", 1, 3, 224, 64, 3, 1, 1),
    L("vgg16", "conv1_2", 1, 64, 224, 64, 3, 1, 1),
    L("vgg16", "conv2_1", 1, 64, 112, 128, 3, 1, 1),
    L("vgg16", "conv2_2", 1, 128, 112, 128, 3, 1, 1),
    L("vgg16", "conv3_1", 1, 128, 56, 256, 3, 1, 1),
    L("vgg16", "conv3_2", 2, 256, 56, 256, 3, 1, 1),
    L("vgg16", "conv4_1", 1, 256, 28, 512, 3, 1, 1),
    L("vgg16", "conv4_2", 2, 512, 28, 512, 3, 1, 1),
    L("vgg16", "conv5_x", 3, 512, 14, 512, 3, 1, 1),
    L("resnet50", "stem7x7s2", 1, 3, 224, 64, 7, 2, 3),
    L("resnet50", "l1_1x1_64", 3, 64, 56, 64, 1, 1, 0),   /* first block reads 64 ch, others 256 */
    L("resnet50", "l1_3x3", 3, 64, 56, 64, 3, 1, 1),
    L("resnet50", "l1_1x1_up", 4, 64, 56, 256, 1, 1, 0),   /* 3 expand + downsample */
    L("resnet50", "l2_1x1_red_s1", 1, 256, 56, 128, 1, 1, 0),
    L("resnet50", "l2_3x3_s2", 1, 128, 56, 128, 3, 2, 1),
    L("resnet50", "l2_1x1_up", 4, 128, 28, 512, 1, 1, 0),
    L("resnet50", "l2_ds_1x1_s2", 1, 256, 56, 512, 1, 2, 0),
    L("resnet50", "l2_1x1_red", 3, 512, 28, 128, 1, 1, 0),
    L("resnet50", "l2_3x3", 3, 128, 28, 128, 3, 1, 1),
    L("resnet50", "l3_1x1_red_s1", 1, 512, 28, 256, 1, 1, 0),
    L("resnet50", "l3_3x3_s2", 1, 256, 28, 256, 3, 2, 1),
    L("resnet50", "l3_1x1_up", 6, 256, 14, 1024, 1, 1, 0),
    L("resnet50", "l3_ds_1x1_s2", 1, 512, 28, 1024, 1, 2, 0),
    L("resnet50", "l3_1x1_red", 5, 1024, 14, 256, 1, 1, 0),
    L("resnet50", "l3_3x3", 5, 256, 14, 256, 3, 1, 1),
    L("resnet50", "l4_1x1_red_s1", 1, 1024, 14, 512, 1, 1, 0),
    L("resnet50", "l4_3x3_s2", 1, 512, 14, 512, 3, 2, 1),
    L("resnet50", "l4_1x1_up", 3, 512, 7, 2048, 1, 1, 0),
    L("resnet50", "l4_ds_1x1_s2", 1, 1024, 14, 2048, 1, 2, 0),
    L("resnet50", "l4_1x1_red", 2, 2048, 7, 512, 1, 1, 0),
    L("resnet50", "l4_3x3", 2, 512, 7, 512, 3, 1, 1),
};
#endif
