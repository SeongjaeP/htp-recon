// lowering.c — the tile-count rule, in isolation
//
// The vendor compiler splits one convolution into several tile convolutions. Two independent
// sources pin the rule down:
//   - observation: feeding graphs of varying height and counting the emitted ConvLayer ops
//   - the vendor's open-source MLIR backend, whose tiling pass restricts tiling to the height
//     and output-channel axes, and requires height tiles to be multiples of 8 and channel tiles
//     multiples of 32 — the dimensions of the 2 KB hardware block ({8,8,32} for INT8,
//     {8,2,32,2} for FP16, both exactly 2048 bytes)
//
// Two variants are implemented:
//   height only          num_tiles = ceil(H / 8)          <- matches the vendor 4/4 in practice
//   height and channel   num_tiles = ceil(H/8) * ceil(OC/32)
//
// Height tiling is unconditional; channel splitting only happens under memory pressure, so the
// height-only form is the one that matches unless the VTCM budget is tight.
// See docs/03-tiling-and-crouton.md.
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Hardware block ("crouton") dimensions.
#define TILE_H  8    // block height   -> tile height must be a multiple of 8
#define TILE_OC 32   // block channels -> channel tiles must be a multiple of 32

static int ceil_div(int a, int b){ return (a + b - 1) / b; }

// Height axis only.
static int lower_conv_h(int H, int W, int OC) {
    (void)W; (void)OC;
    return ceil_div(H, TILE_H);
}

// Height and channel axes. Channel splitting is memory-pressure driven, so this over-counts
// unless the budget actually forces a split.
static int lower_conv_hc(int H, int W, int OC) {
    (void)W;
    int h_tiles  = ceil_div(H, TILE_H);
    int oc_tiles = ceil_div(OC, TILE_OC);
    return h_tiles * oc_tiles;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <H> <W> <OC> [--channel]\n", argv[0]);
        fprintf(stderr, "  e.g. %s 16 16 8         # small conv, height tiling only\n", argv[0]);
        return 2;
    }
    int H = atoi(argv[1]), W = atoi(argv[2]), OC = atoi(argv[3]);
    int with_channel = (argc > 4 && strcmp(argv[4], "--channel") == 0);

    int tiles = with_channel ? lower_conv_hc(H, W, OC) : lower_conv_h(H, W, OC);

    printf("Conv output [1,%d,%d,%d]\n", H, W, OC);
    printf("  tiling: H in steps of %d", TILE_H);
    if (with_channel) printf(" + OC in steps of %d", TILE_OC);
    printf("\n");
    printf("  predicted tiles = %d\n", tiles);
    return 0;
}
