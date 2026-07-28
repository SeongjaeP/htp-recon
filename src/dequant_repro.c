// dequant_repro.c — the dequantisation formula, reproduced from instructions
//
// The runtime's scale-offset accessor for uint8 reads is five instructions:
//
//   movzbl   (%rsi),%eax      ; q = *(uint8*)ptr
//   sub      0x4(%rdi),%eax   ; q - offset     (offset lives at struct + 4)
//   cvtsi2ss %eax,%xmm0       ; (float)(q - offset)
//   mulss    0x8(%rdi),%xmm0  ; * scale        (scale lives at struct + 8)
//   ret
//
// Read in order — load, SUBTRACT, convert, multiply — the implementation computes
//
//     real = (q - offset) * scale
//
// The public SDK header documents the opposite sign, (q + offset) * scale. The subtracting form
// is the correct one; verify_dequant_llama.c settles it against a shipping model's own
// quantisation parameters (6/6 for subtraction, 0/6 for addition).
//
// The store direction is round-to-nearest with saturation into [0, 255].
//
// See docs/06-quantization.md.
#include <stdio.h>
#include <stdint.h>
#include <math.h>

// Struct layout as implied by the instruction operands: offset at +4, scale at +8.
// The first four bytes hold something else (a type tag or vtable slot); only the two fields
// dequantisation needs are modelled here.
typedef struct {
    int32_t _pad0;   //  +0 : unused here
    int32_t offset;  //  +4 : zero point       <- matches  sub 0x4(%rdi)
    float   scale;   //  +8 : scale            <- matches  mulss 0x8(%rdi)
} ScaleOffset;

// The read path, exactly as the instructions do it.
static float dequant_u8(const ScaleOffset* so, uint8_t q) {
    return (float)((int32_t)q - so->offset) * so->scale;
}

// The mathematical inverse: q = f / scale + offset, then round and clamp.
// (The observed store path multiplies rather than divides, so it evidently caches a reciprocal
//  scale. Since the read path is what was confirmed, this implements its exact inverse.)
static uint8_t quant_u8(const ScaleOffset* so, float f) {
    float v = nearbyintf(f / so->scale + (float)so->offset);
    if (v < 0.0f)   v = 0.0f;
    if (v > 255.0f) v = 255.0f;
    return (uint8_t)v;
}

int main(void) {
    // Does the struct layout match the instruction operands?
    printf("=== struct layout (must match the instruction offsets) ===\n");
    printf("  offsetof(offset) = %zu  (instruction: +4)\n", (size_t)&((ScaleOffset*)0)->offset);
    printf("  offsetof(scale)  = %zu  (instruction: +8)\n\n", (size_t)&((ScaleOffset*)0)->scale);

    // Example parameters: symmetric uint8 quantisation, so the zero point sits dead centre.
    // 0 -> -6.4, 128 -> 0, 255 -> +6.35
    ScaleOffset so = { .offset = 128, .scale = 0.05f };

    printf("=== dequantise: real = (q - offset) * scale ===\n");
    printf("  scale=%.3f  offset (zero point)=%d\n", so.scale, so.offset);
    uint8_t samples[] = {0, 64, 128, 192, 255};
    for (int i=0;i<5;i++){
        uint8_t q = samples[i];
        float f = dequant_u8(&so, q);
        printf("  q=%3u -> (%3u - %d) * %.3f = %+.4f\n", q, q, so.offset, so.scale, f);
    }

    printf("\n=== round trip (quantise -> dequantise) ===\n");
    float vals[] = {-6.4f, 0.0f, 3.2f, 6.35f};
    for (int i=0;i<4;i++){
        float orig = vals[i];
        uint8_t q  = quant_u8(&so, orig);
        float back = dequant_u8(&so, q);
        printf("  %+.3f -> q=%3u -> %+.4f  (error %.4f)\n", orig, q, back, back-orig);
    }

    printf("\n=== the two candidate signs, side by side ===\n");
    uint8_t q = 200;
    float impl = (float)((int32_t)q - so.offset) * so.scale;  // what the code does
    float hdr  = (float)((int32_t)q + so.offset) * so.scale;  // what the header documents
    printf("  q=%u:  (q - offset) * scale = %+.4f   <- implementation\n", q, impl);
    printf("         (q + offset) * scale = %+.4f   <- header comment, sign inverted\n", hdr);
    return 0;
}
