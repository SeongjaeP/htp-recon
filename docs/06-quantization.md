# Quantisation: reading a formula out of instructions

This is the one finding that required instruction-level reading, and the one where the
**published documentation turned out to be wrong**.

## Why dequantisation exists

The NPU computes in integers. A float weight or activation is compressed to `int8`/`uint8` (or
16-bit) to cut memory and to use cheaper arithmetic. But an integer like `192` carries no
meaning on its own — recovering the real value needs the tensor's scale and zero point. That
recovery is dequantisation, and it happens at the boundaries: final outputs, transitions
between quantised and float regions, and after integer accumulation.

## What the header says

The public SDK header defines the parameter pair and documents the formula in a comment:

```c
typedef struct { float scale; int32_t offset; } Qnn_ScaleOffset_t;
// documented as:  real = (quantised + offset) * scale
```

## What the code does

The `uint8` read path of the runtime's scale-offset accessor is five instructions:

```asm
movzbl (%rsi), %eax        ; load q (uint8, zero-extended)
sub    0x4(%rdi), %eax     ; q - offset          ← offset lives at struct+4
cvtsi2ss %eax, %xmm0       ; convert to float
mulss  0x8(%rdi), %xmm0    ; * scale             ← scale lives at struct+8
ret
```

Read in order: load, **subtract**, convert, multiply. So the implementation computes

```
real = (q − offset) × scale
```

The sign is the opposite of the documented one. The struct layout also falls out for free:
`offset` at byte 4, `scale` at byte 8.

Store direction (`write_float`) is round-to-nearest with saturation into `[0, 255]`.

## Reproduction

`src/dequant_repro.c` mirrors both the arithmetic and the observed field offsets:

```c
typedef struct {
    int32_t _pad0;   // +0
    int32_t offset;  // +4   ← matches sub 0x4(%rdi)
    float   scale;   // +8   ← matches mulss 0x8(%rdi)
} ScaleOffset;

static float dequant_u8(const ScaleOffset* so, uint8_t q){
    return (float)((int32_t)q - so->offset) * so->scale;
}
```

`offsetof` confirms 4 and 8, and a quantise → dequantise round trip returns the original values
exactly for values on the quantisation grid.

## Verification against a shipping model

Reproducing an instruction sequence proves the reading, not the interpretation. To settle it,
the formula was tested against the quantisation parameters of a **publicly distributed
quantised LLM** built for this hardware (a 1B-parameter instruction-tuned model, 4-bit weights
/ 16-bit activations). Its bundle metadata lists `scale` and `zero_point` per tensor:

| Tensor | dtype | scale | zero_point |
|---|---|---|---|
| KV cache key | uint8 | 0.1092 | 128 |
| KV cache value | uint8 | 0.01049 | 128 |
| embedding | uint16 | 1.028e-05 | 30393 |
| activation | uint16 | 0.01070 | 32261 |
| logits | uint16 | 7.154e-04 | 23218 |

### Test 1 — a zero point must dequantise to zero

By definition, `zero_point` is the integer that represents real 0. Substituting `q = zero_point`:

| Tensor | `(q − offset) × scale` | `(q + offset) × scale` |
|---|---|---|
| KV cache (q = 128) | **+0.000000** ✅ | +27.96 ❌ |
| embedding (q = 30393) | **+0.000000** ✅ | +0.62 ❌ |
| logits (q = 23218) | **+0.000000** ✅ | +33.22 ❌ |

**6 of 6 parameters** give exactly zero with subtraction. **0 of 6** with addition.

### Test 2 — the recovered range must be physically plausible

```
KV cache:  (q − offset) × scale → [−13.98, +13.87]   symmetric about 0, includes negatives ✅
           (q + offset) × scale → [+13.98, +41.83]   strictly positive, no negatives      ❌
```

Neural network activations and KV entries are distributed around zero. Only the subtracting form
can represent them at all.

Run it: `./build/verify_dequant_llama`.

## Conclusion

```
① instructions       five instructions → (q − offset) × scale
② contradiction      public header comments (q + offset)
③ reproduction       C implementation, round-trip exact
④ ground truth       shipping model's own parameters → 6/6 for ①, 0/6 for the header
```

The name `zero_point` is itself the tell: it is *the value that maps to zero*, which is the
quantity you subtract. The header comment's sign is a documentation error.

## Incidental findings on quantisation design

Reading the same metadata exposes calibration decisions:

- **KV cache uses `zero_point = 128`** — dead centre of `uint8`, i.e. **symmetric**
  quantisation, appropriate for values centred on zero.
- **Logits use 23218**, well below centre — **asymmetric**, allocating more range to positive
  values.
- **Key and value tensors of the same layer differ in scale by ~10×** (0.109 vs. 0.0105), so
  calibration is per tensor, not per layer.
- **Scale grows with depth** (layer 0: 0.109, layer 8: 0.143) — deeper layers span wider values.
- **Precision is tiered by size**: weights 4-bit, activations 16-bit, **KV cache 8-bit**. The
  largest consumer is compressed hardest.

## Type layout

The runtime's tensor types combine element type, memory layout, and residency in one name — a
quantised 8-bit tensor in the native 2 KB block layout resident in scratchpad is a distinct type
from the same tensor in DRAM. This confirms that quantisation and tiling are orthogonal axes
(see [03-tiling-and-crouton.md](03-tiling-and-crouton.md)): a 2 KB block holds 2048 INT8
elements or 1024 FP16 elements.
