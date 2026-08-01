# Tiling: why tiles are eight rows tall

## The observation

Feeding a small graph of 6 ops to the vendor compiler returns **19 ops**. A single
convolution comes back as:

```
InputSlice → Transpose → ( SlicePad → flat_from_vtcm → ConvLayer ) × N → Concat
```

Two distinct things happened:

- **Splitting.** One convolution became several `ConvLayer` ops.
- **Insertion.** Ops that were not in the input appeared: slicing, on-chip staging, and a
  concatenation to reassemble the result.

The inserted ops are the interesting part. Their names (`flat_from_vtcm`) point at the
on-chip scratchpad: the compiler cuts the tensor into pieces, moves a piece into fast
memory, computes it there, and stitches the results back together.

## The rule

Sweeping the input size and counting the emitted `ConvLayer` ops:

| Output height H | Tile output height | Tiles emitted |
|---|---|---|
| 16 | 8 | 2 |
| 64 | 8 | 8 |
| 128 | 8 | 16 |
| 256 | 8 | 32 (× channel splits) |

The tile height is **always 8**, regardless of tensor size. So:

```
tiles = ceil(H / 8)
```

The first hypothesis — "it splits because the tensor exceeds on-chip memory" — is wrong. The
smallest case is 8 KB, which fits in any budget, and it still splits. Height-8 tiling is
unconditional.

## Why 8

Three sources agree, and together they pin it down exactly.

**The vendor's own layout documentation** defines the block as a *chunked memory layout*:

```
R4CroutonLayout = ChunkedMemoryLayout<4, 0,0, 1,0, 2,0, 3,0, 1,8, 2,8, 3,32>
                                            ordering          chunk shape
```

Read the `(dimension, size)` pairs right to left: `3,32` takes 32 elements of the channel
dimension, `2,8` takes 8 chunks along width, `1,8` takes 8 along height. So the chunk is
**1 × 8 × 8 × 32** — and the documentation adds that anything smaller is *padded up* to it.

**Its open-source MLIR backend** carries the same constants:

```
INT8_CROUTON_SHAPE = {8, 8, 32}
F16_CROUTON_SHAPE  = {8, 2, 32, 2}
```

The 16-bit entry looks like it has an extra dimension, but the backend's own packing code
multiplies entries `[1]` and `[3]` together to form the width — `2 × 2 = 4`. So the two shapes
are really `8 × 8 × 32` and `8 × 4 × 32`:

| | height | width | channels | element | total |
|---|---|---|---|---|---|
| INT8 | 8 | **8** | 32 | 1 B | **2048 B** |
| FP16 | 8 | **4** | 32 | 2 B | **2048 B** |

Both fill exactly **2 KB**. The element doubles in size, so the width halves — the block is a
fixed *byte* size, not a fixed element count.

**The design guide states the same fact a third way**, as a rounding rule: *"activation widths
need to be rounded up to the nearest multiple of 8 (uint8) / 4 (uint16)"*, depths to 32, heights
to 8. That rule is not a separate constraint — **it is the block shape**, restated.

This block is what the vector and matrix engines address, so tiles land on its boundaries: height
a multiple of 8, channels a multiple of 32, width a multiple of 8 or 4 by precision.

The same open-source pass enforces it explicitly, rejecting any other tile size:

- height tile size must be a multiple of 8
- output-channel tile size must be a multiple of 32

So the closed runtime and the open backend agree, because both answer to the same silicon.

A useful corollary: **INT8 and FP16 occupy the same 2 KB block at different densities** (2048
vs. 1024 elements). Quantisation's memory win is realised at the block level, and
"quantisation" and "memory layout" are independent axes — which is why type names in the
runtime combine both (an 8-bit *quantised* tensor in *crouton* layout is one named type).

### Confirmed against measured tensors

The tile shapes recorded in compiled graphs land on block boundaries exactly as predicted. An
`(1, 8, 256, 128)` fp16 output tile is `1 × (8/8) × (256/4) × (128/32)` = **256 blocks**, i.e.
`256 × 2 KB = 512 KB`, which matches its byte size exactly. The smallest observed tile,
`(1, 8, 48, 32)`, is **12 blocks**.

**Not everything is crouton-resident, though.** The halo input tile `(1, 10, 258, 128)` is a
multiple of neither 8 nor 4, so it is not in block layout at all. Charging it crouton padding —
rounding to `(1, 16, 260, 128)`, a 61 % increase — makes the footprint model *worse*
(12/16 against 14/16, see [10-vtcm-footprint.md](10-vtcm-footprint.md)). The staging tile is
**flat**, which the op inventory confirms: `flat_to_vtcm` and `flat_from_vtcm` exist alongside a
separate `ForceFormat_Crouton`.

## Halo

Tile inputs are larger than tile outputs:

```
tile input  [1, 10, 18, C]
tile output [1,  8, 16, C]
```

Producing 8 output rows from a 3×3 kernel requires 10 input rows — the extra rows are the
overlap with neighbouring tiles. This is why a plain slice is not enough and a dedicated
`SlicePad` op exists: tiles are cut **overlapping**, not disjointly.

## Channel splitting is conditional

Unlike height, channel splitting only happens under memory pressure. Holding H fixed at 128
and sweeping output channels:

| Output channels | Channel tile | Split? |
|---|---|---|
| 64 | 64 | no |
| 96 | 96 | no |
| 128 | 64 | **yes** (÷2) |
| 192 | 32 | yes (÷6) |
| 256 | 32 | yes (÷8) |

Splitting begins between 112 and 120 channels, halves first, and bottoms out at 32 — the
crouton channel dimension, below which it cannot go.

Directly manipulating the memory budget confirms the cause. For a fixed graph:

| VTCM budget | Channel tile | Tiles |
|---|---|---|
| 1 MB | 32 | 256 |
| 2 MB | 32 | 64 |
| 4 MB | 112 (whole) | 16 |
| 8 MB | 112 (whole) | 16 |

Smaller budget, finer splitting. The open-source backend shows the shape of the decision:
compute the operation's memory footprint, and if it exceeds the budget, reduce a dimension by
the next power of two, repeating until it fits, with the crouton size as the floor.

## What this implementation does

`src/lowering.c` and `lower_conv()` in `src/minicc.c`:

```c
#define TILE_H  8    // crouton height
#define TILE_OC 32   // crouton channels
tiles = ceil_div(H, TILE_H);
```

and per tile emit `SlicePad` (with halo) → `flat_from_vtcm` → `ConvLayer` → `Relu`, then one
`Concat`. Predicted tile counts match the vendor compiler for H = 16, 64, 128, 256 and match
counts measured on a device.

## Two corrections, established later

**There is a third tiling axis.** Under enough memory pressure the compiler splits **width** as
well, also to a multiple of 8. The order is height → output channel → width. This is why the
tightest budget produces 768 convolutions where height and channel alone predict 128:
`32 height × 4 channel × 6 width`.

**The footprint arithmetic is now modelled.** The estimate that landed ~4–5× low was missing
three terms: weights occupy on-chip memory too, only half the budget is usable, and the halo
multiplies the input term. The corrected condition reproduces the vendor's split/no-split
decision 16/16 and its exact tile shape 14/16.

Both are derived in [10-vtcm-footprint.md](10-vtcm-footprint.md), with `src/vtcm_tiling.c` as the
implementation. What remains open there is *how far* the search goes once it decides to split —
the condition is a valid bound but not a tight one.
