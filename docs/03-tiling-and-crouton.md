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

The vendor's open-source MLIR backend for the same hardware family answers it. Its
hardware-constant header defines the native tiled memory block:

```
INT8 block shape = {8, 8, 32}      →  8 × 8 × 32 × 1 byte  = 2048 B
FP16 block shape = {8, 2, 32, 2}   →  8 × 2 × 32 × 2 × 2 B = 2048 B
```

Both are exactly **2 KB**, and both have **8** as their first dimension. This block — the
vendor calls it a *crouton* — is the unit the vector and matrix engines address. Tiles must
land on block boundaries, so tile height is a multiple of 8 and tile channels a multiple of 32.

The same open-source pass enforces it explicitly, rejecting any other tile size:

- height tile size must be a multiple of 8
- output-channel tile size must be a multiple of 32

So the closed runtime and the open backend agree, because both answer to the same silicon.

A useful corollary: **INT8 and FP16 occupy the same 2 KB block at different densities** (2048
vs. 1024 elements). Quantisation's memory win is realised at the block level, and
"quantisation" and "memory layout" are independent axes — which is why type names in the
runtime combine both (an 8-bit *quantised* tensor in *crouton* layout is one named type).

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

**Not modelled:** the exact footprint arithmetic that decides *when* to split channels. The
mechanism (halve, floor at 32, driven by budget) is established; the byte-level accounting is
not, because a naive footprint estimate lands ~4–5× below the observed threshold — real
accounting includes block padding, double-buffered staging, and working space this
implementation does not track.
