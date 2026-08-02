# The byte accounting behind tile size

[docs/03](03-tiling-and-crouton.md) established two things: height tiling is unconditional at
`ceil(H/8)`, and channel splitting happens under memory pressure. It left one thing open:

> **Not modelled:** the exact footprint arithmetic that decides *when* to split channels. […] a
> naive footprint estimate lands ~4–5× below the observed splitting threshold.

This document closes that gap, and in doing so corrects the model in two ways.

## Where the estimate went wrong

Three terms were missing.

**Weights occupy on-chip memory.** The vendor's public HTP design guide states it plainly —
*"Weights are stored in TCM and take up TCM space"* — and gives the convolution fit condition as
a sum of three terms:

```
input_activation + filter + output   <=   TCM_SIZE / 2
```

The earlier estimate counted activations only. For a 3×3 convolution with 128 input channels the
filter slice alone is hundreds of kilobytes.

**Only half the budget is usable.** The condition divides by two. The same guide notes that a
footprint between `TCM_SIZE/2` and `TCM_SIZE` "might still fit but will adversely affect
performance" — so half is the planning limit, not the physical one.

**Producing 8 output rows needs more than 8 input rows.** The guide explains that a padded
convolution "typically requires three tiles of input data to be available: the data above, the
data below, and the data in the same location", and that VALID/zero padding needs two instead of
three. The halo is not incidental; it multiplies the largest term.

## The corrected condition

```c
static long footprint(Tile t, int cin, int fh, int fw, int elsize) {
    long in  = (long)(t.h + fh - 1) * (t.w + fw - 1) * cin;   // input, with halo
    long wt  = (long)fh * fw * cin * t.oc;                    // filter slice
    long out = (long)t.h * t.w * t.oc;                        // output
    return (in + wt + out) * elsize;
}
```

with the rounding rules the guide specifies: height to a multiple of 8, depth to a multiple of
32, and width to a multiple of 8 (8-bit) or 4 (16-bit).

## Correction to docs/03: there are three tiling axes, not two

The compiled graphs record the tile shape the vendor chose. Reading them out for one
convolution (H=256, W=256, 128→128 channels, fp16) across budgets:

| Budget | Output tile | Input tile read | Filter slice | conv ops |
|---|---|---|---|---|
| 8 MB | (1, **8**, 256, 128) | (1, 10, 258, 128) | (3,3,128,128) | 32 = 256/8 |
| 4 MB | (1, 8, 256, 128) | (1, 10, 258, 128) | (3,3,128,128) | 32 |
| 2 MB | (1, 8, 256, **32**) | (1, 10, 258, 128) | (3,3,128,**32**) | 128 = 32 × **4** |
| 1 MB | (1, 8, **48**, 32) | (1, 10, **50**, 128) | (3,3,128,32) | 768 = 32 × 4 × **6** |

Two things fall out.

**Width is a third axis.** Under enough pressure the compiler splits width as well, and it too
lands on a multiple of 8 (48 = 6 × 8). docs/03 described only height and channel, so the
768-op case at 1 MB was previously unexplained — it is `32 height × 4 channel × 6 width`.

**The halo is confirmed directly.** An (8, 256) output tile reads a (10, 258) input tile, and a
(8, 48) output reads (10, 50): exactly `+2` in each spatial dimension for a 3×3 kernel, which is
what `lower_conv()` already assumed.

The order of application is **height → output channel → width**. Height never moves off 8, the
hardware block height.

## Scoring

`src/vtcm_tiling.c` implements the condition and a search that halves the next axis in priority
order until the footprint fits. Scored against the tile shapes in the compiled graphs:

```
vendor tiles satisfying the documented condition : 16/16
tile shapes reproduced exactly                   : 14/16
op counts reproduced exactly                     : 14/16
```

Run it with `./build/vtcm_tiling` (add `-v` to watch the search).

**The condition itself is confirmed.** Every tile the vendor chose satisfies
`footprint <= budget/2`, and — checked separately — the unsplit tile at each budget where
splitting occurred does not. That is the criterion, and it is now modelled rather than guessed.

## What is still open

The two misses are both the width-splitting cases at 1 MB, where the vendor picks a **smaller
tile than the condition requires**:

| | vendor | largest tile satisfying `<= budget/2` |
|---|---|---|
| conv_huge @ 1 MB | width 48 | width 128 |
| scan_oc256 @ 1 MB | width 24 | width 64 |

So `footprint <= budget/2` is a **valid bound but not a tight one** — it correctly predicts
*whether* to split, but cannot by itself determine *how far*. There is an additional conservative
margin.

Double buffering is the obvious candidate, since a staged tensor held twice doubles the
activation terms, and for `conv_huge` it fits exactly: with a ×2 factor, width 48 is the largest
multiple of 8 that stays under budget/2 while width 64 does not. But the same factor is too
conservative for `scan_oc256`, where the vendor's width 24 would then exceed the limit. So the
margin is not a single constant, and is left unmodelled.

### The width choice is not a function of per-op footprint

Two hypotheses were tested against the data and both are dead.

**A constant reduction factor.** The first two width-splitting cases both reduce width by 6
(`ceil(256/6) → 48`, `ceil(128/6) → 24`), which looked like a rule. A third case found later in
the measurement artefacts — a 112-channel convolution at 1 MB — reduces by **4** instead
(`128 → 32`). Not a constant, and not a power of two either.

**A constant margin.** If the vendor simply used a tighter limit, `footprint <= k * budget/2`
for some `k < 1`, then across cases the largest accepted ratio would have to fall below the
smallest rejected one. It does not:

| case | vendor width | ratio to budget/2 | next width up | ratio |
|---|---|---|---|---|
| conv_huge | 48 | 0.43 | 96 | 0.71 |
| scan_oc256 | 24 | **0.56** ← accepted | 48 | 0.82 |
| fine_oc112 | 32 | 0.30 | 64 | **0.47** ← rejected |

The accepted 0.56 exceeds the rejected 0.47, so **no single `k` exists**. In the third case the
vendor split the width even though leaving it whole would have fit comfortably (0.47 of the
limit, and the unsplit width fits at 0.80).

So the per-op condition decides *whether* to split — 16/16 — but the width is set by something
outside it. The likeliest candidate is the allocation retry loop: the vendor's allocation entry
point is called twice and logs how many internal passes it ran, and its documented stage order
places a re-scheduling pass *after* allocation. A tile that fits on its own can still fail once
several are live at once for pipelining, and the retry would then come back with a smaller one.
That would make width a whole-graph decision rather than a per-op one, which is consistent with
everything above. Testing it requires modelling the retry loop, which is the next piece of work.

### A note on one discarded experiment

An earlier budget sweep over the same 112-channel convolution produced identical graphs at all
four budgets, which would have looked like evidence that the budget does not drive splitting. It
was an artefact: the configuration named a graph that did not exist, so it was silently ignored.
The corrected run varies as expected (16 → 16 → 64 → 256 ops from 8 MB down to 1 MB). Only the
corrected data is used here. A finer sweep places the first transition between 3 MB and 4 MB.

## Sources

- Vendor HTP network-design guide (public SDK documentation): the fit condition, the rounding
  rules, weights residing in TCM, and the two/three-tile padding behaviour.
- The vendor's open-source MLIR backend documents the shape of the search — compute the
  footprint, reduce the most profitable dimension, floor at the hardware block shape. The
  implementation here was written from that documented behaviour, not copied from its code.
- The tile shapes and op counts scored against come from this project's own compiled artefacts,
  the same measurement runs recorded in [`measurements/RESULTS.csv`](../measurements/RESULTS.csv).
