# On-device measurements

Static analysis produces predictions. This document is the part where they are checked against
a physical device — which both confirmed three of them and produced one result that static
analysis could never have found.

## Setup

| | |
|---|---|
| SoC | Snapdragon 8 Elite Gen 5 class (v81 NPU architecture) |
| Toolkit | vendor AI runtime SDK 2.43 |
| Metric | per-execute accelerator time, from the runtime's own profiler |
| Statistic | median of 3 round-medians, rounds interleaved to cancel thermal drift |
| Compile | offline prepare, `dsp_arch=v81`, `soc_model` **explicitly set**, `O=3` |

The independent variable throughout is the **VTCM budget** (1 / 2 / 4 / 8 MB), set through a
graph configuration option. This is the cleanest lever available: it changes the compiler's
memory constraint without touching the network.

Raw data: [`measurements/RESULTS.csv`](../measurements/RESULTS.csv) (synthetic convolutions),
[`measurements/RESULTS_LLM.csv`](../measurements/RESULTS_LLM.csv) (a transformer block),
[`measurements/RESULTS_BUNDLE.csv`](../measurements/RESULTS_BUNDLE.csv) (a shipping model).

## Confirmation 1 — tile count is `ceil(H / 8)`

Counting `ConvLayer` ops in the compiled graph, with the budget large enough that no channel
splitting occurs:

| Graph | H | predicted by `minicc` | measured | |
|---|---|---|---|---|
| conv_med | 64 | 8 | 8 | ✅ |
| conv_big | 128 | 16 | 16 | ✅ |
| conv_huge | 256 | 32 | 32 | ✅ |
| scan_oc256 | 128 | 16 | 16 | ✅ |

**4 / 4.** The 8-row rule derived from the hardware's 2 KB block shape holds on silicon.

## Confirmation 2 — memory pressure costs performance

Same graph, shrinking budget, measured latency in microseconds:

| Graph | 8 MB | 4 MB | 2 MB | 1 MB | worst ratio | monotonic? |
|---|---|---|---|---|---|---|
| conv_big | 905 | 1152 | 1570 | 2513 | **2.78×** | yes |
| conv_huge | 10817 | 21935 | 25448 | 32105 | **2.97×** | yes |
| scan_oc256 | 7252 | 8078 | 10640 | 18326 | **2.53×** | yes |

Monotonic in every case, up to ~3× — the premise of the whole project (that a few megabytes of
on-chip memory determine performance) measured directly rather than assumed.

## Confirmation 3 — channel splitting is budget-driven

`ConvLayer` count as the budget shrinks:

| Graph | 8 MB | 4 MB | 2 MB | 1 MB |
|---|---|---|---|---|
| conv_huge | 32 | 32 | 128 | **768** |
| scan_oc256 | 16 | 32 | 128 | **768** |
| conv_med / conv_big | unchanged | | | |

Op count explodes 16 → 768 as memory tightens. This reproduces the static finding that height
tiling is unconditional while **channel splitting is a memory-pressure response**
(see [03-tiling-and-crouton.md](03-tiling-and-crouton.md)). The smaller graphs never split,
because height tiling alone already fits them.

## The counter-intuitive result

`conv_med` runs the wrong way round:

```
8 MB → 263 µs        1 MB → 191 µs        smaller budget is 27 % FASTER
```

At 8 MB the compiler emits **57 nodes**; at 1 MB it emits **94**. With plenty of memory it keeps
work in large chunks; with less it is forced to cut finer — and the finer cut **exposes more
compute/DMA overlap**. For a graph whose working set already fits comfortably, that overlap
outweighs the extra op count.

So: *more on-chip memory is not always faster.* Coarse-grained scheduling reduces the
opportunity for overlap.

No amount of static analysis produces this. It requires a real device, a real timer, and a
sweep — which is the argument for combining both methods rather than choosing one.

## Trap found while measuring: `soc_model` is not optional

Compiling with only `dsp_arch=v81` and no `soc_model` yields a **valid but completely different
graph — about 3400× slower**. The architecture flag alone is not enough; per-SoC parameters
(VTCM size, engine counts) come from the SoC identifier.

This invalidated an earlier reference result (`conv_med` at 1 MB reporting a 262144-byte spill),
which turned out to be an artefact of an unspecified SoC. On the correctly targeted device that
graph spills nothing. The reference case was replaced with `conv_big`, which spills 8 MB at a
1 MB budget and nothing at 4 MB.

This is the same class of error as an earlier one where a configuration was silently ignored
because a graph name did not match — see [01-methodology.md](01-methodology.md). Both failures
share one root cause: **a knob that appears to be set but is not, producing a confident wrong
conclusion.**

Secondary observation: identical graphs run ~4.6× slower under offline prepare than online
prepare, a warm-up/caching difference. Comparisons are therefore only ever made within one
prepare mode.

## Transformer block sweep

To check the rules on something other than plain convolutions, one **real decoder layer** was
compiled and swept: hidden 2048, 32 query heads, 8 KV heads, head dim 64, FFN 8192 —
60.8 M parameters per block, 16 blocks in the full 1B model. Both prefill (128 tokens) and
decode (1 token, 128 cached) regimes, at two precisions.

| Precision | Regime | 1 MB | 2 MB | 4 MB | 8 MB | worst ratio |
|---|---|---|---|---|---|---|
| fp16 | prefill | 197224 | 13830 | 5686 | 5512 | **35.8×** |
| fp16 | decode | 17232 | 14939 | 14789 | 14402 | 1.20× |
| w4a16 | prefill | 3825 | 2324 | 1041 | 985 | 3.88× |
| w4a16 | decode | 1417 | 1012 | 863 | 827 | 1.71× |

Three things stand out.

**Memory pressure hurts far more here than on convolutions.** The worst synthetic case was
2.97×; fp16 prefill at 1 MB is **35.8×** — with 7.4 MB spilled to DRAM. A transformer block's
working set is much larger than a single convolution's, so a 1 MB budget is catastrophic rather
than merely tight.

**Prefill is memory-bound, decode is not.** Prefill spans 35.8× across the sweep, decode only
1.20×. Prefill processes 128 tokens at once — large activation matrices, matrix-engine bound.
Decode processes one token — tiny activations, dominated by streaming weights and KV cache from
DRAM, which the on-chip budget does not change. **The two regimes have different bottlenecks**,
which is exactly why the shipping model compiles them as two separate graphs
(see [09-llm-case-study.md](09-llm-case-study.md)).

**Quantisation shrinks the sensitivity, not just the size.** Going fp16 → w4a16 cuts the context
from 122 MB to 31 MB (3.9×) and prefill latency by 5.6×, but it also collapses the spread from
35.8× to 3.88×. A smaller working set fits the budget in more configurations, so the same
hardware behaves far more forgivingly. Quantisation buys robustness as well as speed.

## Whole-model check

Finally, a **pre-built shipping bundle** of a 1B instruction-tuned model for this SoC was run
as-is, without recompilation, to see whether the per-block numbers add up:

| Part | Contents | Context | Decode latency |
|---|---|---|---|
| 1 / 3 | embedding lookup | 525 MB | 60 µs |
| 2 / 3 | decoder layers 0–7 | 257 MB | 7990 µs |
| 3 / 3 | decoder layers 8–15 + output projection | 523 MB | 15221 µs |
| **total** | one full decode step | | **23271 µs ≈ 43 tok/s** |

Cross-check: this project's own w4a16 block measured 827 µs; × 8 layers = 6616 µs against the
bundle's 7990 µs for part 2 — **17 % under**, which is close enough to say the per-block
measurement is sound (the gap is the parts of a real layer the isolated block omits).

Part 3 costs nearly twice part 2 for the same eight layers, because it also carries the output
projection: a 128256 × 2048 matrix, ~263 M parameters — larger than four decoder layers put
together. In a small LLM the vocabulary projection is a first-order cost, not a footnote.

## Summary

```
static analysis  →  three rules  →  device measurement  →  all three confirmed
                                                        →  plus one rule static analysis
                                                           could not have produced
```

The loop is closed in both directions: static analysis makes the predictions testable, and
measurement finds the effects (overlap, prepare-mode differences, the `soc_model` trap) that
compiled artefacts alone do not reveal.
