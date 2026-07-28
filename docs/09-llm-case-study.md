# Case study: how a shipping LLM is compiled

Everything so far was learned from small convolutions. This document applies the same
observation technique to a **production artefact**: a publicly distributed, fully compiled
1B-parameter instruction-tuned LLM built for this exact SoC (4-bit weights / 16-bit
activations).

The method is identical to the convolution work — dump the compiled container's metadata and
read the shapes:

```sh
qnn-context-binary-utility --context_binary part2_of_3.bin --json_file part2_meta.json
```

No weights, metadata dumps, or bundle files are included in this repository. Only the
structural findings are.

## Why this artefact

Two public Llama assets for this hardware exist, and only one is useful here:

| | a PyTorch quantisation-annotated export | this bundle |
|---|---|---|
| stage | **before** the vendor compiler | **after** it |
| quantisation | annotations inserted, still float | **4-bit actually applied** (6.8× smaller) |
| runnable | needs further compilation | runs on device as-is |
| useful for observing the compiler | no — never went through it | **yes** |

The point of observation is to see what the closed compiler *decided*. That requires an artefact
on the far side of it.

## Finding 1 — the model is split into three contexts

| Part | Size | Contents | Interface |
|---|---|---|---|
| 1 / 3 | 525 MB | embedding lookup | `input_ids[1,128]` → `embedding[1,128,2048]` |
| 2 / 3 | 257 MB | decoder layers 0–7 | past KV 0–7 + embedding → activation + new KV |
| 3 / 3 | 523 MB | decoder layers 8–15 + output projection | past KV 8–15 → `logits[1,1,128256]` |

The model does not fit in one context binary, so it is cut into three. **This is graph-level
tiling** — the same principle as splitting one convolution across height, applied one level up.
Splitting to fit a memory budget is scale-invariant: the budget changes, the strategy does not.

## Finding 2 — prefill and decode are separate compiled graphs

Each part contains **two** graphs:

```
[0]  prompt_ar128_cl4096_2_of_3     prefill:  128 tokens at once
[1]  token_ar1_cl4096_2_of_3        decode:   1 token at a time
```

The names decode cleanly: `prompt`/`token` = prefill/decode, `ar128`/`ar1` = tokens processed
per invocation, `cl4096` = context length, `2_of_3` = which part.

Their tensor shapes differ:

| | prefill | decode |
|---|---|---|
| KV cache in | `[8,1,64,3968]` | `[8,1,64,4095]` |
| activation out | `[1,128,2048]` | `[1,1,2048]` |
| new KV out | `[8,1,64,128]` | `[8,1,64,1]` |

**3968 + 128 = 4096** and **4095 + 1 = 4096.** The arithmetic exposes the KV cache design: input
is the existing cache, output is the newly produced entries, and the sum is exactly the context
limit. A sliding window, sized at compile time.

### Why two graphs for the same math

Because the shapes imply different bottlenecks:

- **prefill:** `[128, 2048] @ W` — a large matrix product. **Compute-bound**, matrix-engine
  limited. Tiling should optimise arithmetic throughput.
- **decode:** `[1, 2048] @ W` — a vector–matrix product. The same weights are streamed for a
  single token, so it is **memory-bound**. Scheduling should optimise transfer overlap.

One graph cannot be optimal for both, so the compiler builds two and the runtime switches
between them. The on-device sweep in [08-ondevice-measurements.md](08-ondevice-measurements.md)
confirms the split is warranted: over a 1→8 MB budget sweep, prefill latency varies by 35.8×
while decode varies by only 1.20×. Different bottleneck, different sensitivity, different graph.

Weight sharing is enabled in the bundle's configuration, so the two graphs share one copy of the
weights — two schedules, one weight set.

## Finding 3 — the KV cache cannot live on chip

```
past_key [8, 1, 64, 4095] uint8  =  8 × 64 × 4095  =  2.1 MB   per layer, keys only
  × 2 (keys + values) × 8 layers  =  33 MB          per part
  full 16-layer model              =  67 MB
```

Against a **2–8 MB** on-chip scratchpad. The KV cache is an order of magnitude too large, by
construction, and cannot be resident.

So for an LLM the strategy inverts. Convolutions are handled by *cutting the tensor small enough
to fit on chip*. The KV cache cannot be cut small enough — it only grows. It stays in DRAM and is
**streamed**, with transfers overlapped against compute (the prefetch machinery in
[07-hardware-resources.md](07-hardware-resources.md)). Only weight fragments are staged on chip.

This also explains why the KV cache is quantised **harder than activations**: 8-bit for KV
against 16-bit for activations. Compression pressure follows size, and the KV cache is the
largest thing that has to move.

## Finding 4 — the output projection dominates

Part 3 holds the same eight decoder layers as part 2 but measures **15221 µs against 7990 µs**.
The difference is the vocabulary projection: `128256 × 2048` ≈ 263 M parameters, larger than four
decoder layers combined. For a 1B model with a 128 K vocabulary, the output projection is a
first-order cost.

## Finding 5 — configuration confirms findings from elsewhere

The bundle's own backend configuration:

```json
{"devices":[{"soc_model": 87, "dsp_arch": "v81",
             "cores":[{"perf_profile":"burst","rpc_control_latency":100}]}],
 "memory": {"mem_type": "shared_buffer"},
 "context": {"weight_sharing_enabled": true}}
```

`soc_model` is specified explicitly — the vendor's own shipping configuration does the thing
whose omission caused a 3400× regression during measurement
([08-ondevice-measurements.md](08-ondevice-measurements.md)). Independent confirmation that the
field is mandatory, not decorative.

## Finding 6 — quantisation parameters, as ground truth

The metadata carries per-tensor `scale` and `offset`. These are what settled the dequantisation
formula recovered from instructions — 6 / 6 for `(q − offset) × scale`, 0 / 6 for the public
header's `(q + offset) × scale`. The full argument is in
[06-quantization.md](06-quantization.md); the relevant point here is that **a shipping model's
metadata is usable as ground truth for a formula read out of code.**

## Mapping the convolution lessons onto the LLM

| Learned from convolutions | Realised in the LLM |
|---|---|
| tiling to fit on-chip memory | 3-way context split + prefill/decode duplication |
| on-chip memory is the constraint | 67 MB KV cache → DRAM streaming, weight fragments staged |
| `(q − offset) × scale` | 4-bit weights, 16-bit activations, **8-bit KV** |
| `soc_model` is mandatory | specified in the shipping configuration |
| one matrix engine per core | 9 matmuls per layer, necessarily serial |
| compute/transfer overlap | the entire decode strategy |

The same five constraints explain a 6-op toy convolution and a 1.3 GB production LLM. That is
the useful conclusion: the compiler's decisions are not model-specific heuristics but
consequences of a small fixed set of hardware facts.
