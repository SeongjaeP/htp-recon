# Hardware resource model

The compiler library on the host decides *what* to run; a separate device-side kernel actually
uses the hardware. Reading the diagnostic strings of that device kernel is what pins down the
engine model the scheduler has to respect.

(The device kernel is a DSP binary, so instruction-level disassembly is not available with
standard host tools — findings here come from diagnostic strings, symbol names, and
cross-checking against the vendor's open-source backend.)

## Engines

| Engine | Role | Parallelism | Evidence |
|---|---|---|---|
| **HMX** | matrix — conv, matmul | **1 per core** | `"Invalid number of HMX thread config: %d. Maximum of 1 is allowed."` |
| **HVX** | vector — elementwise, norm, reduce, depthwise conv | several, SoC-capped | `"Invalid number of HVX thread config: %d. Must be less or equal to %d"` |
| **HLX** | special — softmax and similar | **arch ≥ v85 only** | `"Detected DSP arch: %u >= v85. Setting num_hlx_ctx to %u"` |
| **DMA** | DRAM ↔ scratchpad staging | prefetch distance | `df_dma_prefetch_distance` |

The single most consequential line is the HMX limit. **One matrix engine per core means
convolutions and matmuls are inherently serial.** No amount of scheduling makes two
convolutions overlap.

For the target used here (a v81-class device), **HLX is inactive**: symbols exist in the v81
kernel because one codebase is built for several architectures, but the runtime gates its use
behind `arch >= v85`. The target's own metadata reports architecture 81, so this
implementation models three resources and leaves a note for future v85+ extension.

## Worker pools

```
"Started %d vec workers, %d matrix workers, %d eltwise workers"
```

Workers are pooled **per engine class** — vector, matrix, elementwise — with dispatch through a
circular message queue (`ThreadContext`, `m_threadCircularQueue`). Compiled output is likewise
split per engine: the runtime complains separately about an *HVX runlist*, an *HMX runlist* and
an *HLX runlist* when the matching thread count is missing.

This is why a per-resource scheduler is the right shape rather than an approximation: the
hardware genuinely maintains one instruction stream per engine.

Thread counts are resolved at graph build time, not at execution time
(`QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS set config can only be set at graph creation`),
with unspecified values defaulting to the SoC maximum and out-of-range values clamped.

## Automatic work splitting

```
enable_autothread
autothread_hvx_ntiles
autothread_hmx_ntiles
autothread_size_kb
```

Work is divided into `ntiles` pieces and handed to workers, with **independent tile counts for
vector and matrix engines**, and a kilobyte threshold governing how finely to cut. The
effective width is therefore `min(ntiles, threads for that engine)` — which for HMX is 1.

The precise formula relating footprint, `autothread_size_kb` and tile count is inside a stripped
function and was not recovered; only the parameter set is documented here.

## Operation → engine mapping

Derived from kernel naming families:

| Operation | Engine |
|---|---|
| conv 1×1 / N×N / dilated, matmul, fully-connected | HMX |
| relu / prelu / hardswish **following** a conv | HMX (fused into the conv's output stage) |
| depthwise conv | HVX |
| elementwise add/mul/div, layernorm, reduce, argmax, cumsum, table lookup | HVX |
| softmax | HLX on v85+, otherwise HVX |
| layout conversion (to/from the native block layout) | HVX |
| DRAM ↔ scratchpad staging, input gather, output write | DMA |

Two details worth noting:

**Depthwise convolution runs on the vector engine, not the matrix engine** — it is a per-channel
multiply, not a matrix product, so the matrix unit does not suit it. This is a good argument for
matching op names exactly rather than by substring: a substring test for "Conv" would misfile
depthwise convolution onto HMX.

**Activations are fused into the matrix engine.** Control/parameter generators exist for relu,
prelu and hardswish in 8- and 16-bit forms, meaning conv → activation is not a separate vector
op at all. This implementation keeps relu separate and recovers the memory instead via in-place
reuse (see [05-memory-allocation.md](05-memory-allocation.md)) — a different trade than the
hardware makes.

## Scratchpad management

Resource acquisition wraps the platform's compute-resource API: initialise attributes, set
scratchpad size and page parameters, request the matrix engine, then acquire. Notably,
**scratchpad and matrix engine are acquired as one request**, and the matrix engine has explicit
lock/unlock entry points — consistent with it being a single shared unit.

Configuration keys expose the layout rules:

```
vtcm_mb   vtcm_off_mb   vtcm_page_mb   vtcm_total_mb   vtcm_reserved_size   vtcm_request_timeout
```

with constraints `size + offset <= total` and alignment enforcement. Multiple graphs share one
scratchpad by offset, and cached variants allow one graph to yield it to another. Exhaustion
falls back to a DRAM spill buffer.

## Compute/transfer overlap

```
df_dma_prefetch_distance        issue DMA N steps ahead of the consumer
df_dma_move_back_distance       delay the wait, widening the overlap window
cb_max_dma_runlist_dist         keep DMA ops spaced apart in the runlist
cb_reduce_dma_ops               coalesce adjacent transfers
```

Together these are software pipelining: prefetch ahead, wait late, so transfer and compute
overlap. Supporting primitives appear as an overlapped-DMA base class and checkpoint
set/wait pairs — a producer marks a checkpoint, a consumer waits on it — and the profiler
reports `Overlap time` and `Overlap (wait) time` counters to measure the result.

Convolution has dedicated staging stages for activations, weights and bias into the scratchpad,
with both DMA-engine and vector-copy variants.

Measured on device, one convolution reported `Resources: HMX, DMA` and spent **43 % of its
cycles overlapped** with vector work — the behaviour this implementation reproduces as
`conv0 (HMX) ‖ vtcm1 (DMA)` in the same slot.

## Scheduler configuration namespaces

Two families of knobs describe the production scheduler's structure:

**`df_*` (dataflow scheduling)** — `df_parallelism_num_{hvx,hmx,hlx,chains}` sets per-engine
parallel chain counts; `df_max_tcm_ratio` and `df_tcm_group_break` break op groups when
scratchpad pressure exceeds a ratio; `df_op_sort_mode` and reorder options control grouping. The
shape it implies: **group ops, place groups on per-engine chains, break groups when memory
pressure is too high.**

**`cb_*` (cost-based placement)** — `cb_search_tcm_improving_ops`, `cb_max_ops_per_level`,
`cb_tcm_threshold`, `cb_enable_retry`, plus a conv cost model: search for ops that benefit from
scratchpad residency, level by level, with retry.

## What this implementation models

```c
CAP[R_HMX] = 1;   // hardware limit
CAP[R_HVX] = 4;   // configurable
CAP[R_DMA] = 1;
```

plus per-op resource tagging and 2-D (time × resource) slot assignment. Not modelled: cycle
costs, per-op multi-resource use (a real conv occupies HMX *and* DMA simultaneously), the
autothread tile formula, and the group-break/retry machinery.
