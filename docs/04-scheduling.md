# Scheduling

## What the scheduler is for

Lowering produces a **graph**, not an order. The graph only states dependencies:

```
        input
       /     \
  SlicePad0  SlicePad1
     │          │
   vtcm0      vtcm1
     │          │
   conv0      conv1
       \      /
        Concat
```

The NPU executes ops one at a time per engine, so this graph has to be flattened into a
sequence. Several sequences satisfy the dependencies:

```
A:  sp0, vtcm0, conv0, sp1, vtcm1, conv1, concat
B:  sp0, sp1, vtcm0, vtcm1, conv0, conv1, concat
C:  sp1, vtcm1, conv1, sp0, vtcm0, conv0, concat
```

All three are legal — `conv0` always follows `vtcm0`. They differ in **peak memory**:

- **A** finishes tile 0 before starting tile 1, so tile 1 reuses tile 0's memory → 1 tile live
- **B** opens both tiles at once → 2 tiles live

With a small scratchpad, order B may not fit and spills to DRAM. The scheduler's choice of
order is therefore a memory decision, not just an ordering one.

## Topological scheduling (`src/scheduler.c`)

The base algorithm is Kahn's: repeatedly pick a node whose dependencies are all satisfied.

```c
build_dag();        // "who produces the tensor I consume" → deps[] and indeg
compute_depth();    // depth = 1 + max(depth of dependencies)
schedule();         // repeat: pick a ready node, then release its successors
```

Three functions, and the whole of scheduling:

**`build_dag()`** translates tensor names into node dependencies. For each op, for each input
tensor, find the op that produces it — that producer is a dependency. `indeg` counts unmet
dependencies. A node whose inputs come only from outside the graph gets `indeg == 0` and is
therefore an entry point.

**`compute_depth()`** is a memoised recursion: `depth(n) = 1 + max(depth(d) for d in deps(n))`.
Memoisation matters — without it, a diamond-shaped graph re-walks shared ancestors
exponentially.

**`schedule()`** is the loop:

```c
for(;;){
  // ready = not yet scheduled AND indeg == 0
  // choose the best ready node by policy
  // mark it scheduled, append to runlist
  // for every node depending on it: indeg--
}
```

`indeg` acts as a countdown: it reaches zero exactly when a node becomes runnable. Everything
else is the tie-break policy.

### Tie-break policies

When several nodes are ready, which goes first? The policy defines the scheduler's character:

| Policy | Rule | Effect on `branch_merge` |
|---|---|---|
| 0 | lowest id (source order) | `convA, convB, convC, addAB` |
| 1 | greatest depth (critical path) | `convA, convB, **addAB**, convC` |
| 2 | least depth (breadth first) | same as 0 for this graph |
| 3 | highest id (reverse) | `convC, convB, convA, addAB` |

Policy 1 is the interesting one: it pulls `addAB` ahead of `convC`, because `addAB` sits on the
long chain toward the output. Running it early lets the tensors it consumes (`a`, `b`) be
released sooner. All four policies keep the topological order valid — the freedom is real but
bounded.

## Cost-based scheduling (`src/scheduler_cbs.c`)

The vendor's scheduler is cost-based. Its diagnostic strings name the pieces:
`build_graph_deps`, `ops_per_depth`, `select_candidate`, a cost function type, and
`por_dp_greedy_fallback` — i.e. dependency construction, depth grouping, candidate selection
by cost, greedy with a dynamic-programming fallback.

The cost modelled here is **memory pressure**: how much live memory changes if this op runs now.

```c
cost(op) = Σ size(outputs)                       // newly live
         − Σ size(inputs of which op is the last consumer)   // freed
```

Then greedily pick the ready op with the lowest cost.

### What implementing it revealed

On a graph with deliberately asymmetric tensor sizes, greedy cost scheduling produced a peak
**worse** than the simple depth policy. The reason is that the cost function is myopic: it sees
only the immediate delta, so deferring a large tensor does not help when that tensor must
become live before its consumer anyway — and meanwhile a small tensor created early stays
resident.

That is a useful negative result: it is exactly why the vendor pairs greedy with a
**DP fallback**. Pure greedy is not optimal on this objective, and the fallback exists to catch
the cases greedy loses. Implementing the naive version was the fastest way to understand why
the real one is more complicated.

## Parallel scheduling (`src/scheduler_par.c`, integrated in `minicc.c`)

The hardware has separate engines, so a single sequence understates what can run at once. The
device-side profiler shows it directly: a convolution reported `Resources: HMX, DMA` and spent
43 % of its cycles overlapped with vector ops on HVX.

So the schedule becomes two-dimensional — time × resource — with a per-resource worker limit:

```c
CAP[R_HMX] = 1;   // matrix engine: one per core (hardware limit)
CAP[R_HVX] = 4;   // vector engine: several
CAP[R_DMA] = 1;
```

Each slot admits up to `CAP[r]` ready ops per resource:

```c
for(each ready op i){
    r = resource_of(op i);
    if(nfilled[r] >= CAP[r]) continue;   // that engine is full this slot
    picked[r][nfilled[r]++] = i;
}
```

The scheduler also emits a flattened sequential runlist alongside the 2-D slots, so the
allocator downstream needs no changes.

### Results

| Graph | Sequential slots | Parallel slots | Reduction |
|---|---|---|---|
| `branch_merge` (3 convs, all HMX) | 8 | 7 | 12.5 % |
| tiled conv, H=16 (2 tiles) | 11 | 8 | 27.3 % |
| tiled conv, H=64 (8 tiles) | 35 | 14 | **60.0 %** |
| tiled conv, H=128 (16 tiles) | 67 | 22 | **67.2 %** |

Raising the HVX worker count from 1 to 4 on a graph of four independent relus cut slots from 7
to 4 (42.9 %). On `branch_merge` the same change did **nothing** — all three convolutions
contend for the single matrix engine.

### Two findings

**The 12.5 % ceiling on `branch_merge` is hardware, not a code limitation.** With one matrix
engine, three convolutions are necessarily serial.

**Tiling creates parallelism, not just memory fit.** Tiled graphs parallelise far better
(60–67 %) than the untiled one (12.5 %) because tiling spreads work across *different* engines:
`SlicePad` on HVX, staging on DMA, `ConvLayer` on HMX. That produces a pipeline —

```
t4 | conv0 (HMX) | -           | vtcm1 (DMA)   ← compute tile 0 while loading tile 1
```

— which is the real source of the gain, and a second reason the vendor splits convolutions at
all. Prefetching the next tile during the current computation is exactly the
`df_dma_prefetch_distance` / double-buffering behaviour visible in the runtime's configuration
namespace.

## Caveat

A slot is a **logical parallel step, not a duration**. Op cycle counts are not modelled, so
"67 % fewer slots" is not a latency prediction. Turning slots into time would require per-op
cost weights.
