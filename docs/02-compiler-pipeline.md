# The vendor compiler pipeline

## Observed stages

Compiling any graph prints a stage log. The order below is the order the lines appear, which
is the order they execute:

```
Graph Preparation Initializing
Graph Optimizations
Post Graph Optimization
Graph Sequencing for Target
VTCM Allocation
Parallelization Optimization
Finalizing Graph Sequence
Completion
```

Two things are worth noting immediately:

- **Allocation comes before parallelisation.** Memory placement is treated as the constraint,
  and parallelism as an optimisation performed within whatever room is left.
- **Sequencing (ordering) comes before allocation.** This is forced: a tensor's lifetime is
  only defined once execution order exists.

## Mapping stages to code

Stage names are string constants. Finding the code address that loads each string identifies
where each stage runs:

1. `strings -t x <lib>` → the file offset of each stage name.
2. Search a disassembly listing for instructions referencing that offset.
3. The referencing instruction's address is where the stage begins.

The resulting addresses are strictly increasing:

| Stage | Code address |
|---|---|
| Post Graph Optimization | `0xe7c628` |
| Graph Sequencing for Target | `0xe7d08b` |
| VTCM Allocation | `0xe7f74b` |
| Parallelization Optimization | `0xe7fd45` |
| Finalizing Graph Sequence | `0xe80747` |

Within a single function, execution runs from low addresses to high, so **increasing address
order is execution order** — an independent confirmation of what the runtime log already
showed. Two methods, one answer.

## Structure

```
GraphPrepare::prepare(...)                    @ 0xe9cba0   ← orchestrator
├── GraphPrepare::do_prepare1(...)            @ 0xe84080
│     └── stage 1: Graph Preparation Initializing
└── GraphPrepare::do_prepare2(...)            @ 0xe7c410   ← holds stages 2–7
      ├── Post Graph Optimization
      ├── Graph Sequencing for Target
      ├── VTCM Allocation
      ├── Parallelization Optimization
      ├── Finalizing Graph Sequence
      └── Completion
(a retry path also reaches do_prepare2 from do_prepare2_retry_loop @ 0xe7bb00)
```

Stages are not separate functions. They are **scoped timer regions inside one large
function** — entering a region records a timestamp and prints `Starting stage: %s`; leaving
it prints `Completed stage: %s (%s us)`. That is why the addresses cluster so tightly.

## Inside one stage

The same address technique works recursively. Restricting a disassembly listing to the VTCM
Allocation region (`0xe7f74b`–`0xe7fd45`) and listing its calls in address order gives the
stage's internal sequence:

```
0xe7f75a  getPkgPerChannelOpsTmpMap          gather per-channel op info
0xe7f80e  perform_graph_check                validate before placing
0xe7f848  link_source_destructive_operands   ← link in-place-capable ops
0xe7f8a3  allocate_for_reschedule_grdep      ← allocation, pass 1
0xe7f987  getPkgPerChannelOpsTmpMap          re-gather
0xe7fa4a  is_flash_attention_node            special-case check
0xe7fbb6  allocate_for_reschedule_grdep      ← allocation, pass 2
0xe7fc9d  getPkgPerChannelOpsTmpMap
```

Two findings fall out of this ordering:

**In-place linking happens before allocation.** Ops whose output may overwrite their input are
marked first, so the allocator can account for the sharing. This project follows the same
order (see [05-memory-allocation.md](05-memory-allocation.md)).

**Allocation runs twice.** The function name itself — `allocate_for_reschedule` — says
allocation and scheduling are entangled, and the log line `"Ran %d internal passes of VTCM
allocation"` plus `"Cannot allocate for schedule due to vtcm limits, try CBS with lower
threshold"` complete the picture: the compiler **iterates**, relaxing a threshold until the
plan fits.

That resolves an otherwise circular question. Parallelisation needs to know how much memory is
free; allocation needs to know what runs concurrently. Neither can be strictly first — so the
real answer is not an order but a **fixed-point iteration**. This implementation is
single-pass and therefore does not reproduce that convergence.

## Correspondence to this repository

| Vendor stage | Implemented in |
|---|---|
| Post Graph Optimization | `lower_conv()` in `src/minicc.c` |
| Graph Sequencing for Target | `schedule_cbs()` — cost-based ordering |
| VTCM Allocation | `link_inplace()` + `alloc_vtcm()` |
| Parallelization Optimization | `schedule_par()` — per-resource slots |
| Finalizing Graph Sequence | not implemented (serialisation to a binary blob) |
