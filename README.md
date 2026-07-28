# htp-recon — reconstructing an NPU compiler backend

A study of how a commercial NPU compiler turns a neural network into an execution plan, and
a clean-room re-implementation of the three algorithms at its core: **tiling, scheduling,
and on-chip memory allocation**.

Everything starts from one hardware constraint:

> The NPU's on-chip scratchpad memory (VTCM) is **1–8 MB**. Network tensors are larger.
> Anything that does not fit spills to DRAM, which is slow.

The compiler's job is to split operations small enough to fit, order them so memory can be
reused, and place each tensor at an address that does not collide. This repository
re-implements that pipeline in ~900 lines of C, and checks the result against the vendor
compiler's own output and against measurements on a physical device.

> **Scope and ethics:** no vendor libraries, disassembly, decompiler output, or model
> weights are included here. See [DISCLAIMER.md](DISCLAIMER.md) for the clean-room
> methodology and what is deliberately excluded.

---

## Results

| Claim | How it was verified | Outcome |
|---|---|---|
| Tile count is `ceil(H / 8)` | Predicted vs. vendor compiler output, 4 graph sizes | **4 / 4 match** |
| Minimal runtime reproduces inference | Byte-compare against vendor output | **bit-exact** (0 diff) |
| On-chip memory pressure costs performance | On-device latency sweep, VTCM 8 MB → 1 MB | **up to 2.97× slower**, monotonic |
| Channel splitting is driven by memory budget | On-device: tiles per layer as budget shrinks | 16 → 768 tiles, reproduced |
| Dequantisation is `(q − offset) × scale` | Instruction-level reading, then checked against a shipping quantised LLM | **6 / 6 parameters**; the public header's `(q + offset)` fails 0 / 6 |
| Resource-aware parallel scheduling | Slot count vs. sequential baseline | **60–67 % fewer slots** |

The dequantisation result is the sharpest one: the **published header comments the sign
backwards**, and a shipping model's own quantisation parameters settle it.

---

## The pipeline

```
                 ┌─────────────────────────────────────────────┐
   Conv          │  ① Lowering      split into tiles + relu    │
   [1,H,W,C]  ─► │  ② Scheduling    order ops across engines   │ ─►  execution plan
                 │  ③ In-place link  mark overwritable outputs │     (op order +
                 │  ④ Allocation    assign VTCM offsets        │      VTCM offsets)
                 └─────────────────────────────────────────────┘
```

These four stages correspond one-to-one to four stages of the vendor compiler, identified by
mapping its own stage timers to code addresses (see
[docs/02-compiler-pipeline.md](docs/02-compiler-pipeline.md)).

### Why the order matters

Scheduling must come before allocation, because **the schedule is what defines a tensor's
lifetime**. Two tensors whose lifetimes do not overlap can share one address:

```
Order A — finish tile 0, then tile 1        Order B — interleave both tiles
  t0  load tile0                              t0  load tile0
  t1  conv tile0                              t1  load tile1     ← both alive
  t2  load tile1  (reuses tile0's memory)     t2  conv tile0
  t3  conv tile1                              t3  conv tile1
  peak = 1 tile                               peak = 2 tiles
```

Same graph, same dependencies, **half the memory**. Picking order A is the scheduler's job;
noticing that tile 1 may reuse tile 0's address is the allocator's job. The two are one
problem wearing two hats.

---

## Hardware model

Reading the diagnostic strings of the device-side execution kernel pins down the engine
model, which the scheduler then has to respect:

| Engine | Role | Parallelism | Evidence |
|---|---|---|---|
| **HMX** | matrix (conv, matmul) | **1 per core** | `"Invalid number of HMX thread config: ... Maximum of 1 is allowed"` |
| **HVX** | vector (elementwise, norm, reduce, depthwise) | several (SoC limit) | `"Invalid number of HVX thread config: ... Must be less or equal to %d"` |
| **HLX** | special (softmax), **arch ≥ v85 only** | n/a on this target | `"Detected DSP arch: %u >= v85. Setting num_hlx_ctx"` |
| **DMA** | DRAM ↔ VTCM staging | prefetch distance | `df_dma_prefetch_distance` |

Worker pools are split by engine (`"Started %d vec workers, %d matrix workers, %d eltwise
workers"`), which is exactly why a per-resource scheduler is the right shape.

**Consequence:** matrix ops are serial. Convolutions never overlap each other — the gain
comes from overlapping a convolution with the DMA load of the *next* tile:

```
slot  | HMX     | HVX                  | DMA
t2    | -       | slicepad0,slicepad1  | -        ← two HVX workers in parallel
t4    | conv0   | -                    | vtcm1    ← compute overlaps next tile's load
t5    | conv1   | relu0                | -        ← conv0/conv1 in separate slots (HMX=1)
```

---

## Repository layout

```
src/
  minicc.c              the integrated mini compiler (stages ①–④)
  lowering.c            tile-count rule in isolation
  scheduler.c           topological (Kahn) scheduling, three tie-break policies
  scheduler_cbs.c       cost-based scheduling (memory-pressure greedy)
  scheduler_par.c       resource-aware parallel scheduling
  allocator.c           VTCM offset assignment: lifetimes + in-place + first-fit
  qnn_min_runtime.c     minimal dlopen-based inference driver (bit-exact reference)
  dequant_repro.c       dequantisation formula, reproduced
  verify_dequant_llama.c   the same formula, checked against a shipping quantised LLM
graphs/                 small text graphs used as test inputs
measurements/           on-device latency and tile-count data
docs/                   findings, methodology, and evidence
```

## Build and run

```sh
make          # builds everything into ./build; C99, no dependencies beyond libm
make demo     # runs every tool and prints every result in one pass
```

Sample output — `./build/minicc 2097152 16 64 32` (vtcm_bytes, H, W, out_channels), with
`H=16` chosen because two tiles is small enough to read in full:

```
(1) Lowering: 1 high-level Conv -> 11 low-level ops (tiles = ceil(16/8) = 2)

(2) Schedule (parallel; HMX=1 HVX=4 DMA=1): time x resource
    slot  | HMX            | HVX                        | DMA
    ------+----------------+----------------------------+------------
    t2    | -              | cv_slicepad0,cv_slicepad1  | -
    t4    | cv_conv0       | -                          | cv_vtcm1
    t5    | cv_conv1       | cv_relu0                   | -
    -> 11 sequential slots -> 8 parallel slots (27.3% fewer)
   [in-place links: 2]

(3) Allocation (VTCM 2048 KB, first-fit + lifetime reuse):
    11 tensors, 488 KB total -> 170 KB peak (318 KB saved by reuse), 0 KB spill
```

Three rows of the schedule are worth reading closely: `t2` runs two slice ops at once because
the vector engine has several workers; `t4` computes tile 0 while DMA loads tile 1; and
`cv_conv0`/`cv_conv1` land in *different* slots because there is only one matrix engine.

Other entry points:

```sh
./build/lowering 128 64 32                          # tile count for one shape
./build/scheduler graphs/branch_merge.graph 1       # tie-break policy 0..3
./build/scheduler_cbs graphs/asym.graph graphs/asym.sizes
./build/scheduler_par graphs/branch_merge.graph 4   # vector worker count
./build/allocator graphs/branch_merge.graph graphs/branch_merge.sizes 2097152
./build/verify_dequant_llama                        # dequant formula vs. a shipping model
```

`src/qnn_min_runtime.c` needs the vendor SDK headers, so it is not in the default build:
`make runtime QNN_SDK=/path/to/sdk`.

## Documentation

| Document | Contents |
|---|---|
| [01-methodology.md](docs/01-methodology.md) | Observation vs. reverse engineering; the five-way cross-check |
| [02-compiler-pipeline.md](docs/02-compiler-pipeline.md) | Vendor compiler stages, mapped to addresses |
| [03-tiling-and-crouton.md](docs/03-tiling-and-crouton.md) | Why tiles are 8 rows tall; the 2 KB hardware block |
| [04-scheduling.md](docs/04-scheduling.md) | Topological, cost-based, and parallel scheduling |
| [05-memory-allocation.md](docs/05-memory-allocation.md) | Lifetimes, in-place reuse, offset assignment |
| [06-quantization.md](docs/06-quantization.md) | The dequantisation formula and its verification |
| [07-hardware-resources.md](docs/07-hardware-resources.md) | Engines, worker pools, DMA overlap, VTCM management |
| [08-ondevice-measurements.md](docs/08-ondevice-measurements.md) | Latency sweeps and what they confirm |
| [09-llm-case-study.md](docs/09-llm-case-study.md) | A shipping quantised LLM: KV cache, prefill/decode split |

## What is deliberately not modelled

Honest gaps, all observed but not implemented:

- **Retry loop.** The vendor's allocation stage calls its allocator **twice**, and logs
  `"Ran %d internal passes"` — allocation and scheduling iterate until they converge. This
  implementation is single-pass.
- **Activation fusion.** Real hardware folds relu into the matrix engine's output stage.
  Here relu is a separate op that reuses its input's memory instead.
- **Cost-weighted slots.** A slot here is a logical parallel step, not a duration. Op cycle
  counts are not modelled, so slot reduction is not a latency prediction.
- **Parallelism vs. memory trade-off.** More parallelism means more tensors alive at once,
  which increases memory pressure. The vendor runs its parallelisation stage *after*
  allocation for this reason; the two are not co-optimised here.
