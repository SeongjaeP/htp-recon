# Memory allocation

Where the scheduler decides **when**, the allocator decides **where**: which VTCM address each
tensor occupies. This is register allocation's problem in a different unit — tensors instead of
values, byte offsets instead of registers, and the same underlying question of which lifetimes
may share storage.

## Lifetimes come from the schedule

```c
pos[node]      = position of that node in the runlist   // the time axis
tensor.birth   = pos[producer]
tensor.death   = max(pos[c]) over all consumers c
```

A tensor is live over `[birth, death]`. Note the direction of the dependency: lifetimes are
**derived from order**, which is why scheduling must run first. Change the order and every
lifetime changes with it.

## Two kinds of reuse

### 1. Disjoint lifetimes

Two tensors whose live ranges do not overlap can share one address.

```c
static int overlaps(int a, int b){
  return !(T[a].death < T[b].birth || T[b].death < T[a].birth);
}
```

`overlaps` is the whole safety condition: if false, the pair may share an offset.

### 2. In-place (destructive) reuse

An elementwise op can write its result **over its input**, using no new memory at all. The
vendor does this too — its allocation stage calls `link_source_destructive_operands`
*before* allocating, so sharing is known in advance (see
[02-compiler-pipeline.md](02-compiler-pipeline.md)). This implementation follows the same order.

Three conditions must hold:

```c
if(!is_destructive_op(g[j].type)) continue;   // ① elementwise only
if(L[in].size != L[out].size)     continue;   // ② identical size
if(L[in].death != pos[j])         continue;   // ③ this op is the last consumer
L[out].alias_of = in;                         // link
```

**① Which ops qualify.** `Relu`, elementwise add — computations of the form `y[i] = f(x[i])`,
where the read index equals the write index.

Convolution does **not** qualify: producing one output reads several inputs, so overwriting the
first output destroys data a later output still needs.

`Transpose` also does not qualify, and for a subtler reason: it *is* one-to-one, but the read
and write positions **differ**. Writing `y[1]` may clobber `x[1]`, which is still needed for
`y[2]`. (In-place transpose algorithms exist — cycle-following — but they are slow and not what
hardware does here.)

**③ is the load-bearing condition.** If another op reads the same input later, overwriting it
corrupts that read. `L[in].death == pos[j]` says "the last use of this input is right now",
which is precisely permission to destroy it.

## Placement

Tensors are processed in `birth` order — following time forward, so that "already placed"
neighbours are exactly the ones that could conflict.

```c
for each tensor i (in birth order):
    if i is an in-place alias:
        offset[i] = offset[alias_of[i]]      // no search at all
        continue
    candidates = { 0 } ∪ { offset[j] + size[j] : j overlaps i, already placed }
    sort candidates ascending
    offset[i] = first candidate that physically collides with nothing
```

Only tensors whose lifetimes **overlap** are considered as obstacles; the rest are irrelevant
by construction. Choosing the lowest non-colliding candidate is what makes reuse happen: when
a tensor dies, the low address it occupied becomes the first candidate for the next tensor.

Sizes are rounded up before placement, matching the hardware's alignment rule (2 KB for blocks
of 2 KB or more, 128 B below that):

```c
static long align_size(long n){
  long a = (n >= 2048) ? 2048 : 128;
  return (n + a - 1) & ~(a - 1);
}
```

Strictly this is **first-fit** (lowest address that fits), not best-fit (tightest gap).

## Results

`branch_merge`, 8 KB tensors, generous budget:

```
tensor        size  birth  death   offset  in-place
input_0231    8192      0      4        0
a             8192      1      3     8192
b             8192      2      3    16384
c             8192      4      5    16384          ← reuses a dead tensor's address
ab            8192      3      5     8192   a      ← in-place over `a`
abc           8192      5      6    16384   c      ← in-place over `c`
output_0231   8192      6      7        0
output        8192      7      7     8192

11 tensors, 36 KB total → 12 KB peak
```

In-place effect measured separately, by disabling it:

| H | tiles | peak without | peak with | saved | links |
|---|---|---|---|---|---|
| 16 | 2 | 44 KB | 44 KB | **0** | 2 |
| 64 | 8 | 596 KB | 554 KB | 42 KB | 8 |
| 128 | 16 | 2212 KB | 2130 KB | 82 KB | 16 |

**H = 16 links two tensors and saves nothing.** Peak is set by the single busiest moment; with
only two tiles that moment is dominated by a different tensor, so removing the relu allocations
does not lower it. In-place always reduces *total* allocation but only reduces *peak* when the
saved tensors are live at the peak. Total and peak are different objectives.

## How the vendor's allocator differs

Reading its symbols and instruction patterns, the production allocator is a different design:

| | this implementation | vendor |
|---|---|---|
| fit strategy | lowest-address first-fit | bitmap lowest-offset (first-fit family) |
| alignment | 2 KB / 128 B | **256 B** dominant |
| passes | single | **multiple internal passes** + threshold retry |
| lifetimes | explicit `[birth, death]` | sweep-line over lifetime intervals |

Both are lifetime-interval colouring; the vendor adds sweep-line event processing, finer
alignment, and iteration to convergence. So earlier spill mismatches between the two were never
a bug on either side — they are two valid strategies with different constants.

## Not modelled

- **The retry loop.** The vendor's allocator is invoked twice per stage and logs
  `"Ran %d internal passes"`. Allocation failure triggers rescheduling and retry with a relaxed
  threshold. This is single-pass, so it cannot recover from a placement that does not fit.
- **Padding and double buffering.** A naive footprint estimate lands ~4–5× below the observed
  splitting threshold; the difference is block padding, double-buffered staging and working
  space that is not tracked here.
