# Test graphs

Small hand-written graphs used as inputs to `scheduler`, `scheduler_cbs`, `scheduler_par` and
`allocator`. (`minicc` builds its graph internally from the command line instead.)

## `.graph` format

One op per line, tab-separated:

```
id <TAB> name <TAB> type <TAB> IN <TAB> <input tensors, space-separated> <TAB> OUT <TAB> <output tensors>
```

Dependencies are not written down — they are recovered by matching tensor names, so whoever
produces a tensor you consume becomes a dependency. A tensor with no producer in the file is a
graph input or a weight.

## `.sizes` format

```
tensor_name <TAB> bytes
```

Only needed by the tools that reason about memory (`scheduler_cbs`, `allocator`). Tensors absent
from the file are treated as size 0, i.e. free — appropriate for external inputs and scalars.

## The graphs

**`branch_merge`** — three convolutions from a shared input, merged by two elementwise adds, with
transposes at the boundaries. Uniform 8 KB tensors. This is the graph that exposes the scheduler's
freedom: the four tie-break policies produce different orders, all valid. It is also where the
parallel scheduler hits its ceiling — three convolutions on one matrix engine can only be serial,
so parallelism saves 12.5 % and no more.

**`asym`** — deliberately asymmetric tensor sizes: one branch produces 1 MB, the other 4 KB.
Built to test whether greedy memory-cost scheduling actually beats the simple depth policy. It
does not, and that negative result is the point — see [docs/04-scheduling.md](../docs/04-scheduling.md).
