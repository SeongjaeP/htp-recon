# Disclaimer and Scope

## What this repository is

An independent study of **how an NPU compiler backend works**, carried out by observing a
commercial NPU runtime from the outside and then re-implementing the algorithms it appears
to use, in clean-room C.

The goal is educational: to understand tiling, scheduling, and on-chip memory allocation
for a real NPU, and to verify that understanding against measurable behaviour.

## What this repository is NOT

- **Not a replacement** for, or a fork of, any vendor library. The code here does not run
  neural networks on a device. It computes an execution *plan* (op order, memory offsets).
- **Not a redistribution** of vendor material. No SDK libraries, DSP skeletons, model
  weights, disassembly listings, or decompiler output are included. See `.gitignore`.
- **Not affiliated** with or endorsed by any hardware vendor.

## Clean-room methodology

Every algorithm in `src/` was written from **observed behaviour**, not from vendor source
or decompiled code:

1. **Observation** — the vendor SDK's own public tools (graph dumpers, profilers, a
   call-recording backend) were used to record what the compiler *produces* for a given
   input graph. These outputs are the ground truth this project scores against.
2. **Experiment** — documented configuration knobs (on-chip memory budget, SoC target)
   were swept to establish causality, e.g. "does reducing the memory budget change how
   the graph is split?".
3. **Static inspection** — symbol names, log format strings, and short instruction
   sequences were read to identify *which* component does what. Findings are reported as
   facts about observed behaviour (for example, a documented limit of one matrix-engine
   thread per core), never as copied code.
4. **Cross-check against open source** — the vendor also publishes an open-source MLIR
   backend for the same hardware family. Where a rule could be confirmed there, it was.
5. **Re-implementation and scoring** — the algorithms were written independently in C and
   scored against the observations from step 1, and against on-device measurements.

No decompiled code was transcribed, adapted, or used as a starting point for any file in
this repository.

## On the quoted strings

Some documents quote short diagnostic strings (a few words) observed in a vendor binary,
because they are the evidence for a factual claim about hardware behaviour — for instance
that only one matrix-engine thread exists per core. These are minimal factual citations
supporting interoperability research, not reproductions of program logic.

## Legal note

Reverse engineering for interoperability and study is permitted in many jurisdictions, but
the terms of the vendor's own licence agreement still apply to anyone who has accepted it,
and those terms may be more restrictive than local law. **Nothing here is legal advice.**
If you intend to build on this work, review the licence that applies to the SDK you use,
and keep vendor material out of any repository you publish.

## Licence

The code and documentation authored in this repository are released under the MIT licence
(see `LICENSE`). That licence covers only this repository's own content.
