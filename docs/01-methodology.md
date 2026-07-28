# Methodology

## The distinction that shapes everything: observation vs. reverse engineering

Most of what a closed compiler does can be learned **without** looking inside it. The vendor
ships introspection tools for its own developers, and those tools expose the compiler's
input and output in full. Reading disassembly is only necessary for the residue.

| Question | Answered by | Needs disassembly? |
|---|---|---|
| What ops does the compiler emit for my graph? | vendor graph dumper (`--save_backend_op_mapping`) | no |
| In what order do the compile stages run? | the compiler's own stage log | no |
| How much memory spills to DRAM? | compile log (`spill_bytes`) | no |
| Does it get slower when memory is tight? | on-device profiler | no |
| **Which function implements a stage** | symbol table + address of the stage's log string | yes |
| **The exact arithmetic of a formula** | instruction sequence | yes |

Roughly 90 % of the findings in this repository came from the top half of that table.
Treating disassembly as the first tool rather than the last would have been slower and would
have produced weaker evidence.

## The five-way cross-check

No single method is trusted on its own. A claim is only recorded once independent methods agree.

**1. Observation.** Feed a graph to the vendor compiler, capture what it produces. This is
the *golden* output that everything else is scored against. Example: a 6-op graph came back
as 19 low-level ops, revealing that convolutions are split and that memory-staging ops are
inserted automatically.

**2. Experiment.** Sweep a documented knob and watch what changes. This is how causality was
established rather than guessed — reducing the on-chip memory budget provably changes how a
convolution is split, which no amount of static reading would have proven.

**3. Static inspection.** Symbol names give a map (`nm -D` + demangling); log format strings
say what a component believes it is doing; a short instruction sequence settles arithmetic.

**4. Cross-check against open source.** The vendor also publishes an open-source MLIR backend
for the same hardware family. Rules observed in the closed runtime were confirmed there —
for instance, the constraint that tile heights are multiples of 8 appears in open source as
an explicit validation error.

**5. Re-implementation and scoring.** The rule is written in C and its output compared to
step 1's golden data, and to on-device measurements. A rule that cannot be re-implemented
and scored is not considered understood.

## Traps encountered (and what they teach)

These cost real time, and each one generalises.

**A misconfigured experiment looks like a refutation.**
An early sweep of the memory budget appeared to *disprove* the hypothesis that memory drives
tile splitting — the output never changed. The cause was that the configuration file keyed
its settings by graph name, and the name did not match the graph, so the whole config was
silently ignored. The hypothesis was correct; the experiment was void.
→ **Before trusting a negative result, prove the experiment had any effect at all.**

**Names lie.**
One function looked like the scheduler by name. Reading which functions it calls
(`fopen`, `system`, `fscanf`) showed it writes the graph to a file and shells out to an
external script — a development path, not the production scheduler.
→ **Identify a component by what it calls, not what it is called.**

**A missing target parameter silently produces a different program.**
Compiling for the right DSP architecture but omitting the specific SoC model produced a
graph that ran **~3400× slower** on the device. Both compiles "succeeded".
→ **When a build has target parameters, verify you got the target you asked for.**

**Stale binaries.**
Twice, an edited C file appeared not to work because the previously built binary was still
being executed. Once this surfaced as a count of two billion in-place links.
→ **In a compiled language, a source edit is not a change until it is rebuilt.**

**Compiler warnings are bug reports.**
Two real defects were found by reading warnings rather than by debugging:
`non-void function does not return a value` (a missing `return`, which produced the two
billion above) and `variable set but not used` (a computed decision was never applied, so a
scheduling policy silently did nothing).

## When to stop

Depth was capped deliberately. Instruction-level analysis stopped once the *algorithm shape*
was established (for example: lifetime-sorted linear scan, bitmap lowest-offset placement,
iterative passes). Recovering byte-exact internal accounting would have required transcribing
decompiled logic — which is both the point at which returns collapse and the point at which
clean-room separation breaks.
