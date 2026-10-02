# RVGPU 1.0-draft — a RISC-V GPU instruction set with a physical timing model

**Status: UNFROZEN.** This spec is deliberately not ratified. Per the riscgp
plan, the ISA must not be frozen until the P0 decision study reports and the
P1 gate selects Path A or Path B. Both datapaths are encoded here, tagged, so
that selection is a choice rather than a rewrite.

The typed opcode table in `rvgpu-table.json` is the machine-readable source of
truth. This document is its rationale. **If they disagree, the JSON is right and
this document is the bug.**

---

## 1. What this ISA is for

A RISC-V tile that does AI work. It is not a GPU in the NVIDIA sense: there is
no grid, no block scheduler, no SIMT warp, and no coherent memory. It is a
**tile with five tiny RISC-V cores that do not compute**, feeding a custom
coprocessor that does, connected by a NoC. That structure is borrowed from
Tenstorrent's Tensix, and the borrowing is deliberate — it is the only shipping
RISC-V accelerator architecture with published, detailed ISA documentation, and
its own docs show why it works:

> The RISC-V cores are 32-bit in-order single-issue, "optimized for area and
> power efficiency rather than for high performance."
>
> Loads from L1 sustain ~18.3 bits/cycle; stores ~6.4 bits/cycle. RISC-V
> cores are *strongly encouraged* to instruct other clients to access L1 on
> their behalf whenever viable.
>
> The baby RISC-V cores do not implement the "A" instruction set extension.

Those cores cannot do arithmetic and cannot move bulk data efficiently, and the
architecture wins anyway. **Keeping the core tiny is the entire efficiency
argument.** A design that puts the FLOPs in the RISC-V core has thrown that away.

## 2. The one design commitment: provenance

Every instruction carries a `domain` — the unit that executes it:

| domain | executes on | path |
|---|---|---|
| `core` | RV32I integer pipeline | A, B |
| `dma` | DMA engine / NoC client | A, B |
| `mop` | Matrix coprocessor (Matrix Unit, Unpackers, Packers, Vector Unit) | B |
| `vec` | RVV vector unit + matrix unit | A |
| `sync` | sync unit | A, B |

This is not decoration. On Path B a kernel is mostly `core` instructions that do
*no arithmetic* — they are `dm.push` calls that hand words to the coprocessor.
An IR that cannot distinguish "this instruction computes" from "this instruction
tells something else to compute" will produce a scheduler that reorders across
the only barrier that matters, and the failure mode is a wrong answer with no
crash. Provenance is what lets the validator and the cycle model reason about
resource contention at all.

## 3. Register model

| file | count | role |
|---|---|---|
| `x0..x31` | 32 | RV32I GPRs. `x0` reads zero, never written. |
| `dst0..3` | 4 | Matrix accumulators. |
| `lreg0..3` | 4 | Left matrix operand staging. |
| `srca0..1` | 2 | Right operand A. **Double buffered.** |
| `srcb0..1` | 2 | Right operand B. **Double buffered.** |
| `v0..v31` | 32 | RVV vector registers. Path A only. |
| `sem0..15` | 16 | Tile semaphores. |
| `cb0..7` | 8 | Circular-buffer descriptors. |

`dst` is *logically* four registers and *physically* one Matrix Unit. The
`mop.mma` throughput of 32 cycles is the single most important number in this
spec, because every design decision about how to use the tile falls out of it:
one Matrix Unit serves all three compute threads and all four `dst` banks, so
three threads all wanting the matrix unit contend, and the software pattern
(exactly one thread on the Matrix Unit, one on the Unpackers, one on the
Packers) is not a style preference — it is the only way to keep the one
scarce unit fed.

The double buffering of `srca`/`srcb` is what makes that pattern possible:
the Unpackers write the copy the Matrix Unit is *not* reading, and the software
flips between them.

## 4. Encoding

Free choice of encoding, with one constraint: **`dma.mcast` and `sync.*` must
exist in a form the mesh router can act on without waking a core.** In practice
that means reserved opcodes in a contiguous block the NoC intercepts, so that a
multicast costs a doorbell write rather than a core stall.

The `dm.push` / `dm.push.insn` pair is the notable one. Tenstorrent pushes
coprocessor instructions with a `sw` to a magic address, plus a `.ttinsn`
encoding that pushes a 32-bit immediate in a single instruction. We keep both
shapes: `dm.push` (word in a register, general) and `dm.push.insn` (word as an
immediate, for static pushes in a hot loop). The immediate form matters because
it lets a producer thread emit a whole constant instruction stream without
loading from SRAM first — on a core whose L1 store bandwidth is 6.4 bits/cycle,
that difference is not a micro-optimization.

## 5. Timing model

Latency is cycles from issue to the result being visible **to the issuing
core**. This distinction is load-bearing and is the single most important
asymmetry in the architecture.

| instruction | latency | throughput | unit |
|---|---|---|---|
| `nop`, `addi`, `add`, `sub`, shifts, logic | 1 | 1 | core |
| `mul` | 3 | 1 | core |
| `div`, `rem` | 34 | 1 | core |
| `lw` / `lb` / `lbu` | 8 | 7 | core |
| `sw` / `sb` | 5 | 5 | core |
| `dm.push` | 2 | 1 | core |
| `dma.issue`, `dma.mcast`, `dma.fence` | 1 | 1 | dma |
| `dma.cb.reserve`, `dma.cb.commit` | 1 | 1 | dma |
| `mop.mma` | 16 | **32** | matrix |
| `mop.elt` | 8 | 8 | matrix |
| `mop.set` | 4 | 2 | unpack |
| `mop.splat` | 2 | 2 | unpack |
| `mop.store` | 4 | 2 | pack |
| `mop.reduce` | 24 | 24 | vector |
| `stallwait` | 1 | 1 | sync |

Core memory numbers are set from the Tenstorrent measurements above on purpose:
a 7-cycle sustained load rate is what a core genuinely sustains against SRAM, and
a model that pretends otherwise will schedule kernels that do not fit in silicon.

`mop.mma` at 16/32 is the arithmetic-intensity boundary of the whole machine.
A multiply-accumulate that issues every 16 cycles has real compute; one that
issues every 32 cycles is memory-bound, and no amount of instruction-level
cleverness in the core changes that.

## 6. Memory model

There is no cache coherence, because there are no caches. SRAM is plain
scratchpad, explicitly described as "a slight misnomer" in the Tenstorrent
docs, and the tile is the atomic unit of scheduling.

Correctness therefore rests entirely on explicit synchronization:

- `sync.sem acquire` / `release` — the ordering-carrying half. **`sync.sem wait`
  alone provides no memory ordering.** A program that uses `wait` to order
  shared data is broken in a way that will pass every test that runs in a
  comfortable order.
- `sync.tile` — barrier across the three compute threads of one tile. The only
  full tile barrier.
- `sync.noc` — NoC barrier, orders all outstanding multicast and DMA.
- `dma.cb.reserve` / `dma.cb.commit` — producer/consumer on a circular buffer,
  the mechanism the three-thread pattern actually runs on.
- `stallwait` — block the core until a named unit is idle.

## 7. The async hazard, stated plainly

> Once a RISC-V core has pushed a Tensix instruction, execution of the RISC-V
> core proceeds completely asynchronously. The pushed instruction *will*
> eventually be executed, but if there are lots of instructions queued up in
> front of it, the RISC-V can get quite far ahead before the pushed instruction
> actually executes. This presents a common pitfall for programmers, if they
> expect the results of a Tensix instruction to be available to the very next
> RISC-V instruction.
>
> — `tt-isa-documentation`, Tensix Coprocessor

We inherit this hazard and therefore inherit `stallwait`. A kernel that reads
a `dst` it has just written, without a `stallwait` between, gets a **silently
wrong answer**, not a fault. There is no trap, no poison value, no bus error.

This is why `stallwait` is in the ISA and not left to a runtime library, and it
is why the validator treats it as required rather than advisory. It is also the
strongest argument for Path B: core→core debugging over a GDB-attached
RISC-V core is free here, and on a path where the bug is a silent wrong answer
that is worth more than the flops it costs.

## 8. Path A vs Path B

| | Path A (RVV + matrix extension) | Path B (RV cores + coprocessor) |
|---|---|---|
| math happens on | RVV vector cores | custom coprocessor |
| instruction that computes | `vfmacc` | `mop.mma` |
| core size | needs full vector + FP | minimal, no FPU, no `A` |
| portability | real — runs on any RVV host | none |
| fusion | **capped** | unconstrained |
| "is it a RISC-V GPU" | yes, unambiguously | arguable |

Path A's ceiling is the ratified-spec problem: **RVV 1.0 was ratified
2023-08-08, and ratified extensions are never revised.** Any extension is
purely additive and cannot change core vector semantics. An additive extension
also fights `vl`/`vtype`, which is state a pipelined async datapath does not
want to be governed by.

The honest summary: Path A is the more defensible *name*, Path B is the more
promising *machine*. That tension is the real content of the P1 gate, and it is
why nothing here is frozen.

## 9. Verification burden this creates

The combination that makes this design cheap and correct is the same combination
that makes it hard to verify:

- no caches, so no coherence bugs — but every correctness argument is now
  explicit and hand-written, so a missing `stallwait` is a silent wrong answer
- async execution, so latency is not a function of program order
- expanders, so one instruction is not one executed operation, and the cycle
  model must expand them too
- cross-thread SRAM sharing, so the tile's three threads interact

`riscgp` handles the class of bug that is most likely here — model/ISA
divergence — with the cycle model in `model/` and validation layer L2, which
compares a cycle count derived independently from this document against a
simulator run. **L2 is the highest-value cheap check in the program and the one
most likely to be skipped.**
