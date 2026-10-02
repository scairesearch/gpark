# Taichi Notes

Findings from studying Taichi as prior art. Recorded because the comparison is the
sharpest available test of gpark's design — and because several of the conclusions
are unflattering to gpark.

These are research notes from source reading, not benchmarks. Nothing here has been
measured. Where a claim rests on inference it is marked. Treat the whole file as
hypotheses to check, not as established results.

---

## What Taichi gets right

Worth saying first, because the criticisms below are only meaningful against a
project that is genuinely good in several places.

**Backend-neutral SIR.** Taichi compiles to a single vendor-neutral intermediate
representation and then lowers per backend. That is architecturally the right shape
and matches where gpark is heading — gpark's `Gpark.IR` is the same idea, taken far
less far.

**`full_simplify` fixpoint.** The simplification loop runs to a fixpoint rather than
a fixed number of passes. A bounded pass count leaves easy wins on the table and
makes optimisation quality depend on a tuning constant nobody understands. This is
the kind of detail that only comes from having been burned by it.

**Real end-to-end stack.** Python front end, IR, optimiser, codegen, runtime. A
research prototype that cannot actually run a program teaches you less than a
complete system with a smaller ambition.

The lesson gpark takes: the neutral-IR-plus-per-backend-lowering structure is worth
copying, and the fixpoint is worth copying. Both are on the `Gpark.IR` roadmap
without needing to be restated.

---

## Where Taichi's abstraction leaks, and gpark's does not yet

**Sub-byte types do not reach the hardware.** Taichi's type system carries
quantised types, and its optional `quant` extension does not lower to hardware
operators — it becomes arithmetic on wider integers, if it becomes anything at all.
So the type exists in the front end and not in the generated code.

This is the closest analogue to gpark's `u4` problem, and Taichi's answer is the
interesting part: **be honest that the type is a front-end fiction**. gpark reaches
the same place by construction — `u4` is a bit container with `u32` storage, and
nothing in the backend believes it is arithmetic.

*Confidence: high on the sub-byte case, based on the quant extension's
documentation. Lower on the general claim that Taichi "loses" types at lowering — it
probably handles the common cases correctly and simply declines to synthesise
hardware ops.*

**Launch overhead dominates small kernels, and Taichi is a single-stream runtime.**
This is the most important finding for gpark's positioning. Taichi's async execution
is a documented research feature rather than a shipped path.

For the workloads gpark targets — RL rollouts, agent loops, small quantised
inference steps — launch overhead is frequently the dominant cost. A runtime that
emits a stream of small kernels and cannot collapse them is leaving the single
largest available win on the table, and CUDA Graphs removes most of it.

`remote/graph_bench.cu` exists to quantify this, and its output is explicit that it
establishes a floor for a graph implementation rather than claiming a result.

**Codegen quality gap.** The front end accepts a lot that the codegen cannot
actually make fast, and the failure surfaces late, as bad numbers rather than as an
error. *Confidence: moderate — this is inference from the structure of the stack
rather than a specific measurement.*

---

## The cross-backend honesty problem

Taichi is backend-neutral on paper. Its Metal path reportedly needs zero language
extensions to reach a usable GPU path. Its AMDGPU path is, at points, closer to an
assertion than an implementation.

Both statements are true, and together they are the cautionary case.

The failure is not that one backend is worse. It is that "vendor neutral" was a
*marketing* property while the abstraction was only *genuinely* neutral on one
target. Users on the other targets find out at runtime, from numbers, with no
warning.

The structural lesson: **backend neutrality is a claim unless something enforces
it.** A backend that cannot do an operation needs to fail loudly at compile time,
not silently emit a slower fallback.

gpark's response, in `SYSTEMS.md` and `ROADMAP.md`: a cross-backend parity gate where
every backend passes the same corpus and CI fails if a capability is silently lost.
This is why `unpack_u4_f32` exists — it is a corpus entry precisely because sub-byte
bit manipulation is the operation most likely to differ between backends, and a
backend-neutral project that cannot do it should say so immediately.

*Confidence: high on the structural argument. The specific claims about Taichi's
Metal and AMDGPU maturity are from documentation reading and should be re-verified
against the current source before being relied upon.*

---

## Comparison with gpark

| | Taichi | gpark |
|---|---|---|
| IR | Vendor-neutral SIR, fixpoint simplification | Direct PTX, single target |
| Register allocation | Compiler-managed, backend-dependent | **Hand-assigned, visible** |
| Sub-byte types | Front-end types, do not reach hardware | Bit containers, explicit by construction |
| Launch overhead | Single stream; async is research | Graphs measured, `Gpark.Graph` planned |
| Backends | Metal, AMDGPU, CUDA | CUDA only, others planned |
| Guarantees | Best-effort per backend | Byte-stable goldens, parity gate planned |

Two rows are honest wins for Taichi today: it is complete, and it runs on three
vendors. gpark is one vendor, four kernels, and has never executed on a GPU.

One row is gpark's actual thesis: the register file being visible. Every other row
is a consequence of that choice rather than an independent advantage.

The most useful thing Taichi does for gpark is show what a *complete* system looks
like. The most useful thing it shows gpark to avoid is claiming a capability —
backend neutrality above all — without a mechanism that enforces it.

---

## What to check next

Stated as open questions because none of this has been measured:

- [ ] Verify the current state of Taichi's AMDGPU path against source, not docs
- [ ] Measure actual launch overhead on a real part; `graph_bench.cu` is written but
      has never run
- [ ] Establish a roofline number for the corpus kernels, so "bandwidth-bound" stops
      being an assumption
- [ ] Determine what Taichi would do with `unpack_u4_f32` — it is the most direct
      comparison available, and gpark's kernel is already written
