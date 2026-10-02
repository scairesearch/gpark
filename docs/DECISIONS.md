# Decisions

Architecture decisions, with the reasoning and what was given up. Reverse-engineerable
commits are worse than useless; this is the record of *why*.

---

## Direct PTX, no vendor libraries

**Decided.** gpark emits PTX and nothing else. No nvcc in the build path, no
`cublasLt`, no cuDNN.

**Why.** The four things that make a GPU kernel slow — bandwidth, latency,
occupancy, launch overhead — are all decided at instruction level. Any layer that
hides instructions eventually hides the decision. Since registers are numbered by
hand, "this kernel needs 8 live output offsets" is a visible fact rather than
something to be inferred from a profiler.

**Gave up.** Tensor Cores, vendor heuristics, mature library kernels, and the
convenience of letting a compiler handle the boring parts. gpark is slower to write
kernels in than Triton.

**Reversible?** No, and it should not be. This is the project.

---

## No SSA, hand-assigned registers in v0.1

**Decided.** `Gpark.IR` is straight-line blocks with explicit register identifiers.
No SSA, no phi nodes, no allocator.

**Why.** The interesting variable must not be implicit. When a bandwidth kernel is
at 40% of roofline, the cause is nearly always how many loads are in flight per
thread, which is decided by the register assignment. An allocator optimises for
something else — usually occupancy — and hands you a correct kernel you cannot
reason about.

It also keeps `Gpark.Validate` tractable. Without SSA there is a single register
state per type, so "read before write" is a check rather than a dataflow fixpoint.

**Gave up.** Loops. `unpack_u4_f32` unrolls eight shift/mask/convert/store blocks
because a loop needs a phi node. That is 8 live output addresses instead of 2, which
is a real cost — and a real demonstration of the tradeoff the allocator will have to
solve.

**Reversible?** Yes, and planned: `Gpark.Mid` introduces SSA with phi nodes, and the
allocator replaces hand assignment. The IR is deliberately positioned to be lowered
*into* it.

---

## Hand-rolled kernel IR instead of textual templating

**Decided.** Kernels are built from IR constructors in both languages.

**Why.** Textual templating pushes parsing, validation and lowering into string
handling, which is where a wrong answer becomes a plausible answer. Structuring the
IR makes `cvt.rn.f32.u32`'s two-type signature a type error instead of a rendering
detail, and lets the validator check the same structure the emitter walks.

**Gave up.** Concision. A kernel takes ~150 lines instead of ~40. This shows up
constantly and is the correct trade.

---

## Sub-byte types are packed bit containers, not arithmetic types

**Decided.** `s2 u2 s4 u4` are logical types with *no* direct PTX spelling, which is
what keeps the backend from believing they are arithmetic. They exist as the element
of a `Packed` value, whose *container* (`:b16`/`:b32`/`:b64`) is what allocates.

**Why.** PTX has no 4-bit float arithmetic, and 4-bit packed *integer* ops exist but
cover neither floats nor most of the operations a quantised kernel needs.
Pretending `u4` is a first-class arithmetic type would mean inventing semantics the
hardware does not have.

Splitting the two questions also stops the type system lying about hardware. A `u4`
occupies a whole `.b32` register, and anything that treats it as occupying 4 bits is
wrong about occupancy, register pressure, and memory traffic — which is exactly the
information gpark exists to expose.

**Gave up.** Convenience. `unpack_u4_f32` writes eight shift/mask/convert/store
sequences that a hypothetical packed instruction could halve, and does so on `:u32`
registers rather than through `Packed`, which is a known gap rather than a design.

**Note.** The container/element split (`b32` vs `f32`) came out of this and now
applies to every type: 30 types, 8 containers.

---

## Two independent implementations, byte-for-byte

**Decided.** Elixir and Python each build every golden from the same JSON spec. The
tests assert identical bytes and compare hashed opcode tables.

**Why.** Drift between language backends is silent. It surfaces as "works in Elixir"
and gets debugged as a language problem rather than a backend problem. Two
implementations plus a shared corpus is the cheapest available detection.

**Gave up.** Every emitter change is two changes. Every type bug is two bugs. This
cost is paid continuously.

**Verification.** The `unpack_u4_f32` bounds bug is a live argument for the setup —
and also a limit of it, since both implementations agreed on the *wrong* answer and
byte-parity passed. Parity proves two implementations match, not that either is
correct. That gap is what `ptx_test`'s direct property assertions exist to cover.

---

## Byte-stable goldens, not just equivalent ones

**Decided.** PTX output is deterministic to the byte. No map iteration in emit
order, no timestamps, no environment-dependent formatting.

**Why.** The goldens are meant to be diffed against real `nvcc` output to track
divergence from the real compiler. That only works on stable input. "Stable modulo
whitespace" is not a contract anyone can rely on.

**Gave up.** Freedom to format for readability. The emitter is boring on purpose.

---

## The corpus is four kernels, and the rule for adding more

**Decided.** A kernel joins `corpus/` only if it forces something new into the
backend: a new opcode shape, addressing form, register class, or validator rule.

**Why.** A corpus that grows by accretion stops being evidence. Four kernels, four
capability families — addressing, arithmetic, synchronisation, sub-byte — is enough
to make the backend non-trivial and few enough that each test has a stated purpose.

**Gave up.** Immediate breadth, and the comfort of a bigger corpus looking more
serious.

**Enforced.** `ptx_test.exs` asserts the exact kernel list, so the corpus cannot
quietly shrink.

---

## Bandwidth-bound corpus kernels as controls

**Decided.** Every current kernel is bandwidth-bound with near-zero arithmetic
intensity.

**Why.** A bandwidth-bound kernel has one expected answer. If gpark misses roofline
on a saxpy, the cause is addressing or registers — not a scheduling subtlety — and
the measurement isolates the thing worth debugging. It also means each kernel is a
*control* for the next.

**Gave up.** Nothing, yet, and it is why no kernel needs shared memory so far.

---

## Spills are a hard failure, not a warning

**Decided.** `ptxas_check.sh` fails the build on any register spill.

**Why.** gpark assigns registers by hand. A spill means the hand assignment could
not hold the working set, and ptxas silently papered over it by spilling to local
memory — turning a bandwidth-bound kernel into a bandwidth-bound kernel plus a
`local` round trip.

Accepting spills would mean the premise of the whole approach was wrong and nothing
said so. The number that matters is the one that should stop you.

**Gave up.** The ability to accept a slightly-spilling kernel and move on. That
seems fine until the regression is six weeks old and nobody remembers approving it.

---

## Driver API in the execution harness

**Decided.** `exec_harness.cu` JITs goldens through the CUDA driver API and reads
`.ptx` at runtime.

**Why.** Routing the harness through nvcc would let the toolchain normalise the
input, so the harness would test nvcc's opinion of the PTX rather than the PTX. This
is the one place where "no vendor library in the path" needs to apply to the test
infrastructure too, because a harness that normalises its input cannot catch an
emitter that produces something odd.

**Also.** The `.ptx` files are read from `corpus/golden/` at runtime, so
`make corpus && make remote-build` is the whole loop, with no build step that could
quietly regenerate the thing under test.

**Consequence.** Both harnesses must be syntax-checked without CUDA, or they are the
part that fails on the remote machine. That check found three real compile errors in
the first draft.

---

## Tolerances only where hardware promises none

**Decided.** `exec_harness` bit-compares the integer and unpack kernels and uses a
tolerance for `saxpy_f32`.

**Why.** `fma.rn.f32` rounds once; the CPU expression rounds twice or contracts to
one. Both are correct and they differ in the last ulp. Demanding bit equality would
assert something PTX does not promise — and the harness would fail on correct code.

**Rule.** Exact where the operation is exact, tolerance where it is not. Tolerances
applied uniformly hide real bugs, so the question is always whether the hardware
*promises* exactness for that specific operation.

---

## Assert properties, not just bytes

**Decided.** Beyond byte-for-byte goldens, tests assert properties directly — most
notably that no global store precedes a bounds guard.

**Why.** `unpack_u4_f32` shipped with its bounds guard emitted *after* the work it
guarded, so every out-of-range lane performed a 32-byte out-of-bounds write. The
golden was self-consistent. The validator passed. Every test was green. Both were
handed a program that was valid PTX and merely not the program intended.

Byte comparison detects *change*, not *incorrectness*. That is a limitation it has
by definition, and no amount of golden discipline removes it.

**Cost.** A property assertion that a valid program violates — "no load before the
guard" — is wrong for `reduce_sum_f32`, which must load every lane and relies on a
documented warp-multiple `n`. The test encodes that exception with its reasoning
rather than asserting a rule that does not hold.

**Discipline.** A new assertion is checked against the code it is supposed to catch
before being trusted. Verified by reverting the fix and watching the test fail.

---

## AGPL-3.0

**Decided.** AGPL-3.0, and the network clause is intentional: this is meant to stay
usable as a service.

**Resolved.** `LICENSE` now carries the verbatim AGPL-3.0 text (SHA-256
`0d96a4ff…`, retrieved from gnu.org and recorded in `LICENSING.md` with a command to
re-verify). It was flagged rather than filled in while it was still a gap, because
fetching the canonical text is a checkable step and a half-attribution is worse than
an acknowledged absence. The licence body is unmodified; provenance lives in a
separate file so the text stays byte-identical to the FSF's.

---

## Two languages, not three

**Decided.** Elixir and Python only. No third implementation yet.

**Why.** Each additional backend multiplies emitter maintenance. Two already exist
purely to catch drift; a third costs more than it catches until the corpus is larger
and the emitter more stable.

**Revisit when** the corpus passes roughly a dozen kernels or a Metal backend is
serious.

---

## Not using shared memory yet

**Decided.** No kernel uses shared memory, though the emitter validates it.

**Why.** No current kernel needs it. `unpack_u4_f32` is register-limited rather than
tiling-limited, and the fix there is the register allocator, not shared memory. Adding
the feature to have it is how backends accumulate surface they cannot justify.

**Trigger.** A kernel that cannot be written without it. Recorded in `ROADMAP.md` with
the two candidates already identified.

---

## Adopting Taichi's `full_simplify` fixpoint

**Decided.** `Gpark.Opt.Simplify` iterates rules until nothing fires, rather than
running a fixed number of passes.

**Why.** A bounded pass count makes optimisation quality depend on a tuning constant
nobody understands, and it fails silently: the output is still correct, just less
simplified, so nothing anywhere tells you to raise the bound. The correctness shows
up as slow PTX on hardware nobody is watching. A fixpoint either converges or a rule
is wrong, and `simplify/2` raises on non-convergence rather than returning a
half-simplified kernel.

**Gave up.** The ability to bound compile time in the pathological case. Kernels this
size cannot hit the guard; if one ever does, the rule is oscillating and that is a bug
worth surfacing rather than papering over.

**Rule set in v0.1 is deliberately two rules:** unreachable blocks, and dead
side-effect-free instructions. See the next two entries for why the set is not larger.

---

## Dead loads are not dead code

**Decided.** `Gpark.Opt.Simplify` never removes a `ld`, even when its destination is
never read. Every normal compiler does remove them.

**Why.** A load whose result is unused can still fault, and removing it silently
deletes the evidence of an out-of-bounds access.

This is not hypothetical. `unpack_u4_f32` once emitted its bounds guard *after* the
work it guarded, so every out-of-range lane performed a 32-byte out-of-bounds write.
A dead-load rule would have removed the faulting load and left a kernel that looks
clean and is still wrong. The bug was found by property tests asserting that stores
follow their guard; a simplifier willing to delete the load is exactly the thing that
would have hidden it.

The cost is real: dead loads survive into the PTX and occupy register lifetimes. That
is the correct trade for a backend whose entire premise is that you can see every
instruction and reason about it. A rule that hides faults is worse than a missed
optimisation.

**Reversible?** No, unless gpark grows a memory-safety proof that no load address is
out of bounds. That proof does not exist yet.

---

## Opcode purity is classified explicitly, never inferred

**Decided.** Every one of the 48 opcodes is listed in `Gpark.Opt.Simplify` as either
pure or impure, and a test fails CI if the table grows an unclassified opcode.

**Why.** The obvious implementation is a prefix test — `String.starts_with?(base,
"st")` for stores, and treat everything else as arithmetic. That is silently wrong the
day someone adds an opcode: anything unrecognised falls through as "pure", and the
pass starts deleting instructions that write memory.

`classify/1` therefore returns `:unknown` as a real answer rather than defaulting, and
`:unknown` is not `:pure`. An unclassified opcode is kept.

**Note.** `classify/1` takes a base name (`"red"`), because the IR keeps space,
modifier and vector width in separate fields. A fully-spelled opcode like
`red.global.add.s32` is never handed to it, and lands on `:unknown`.

---

## What gpark is *not* adopting from Taichi

Recorded so the omissions read as decisions rather than as work not reached.

| Taichi | gpark | Reason |
| --- | --- | --- |
| `full_simplify` | **Adopted** | Fixpoint over rules. See above. |
| Neutral SIR / no language coupling | **Adopted** | `Gpark.Backend` is the contract and `require!/2` is the enforcement. The IR was already neutral in that nothing in it knew about PTX; what was missing was anything stopping a backend from emitting a slower sequence for a capability it lacked. A gate that accepts everything is no gate, so the tests refuse real kernels and check the refusal names the capability. |
| `ti.kernel` / `ti.data` Pythonic surface | **Planned** | The DSL is the ergonomics answer. Not written. |
| Reverse-mode autodiff | **Declined** | gpark has no runtime and no tape. Adding one means owning a graph format and a backward pass, which is a second project. The quant interest here is explicit quantisation, not learning. |
| Runtime, `ti.init()`, memory pools | **Declined** | Launch overhead and allocation are exactly what gpark wants visible. A pool hides the allocation the benchmark is measuring. |
| Vulkan / Metal / OpenGL backends | **Declined** | gpark's premise is direct PTX at instruction level. Multi-API is a different project with a different justification. |
| `ti.fuse` (loop fusion) | **Deferred** | Needs the loop machinery, which needs SSA — see the no-SSA entry. |
| Aggressive vectorisation | **Deferred** | `vec_add_f32_v4` is in the corpus to pin the surface down; tuning it needs a GPU to measure against. |
| Autotuning templates | **Declined** | gpark has one target and no measured hardware baseline yet. An autotuner with no roofline to search is just a random number generator. |
