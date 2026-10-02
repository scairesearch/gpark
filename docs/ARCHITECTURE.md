# Architecture

How the pieces fit, and why the pieces that exist are the pieces that exist.

## The layer stack

Nothing bypasses this. Every path from user code to PTX goes through `Gpark.IR`.

```
   future:  Gpark.DSL · Gpark.Runtime · Gpark.Graph · Gpark.Autotune
                              |
                              v
           legalization · vectorization · scheduling · register allocation
                              |
                              v
                        Gpark.IR  ──────────────────────────┐
                              |                               │
                              v                               v
                        Gpark.PTX                     Gpark.Validate
                              |                               |
                              v                               v
                     PTX text (corpus/golden)          static checks
```

The constraint is the design. A DSL that emits text directly, or a "fast path" that
bypasses the IR, would make every invariant in `Validate` conditional on which API
you used — and the whole reason to have a validator is that it is unconditional.

## Current modules

### `Gpark.Ops` — the typed opcode table

One table maps opcode → operand types, result type, and required `otypes`. 48 ops
today. This is the single source of truth for what gpark can express, and both
implementations mirror it.

Notable shapes:

- `cvt` takes source and destination types: `cvt.rn.f32.u32`, not a guessed pair.
  Integer↔float conversion is not inferable from operands.
- Bitwise ops (`and`, `or`, `xor`, `not`, `shl`, `shr`) accept *bit-container* types
  (`:b16`, `:b32`, `:b64`) alongside the integer types. This is the whole
  sub-byte mechanism: there is no packed arithmetic in hardware, so a 4-bit value
  lives in a `.b32` register and every operation on it is a 32-bit op plus a mask.
- `otypes` is enforced rather than advisory. An instruction that produces `f32`
  cannot be given a `.u32` register, which is how a real class of emitter bug
  surfaced during development.

A separate test hashes this table on both sides and compares digests, so it cannot
drift.

### `Gpark.IR` — the program representation

Blocks, instructions, registers, addresses. No SSA, no allocator, no hidden
temporaries: registers are numbered explicitly.

Addresses are normalisers, not just syntax sugar. Given `IR.addr(base, offset)`
where `offset` is a register or immediate, the IR computes the full byte address up
front rather than emitting an address expression the emitter would then have to
lower. `addr/2` also accepts a bare integer offset, and address-nested registers
join `.reg` discovery — both were bugs where a kernel that validated cleanly produced
undeclared registers in the emitted text.

### `Gpark.Type` — containers and element formats, kept separate

This split is the most important structural decision in the type system, and it
exists because of sub-byte data.

Two questions, previously conflated into one list:

- **Container** — how many bits does this register hold? `b1 b2 b4 b8 b16 b32 b64`
- **Element format** — what do those bits mean? `s2 u2 s4 u4 s8 u8 s16 u16 s32 u32
  s64 u64 f16 bf16 f32 f64 e2m1 e2m3 e3m2 e4m3 e5m2 e8m0 pred`

30 types total, 26 with a direct PTX spelling and four (`s2 u2 s4 u4`) that exist
only inside a packed container. Conflating the two lists makes `u4` look like a
thing the hardware can compute on, and it cannot — it is 4 bits inside a `.b32`.

`Packed` carries a sub-byte logical type (`s2`…`u4`) alongside its container width.
`Gpark.Type.describe/1` resolves both tables through one lookup, so `width`, `kind`
and `sign` cannot disagree about what a sub-byte type is.

Worth being precise about: `unpack_u4_f32` does **not** use `Packed`. It hand-rolls
shift/mask/convert on `:u32` registers, because that is expressible today. `Packed`
exists and is introspectable but no corpus kernel uses it yet — see `ROADMAP.md`.

### `Gpark.PTX` — the emitter

Deterministic text output. `.version 8.7`, `.target sm_80`, `.address_size 64` on
every current kernel.

Deliberately boring. No map iteration anywhere in emit order, no timestamps, no
environment-dependent formatting. The goldens are byte-stable, which is what makes
tier-two and tier-three validation possible at all.

### `Gpark.Validate` — the invariant checker

Described in `VALIDATION.md`. Short version: structural, type, bit-container,
initialisation and exit checks over IR and emitted text, run independently by both
implementations.

### `Gpark.IR.JSON` — the corpus format

Canonical, byte-stable serialisation of IR to `corpus/specs/*.json`. Golden PTX to
`corpus/golden/*.ptx`. This pair *is* the cross-language contract, which means the
serialiser is load-bearing: a change to field ordering is a breaking change to
every golden's provenance.

`srctype` is nullable and round-trips as `null`. Sub-byte kernels need no source
type on their element loads, so requiring one would mean inventing a fake.

## Kernels

Four, and the rule for adding one is that it must force something new in the
backend — see `SYSTEMS.md`. All four are bandwidth-bound with near-zero arithmetic
intensity, chosen as controls: if gpark cannot reach roofline on these, nothing
harder means anything.

| Kernel | Forces |
|---|---|
| `vec_add_f32` | baseline: params, `ld.global`, `st.global`, address normalisation, grid mapping |
| `saxpy_f32` | scalar `fma.rn`, the arithmetic/bandwidth boundary |
| `reduce_sum_f32` | `shfl.sync.bfly`, warp collectives, a lane-dependent output path, narrowing |
| `unpack_u4_f32` | packed bit containers, `shr`/`and` masking, `u32`→`f32` `cvt`, multi-word expansion |

`reduce_sum_f32` is one warp and one block by construction: the butterflies are
warp-level, and there is no shared-memory or grid-level reduction yet. It documents
a warp-multiple `n`. Harness input is padded accordingly.

`unpack_u4_f32` is the kernel the type system exists for: 8 f32 lanes out of one
`u32` word, 4 words per thread, byte offsets rather than element indices on the
output.

## Two implementations, deliberately

Elixir and Python both build the same goldens from the same specs.

The cost is real: every emitter change is two changes, every type bug is two bugs.
The justification is that it is the only automated defence against drift, and drift
between language backends is silent. It surfaces as "works in Elixir" and takes an
afterthought to debug.

A third backend would triple the maintenance cost before the corpus grows enough to
pay for it. Metal and AMDGCN wait until `SYSTEMS.md`'s priority list reaches them.

The repo also contains `riscgp/`, an unrelated project. It is not part of gpark and
is not to be committed here.

## What is not here yet

Listed with reasons, because a roadmap section that says "and then everything" hides
the decisions.

- **`Gpark.DSL`** — blocked on the IR being settled. A DSL written against a moving
  IR is rewritten, not extended.
- **Shared memory / tiles** — no kernel needs it yet, and it changes register
  pressure completely. `unpack_u4_f32` is the first kernel with enough working set
  to make the decision real.
- **`mma.sync` / Tensor Cores** — fragment layouts and `ldmatrix` are a different
  programming model. Doing them before the scalar path is measured means optimising
  something nobody has validated.
- **Vectorised `.v4` access** — in the emitter, unused. `vec_add_f32` is scalar on
  purpose: if vectorising it does not measurably help, the addressing is wrong.
- **Metal / AMDGCN backends** — need the cross-backend parity gate first, or
  "vendor neutral" becomes a claim instead of a fact.
