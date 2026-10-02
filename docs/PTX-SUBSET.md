# PTX Subset

What gpark can emit today, and what it refuses to.

The rule: an opcode enters `Gpark.Ops` only when gpark can state its exact PTX
signature. "It is probably fine" is how a backend ends up emitting plausible text
that `ptxas` rejects four hundred lines later.

48 ops. Every one is declared with operand types, result type, and whether an
explicit output type is required.

## Module and program structure

```
.version 8.7
.target sm_80
.address_size 64
.visible .entry <name>( .param ... )
<blocks>
```

- `.version 8.7`, `.target sm_80`, `.address_size 64` on all four kernels. Target is
  hardcoded for now; `ptxas_check.sh` reassembles at whatever `GPARK_ARCH` says,
  so the hardcoded target is validated rather than trusted.
- `.visible .entry` only. No `.weak`, no `.extern .func`, no call graphs.
- `.shared` arrays are emitted by the validator but no kernel uses them yet.

## Parameter access

| Opcode | Form |
|---|---|
| `ld.param.u32/u64/f32` | scalar param load |
| `ld.param.v4.*` | vector param load (emitter supports, unused) |

`.const` params are not distinguished in v0.1. Harmless today, wrong the moment a
kernel takes a pointer-to-param.

## Memory access

| Opcode | Notes |
|---|---|
| `ld.global.f32`, `ld.global.u32`, `ld.global.u64` | scalar |
| `ld.global.v2/v4.f32/u32` | emitter supports, unused |
| `st.global.f32`, `st.global.u32` | scalar |
| `st.global.v2/v4.f32/u32` | emitter supports, unused |
| `ld.local.*` / `st.local.*` | present for spill reasoning; no kernel uses them |

Addresses are **normalised, not deferred**. `IR.addr(base, offset)` computes the
full byte address as an `add.u64` when the offset is dynamic. The alternative —
emitting the address expression and lowering it in the emitter — pushes arithmetic
into the layer that should not be doing arithmetic, and it is how you get addressing
bugs that only show up on large arrays.

Memory space is a literal in the opcode name (`ld.global`), so a kernel cannot
accidentally emit `ld.shared` with a global address. Global loads are not marked
`.nc` or `.cs`: cache hints are a performance decision, and gpark does not yet make
performance decisions.

## Arithmetic

| Group | Opcodes |
|---|---|
| Integer | `add` `sub` `mul` `mul.wide.u32` `mad.lo` `div` `rem` (u32/s32) |
| Float | `add.f32` `sub.f32` `mul.f32` `fma.rn.f32` `neg.f32` |
| Comparison/pred | `setp.{eq,ne,lt,le,gt,ge}.{u32,s32,f32}` → `.pred` |
| Conversion | `cvt.{rn,sat}.f32.u32` `cvt.{rn,sat}.f32.s32` `cvt.rn.u32.f32` `cvt.rn.s32.f32` |
| Bit | `and.b32` `or.b32` `xor.b32` `not.b32` `shl.b32` `shr.{b,u}.b32` |
| Move | `mov.{u32,s32,f32,b32}` |

**Float division and square root are absent on purpose.** `div.rn.f32` and
`sqrt.rn.f32` are slow on NVIDIA hardware, and any correct kernel that uses them
should be using a reciprocal approximation instead. Emitting the exact division by
default would hide a performance cliff behind a convenience. When they are added,
they will be named to make the cost obvious.

`cvt` requires explicit source and destination types — `cvt.rn.f32.u32`, never a
guessed pairing. Integer↔float conversion is genuinely ambiguous (`f32`→`u32`
saturates, `.sat` says so explicitly), and inference would pick wrong silently.

`mul.wide.u32` exists with operand types `[:u32, :s32]` — the PTX signature really is
asymmetric, and the table was wrong about it until a corpus test caught the
mismatch.

## Control flow and sync

| Opcode | Notes |
|---|---|
| `bra` | unconditional and predicated |
| `bra.uni` | present |
| `ret` | block terminator |
| `setp.*` | produces `.pred` |
| `not.pred` | used by every predicated branch |
| `shfl.sync.bfly.f32/u32` | warp butterfly, for reductions |
| `shfl.sync.idx` | emitter supports, unused |
| `exit` | present; validator's exit check is imprecise (see `VALIDATION.md`) |

Every block must end in a terminator, and the validator enforces it. Early exit
inside a body is the tempting way to write a bounds check, and it is exactly how an
out-of-range lane ends up doing the guarded work anyway — see `KERNELS.md`.

`bar.sync` is not implemented. Shared-memory kernels will need it, and it is the
first thing that makes multi-block kernels possible.

## Sub-byte types

`u2 s2 u4 s4 u6 s6 fp4` and friends are **packed bit containers**. There is no
packed arithmetic in hardware, so:

- A `u4` lives in a `.b32` register.
- Operations validate against the logical type but allocate against storage width.
- Extracting 8 `f32` from one `u32` is: shift to position, `and.b32` with a 4-bit
  mask, `cvt.rn.f32.u32`, store. 8 shifts, 8 masks, 8 conversions, 8 stores per
  word.
- The mask must be applied after shifting. Applying it before leaves the wrong bits
  in the high lanes, which produces plausible numbers on the first element and
  garbage on the rest — a bug that survives a small test and fails on real data.

PTX does have packed instructions (`sub`, `mul`, `min`/`max` on `.b8`/`.b16` and
below) for sub-byte *integer* types, and gpark could emit them. It does not yet,
because they only exist for sub-byte integers — not floats, not anything with a
fraction. Supporting them would create two different unpack paths for `u4`
depending on the operation, which is a worse first outcome than one clear
bit-twiddling path. Native `mma` with sub-byte types is a much later question.

## Explicitly not supported

Stating this is more useful than leaving it to be discovered:

- **No `mma.sync` / Tensor Cores.** Fragment layouts and `ldmatrix` are a different
  programming model.
- **No `red`/`atom`/`red` across shared memory**, no atomics.
- **No texture, surface, or grid-constant `__grid_constant__`.**
- **No cooperative groups.**
- **No async copy (`cp.async`)** — this is the big one for bandwidth kernels, and the
  first thing to add once shared memory exists.
- **No `prefetch`/`discard`.**
- **No `.section`, no relocatable device code.**
- **No branch hints (`bra` with `.L_x_1` hints).**

## Coverage

The subset is chosen backwards from the corpus. Every opcode in the table exists
because a kernel in `corpus/` needed it; every opcode a kernel needed is in the
table. There is no speculative surface, and `ptx_test.exs` asserts which kernel
families are covered so the corpus cannot silently shrink.

48 ops, 30 types, 4 kernels, 30 Elixir tests, 4 Python tests, all static checks
green. No GPU has assembled or run any of it.
