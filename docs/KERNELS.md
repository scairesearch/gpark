# Kernels

Four kernels, and the reasoning behind each. The rule for adding one is in
`SYSTEMS.md`: it must force something new in the backend.

All four are bandwidth-bound with near-zero arithmetic intensity, deliberately. They
are controls. If gpark cannot approach roofline on a saxpy, then a gpark matmul
number means nothing.

---

## `vec_add_f32` — the baseline

`out[i] = a[i] + b[i]`, one thread per element.

The control kernel. Scalar, unrolled, nothing clever. If this is not fast, the
problem is addressing or register assignment, not the algorithm.

Forces: parameter loads, `ld.global`/`st.global`, address normalisation, grid
mapping, `mul.wide`.

### The one detail worth stealing

```elixir
IR.instr("mul.wide", dtype: :u32,
  dest: IR.reg(:u64, @rd_off),
  ops: [IR.reg(:u32, @r_gid), IR.imm(4)])
```

`gid * sizeof(f32)` in plain `.u32` overflows at 2^30 elements — a 4 GiB array, which
is not exotic. `mul.wide` widens the multiply to 64 bits, so the offset is correct
past that point. Every index computation in gpark uses it, because the alternative
is a bug that only appears on arrays nobody tests.

The output offset is computed the same way and the same reason.

---

## `saxpy_f32` — the arithmetic boundary

`y = alpha * x + y`, one thread per element.

Same memory traffic as `vec_add`, but `fma.rn.f32` instead of a separate multiply
and add. The point is the *category* difference: this kernel is arithmetic-bound by
a hair, which makes it the place where the "bandwidth-bound is always true"
shortcut stops being true.

Forces: scalar `fma`, the tolerance question in validation.

### Why this kernel cannot be bit-compared

`fma.rn.f32` computes `alpha * x + y` with a **single** rounding. The CPU computes
it with two, or contracts it and gets one. Both are correct; they disagree in the
last ulp. A harness that demanded bit equality would be asserting something PTX does
not promise, so `exec_harness` uses a tolerance here and exact comparison for the
integer and unpack kernels, where exactness is real.

That asymmetry is deliberate. Tolerances applied uniformly hide real bugs; the
question is always whether the hardware *promises* exactness for that operation.

---

## `reduce_sum_f32` — the first kernel that must know the hardware

Sum one warp's worth of `f32` into lane 0.

### Butterfly, not down-shift

`shfl.sync.bfly` with a halving stride gives **every** lane the full sum in 5 rounds
for 32 lanes. The naive down-shift reduction gives the answer to lane 0 only. The
butterfly costs the same and is what you want whenever the result feeds back into
per-lane work — which is most of the time.

No shared memory, no `bar.sync` in the reduction itself. `bar.sync` appears only
because the grid is capped at one block in v0.1; a real multi-block reduction needs
atomics across blocks, which is a separate kernel and a separate problem. Faking it
here would hide the hard part behind the easy one.

`volatile` is deliberately absent: `shfl.sync` is already warp-synchronous, so its
operand needs no memory fence.

### This kernel's bounds contract is weaker, and that is a real limitation

`reduce_sum_f32` loads **before** it guards. A reduction has to load every lane, so
there is no per-element guard to hoist; instead the kernel clamps the lane index to
`n - 1` and relies on the caller passing a warp-multiple `n`.

That is a genuinely weaker guarantee than `vec_add_f32` or `unpack_u4_f32` have, and
it is a documented contract rather than an enforced one. `exec_harness` pads its
input to a warp multiple for exactly this reason. A caller who launches with a
non-warp-multiple `n` gets an out-of-bounds read.

The correct long-term fix is to clamp the *index* rather than trust the count — and
that is written down in `ROADMAP.md` rather than done here, because fixing it
properly needs hardware to validate.

`ptx_test.exs` encodes this asymmetry explicitly: the "no load before the guard"
assertion covers the three per-element kernels and excludes this one, with the reason
in the test.

### Numerical note

Butterfly reduction sums in a different order than a sequential sum, so it will not
be bit-identical to `numpy.sum`. Inherent to any parallel reduction.

---

## `unpack_u4_f32` — the kernel the type system exists for

Unpack a `u4` weight tensor to `f32`: 8 values per `u32` word, one thread per word.

```
4 bytes in, 32 bytes out
```

Triton has no 4-bit arithmetic, so a quantised kernel there means upcasting by hand
and hoping the compiler keeps the shape. Taichi has no sub-byte primitive at all,
only an optional `quant` extension that does not lower to hardware. Here the unpack
is explicit and the cost of that explicitness is the whole story: **32 of the 36
bytes are output**, so this is a dequantise kernel, not a matmul. The only lever is
issuing the loads and stores well enough not to become the bottleneck.

### The bit twiddling

One `u32` holds 8 unsigned 4-bit values, low nibble first:

```
t = (word >> (4 * k)) & 0xF
```

The `.b32` suffix on the shift and mask is the entire reason `Gpark.Ops` accepts
bit-container types. Masking a `.b32` word is not expressible in the signed/unsigned
integer types, so without that change the kernel cannot be written at all.

Mask **after** shifting. Masking first leaves the wrong bits in the high lanes: the
first element is correct, the rest are garbage. That failure mode survives a small
test and only shows up on real data.

Nibble 0 skips the shift-and-mask entirely and reads the word directly — two
instructions that buy nothing.

### Why it is unrolled

Eight copies of shift/mask/convert/scale/store rather than a loop. A loop needs a
phi node and a dynamic trip count; gpark v0.1 has no SSA and no phi, by design,
because that belongs in the register allocator.

The unrolled form costs registers — 8 live output offsets plus the word and shift.
That cost is the exact tradeoff the allocator will later be asked to get right, and
the honest thing is to leave it visible rather than hide it behind a loop the
compiler can unroll for us.

`bfi`/`prmt`/`bfe` would cut the instruction count substantially on real hardware.
Deferred, and listed in `PTX-SUBSET.md`.

### The bug that made this kernel interesting

This kernel shipped with a bounds guard that **emitted after the work it guarded**.

The branch was the block terminator and the body was the instruction list, so it
necessarily came last. Every out-of-range lane ran the full body, including a
**32-byte out-of-bounds write** past the end of the output buffer.

Nothing caught it. The golden was self-consistent. The validator passed. Every
existing test stayed green. Both were handed a program that was valid PTX and merely
not the program intended — which is precisely the class of bug byte-for-byte
regression testing cannot find, since it detects *change*, not *incorrectness*.

The fix matches what `vec_add_f32` already did correctly: the guarded branch is an
instruction in the list, and the block terminates with `ret`.

`ptx_test.exs` now asserts that no global store precedes the bounds guard in any
kernel. The regression test was verified to fail against the pre-fix golden before
being trusted.

---

## The corpus as a contract

```
vec_add_f32      addressing, widening, the baseline
saxpy_f32        fused arithmetic, the tolerance boundary
reduce_sum_f32   warp collectives, a lane-dependent store, a documented contract
unpack_u4_f32    packed bit containers, multi-word expansion, the memory-safety rule
```

Four families: memory addressing, arithmetic, synchronisation, and sub-byte data.
The backend needs all four to be non-trivial before it is worth optimising.

`ptx_test.exs` asserts exactly which kernels are present, so the corpus cannot
quietly shrink into testing nothing.
