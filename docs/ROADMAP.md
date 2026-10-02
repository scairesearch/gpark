# Roadmap

Ordered by what unblocks the most, not by what is most fun. Hardware-dependent
items are marked, because a large fraction of this list cannot be verified from a
laptop and pretending otherwise produces confident nonsense.

## Now: make the foundation provable

- [x] Elixir backend: ops, IR, types, emitter, validator
- [x] Four kernels covering four capability families
- [x] Independent Python implementation, byte-for-byte parity
- [x] Static validation tier — runs on every commit
- [x] ptxas assembly tier with spills as a hard failure
- [x] Execution tier with CPU references and achieved bandwidth
- [x] Documentation of architecture, systems, validation, PTX subset

## Needs hardware, in order

**1. First real numbers.** Everything else is inference until `make remote-build`
runs. Priorities: assembly for a second architecture (`sm_90`), then execution, then
achieved bandwidth against a roofline calculation for the specific part.

Without this the project has no performance claim at all, and that is the correct
current state rather than a shortfall.

**2. Clamp the reduction index.** `reduce_sum_f32` trusts the caller for a
warp-multiple `n` and reads out of bounds otherwise. Clamping the index is the real
fix. It needs hardware to validate, so it is queued here rather than done blind.

**3. Graph systems: all of them, not one.** The "GraphSuite" reference was ambiguous
from the start — CUDA Graphs, cuGraph, or a named third-party system — and the answer
is to stop narrowing it. `graph_bench.cu` now measures three paths on one device in
one process: plain launches, runtime-API stream capture, and a hand-built `cuGraph`.
Capture and `cuGraph` are different mechanisms, not duplicates: capture records a
stream, so the graph it produces is linear unless the stream already implied the
dependencies, while `cuGraphAddKernelNode` lets each node name its own predecessors
and can express a DAG. Real multi-kernel pipelines are DAGs.

`--external <path>` times another graph implementation from inside this harness.
The reason is that graph-overhead claims are usually not comparable — clocks, driver
state and thermals move between runs — so measuring both sides in one process is the
only way the numbers mean anything.

What is still missing is the comparison against a system that does *not* use CUDA
Graphs at all, which is where the real portability claim would live.

**4. The MAGMA comparison.** A real benchmark needs a real baseline: pick a target
problem, write it honestly, and be clear that gpark has no matmul yet. Not written
because there is currently nothing to compare against.

## Next: shared memory, because two kernels now need it

Both current limitations trace to the same missing piece.

`unpack_u4_f32` writes 32 bytes per thread and holds 8 live addresses. On a part
with 255 registers per thread the tile could be sixteen elements instead of four, if
the offsets were not consuming the whole register file. `reduce_sum_f32` needs
cross-block reduction, which needs `bar.sync`.

- [ ] Shared memory declarations and `ld.shared`/`st.shared` — the emitter
      validates them, no kernel uses them
- [ ] `bar.sync`, the gate on every multi-block kernel
- [ ] `cp.async` — the async-copy path for bandwidth kernels, and the single largest
      available win once tiling exists
- [ ] A tiled kernel that would be impossible without it, so the feature is proved
      by use rather than by test

The trigger for this section is a kernel that *cannot* be written without shared
memory, not a desire to have it.

## Then: the parts with a real argument behind them

- [ ] `mma.sync` and Tensor Cores. Fragment layouts and `ldmatrix` are a different
      programming model; doing this before the scalar path is measured means
      optimising a program nobody has validated.
- [ ] Vectorised `.v4` access, used at last. `vec_add_f32` is scalar on purpose: if
      vectorising it does not measurably beat it, the addressing is wrong, and that
      is worth knowing before tiling builds on it.
- [ ] A kernel that actually uses `Gpark.Type.Packed`. `unpack_u4_f32` hand-rolls
      shift/mask/convert on `:u32` registers, so the packed representation is
      exercised only by doctests. It should be exercised by a kernel before any
      performance claim is made about sub-byte work.
- [ ] Register allocator with live-range splitting, and `Gpark.Mid` SSA with phi
      nodes to replace the unrolled unpack body.
- [ ] Cross-backend parity gate: Metal and AMDGCN must pass the same corpus, and CI
      fails if a backend silently loses a capability. Without this gate,
      "vendor neutral" is a claim rather than a fact.
- [ ] Sub-byte packed instructions. PTX does have them for sub-byte *integers* — not
      floats. Supporting them creates two unpack paths for `u4` depending on the
      operation, which is a worse outcome than one clear bit-twiddling path until
      there is a reason.

## Later: the user-facing surface

Sequenced so each layer is provable before the one above it exists.

- [ ] `Gpark.DSL` — a Triton-shaped front end: blocked tensors, tile shapes,
      `tl.load`/`tl.store`/`tl.dot`. Blocked on the IR settling, because a DSL
      written against a moving IR gets rewritten rather than extended.
- [ ] Legalisation and lowering — DSL ops to the four kernel families above.
- [ ] `Gpark.Runtime` — `Array`, caching `MemPool`, `Stream`, `Event`.
- [ ] `Gpark.Graph` — capture, instantiate, update. The measured win is already
      quantified by `graph_bench.cu`; this is just the API around it.
- [ ] `Gpark.Autotune` — needs a target architecture and a real problem to tune.

## Not planned

Named so they are decisions rather than omissions.

- **A third language implementation before the corpus grows.** Every additional
  backend multiplies the emitter maintenance cost. Two already exist to catch drift;
  a third would cost more than it catches until there are more kernels.
- **Implicit register allocation.** The hand-assigned register file is the point.
  See `SYSTEMS.md`.
- **Occupancy as a target.** It is usually a proxy for register pressure. Optimising
  it directly usually means fixing the registers instead.
- **Vendor libraries.** No `cublasLt`, no cuDNN, no vendor kernels. gpark emits PTX
  or it does not.

## Standing honesty rules

These exist because the failure modes are real and this project is young.

- No performance number in any document until it came from hardware. There are
  currently none.
- Every limitation is written down where it lives, not just in a changelog.
  `VALIDATION.md` has a "known limits" section; `KERNELS.md` documents the
  reduction's weaker contract; `PTX-SUBSET.md` lists what is refused.
- A test that has never been seen to fail is not a test. New assertions get checked
  against the code they are supposed to catch before being trusted.
- No feature in the table unless a kernel needs it. That is why it is 48 ops.
