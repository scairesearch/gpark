# Systems

This is the document that decides what gpark is for. Everything else — the opcode
table, the type system, the kernels — is downstream of the argument here.

## The argument

A GPU kernel is not slow because of arithmetic. It is slow because of one of four
things, and the fix for each is completely different:

| Limit | Symptom | What actually fixes it |
|---|---|---|
| **Bandwidth** | time ∝ bytes moved | fewer bytes: fuse, compress, keep data resident |
| **Latency** | time ∝ dependent memory ops | more concurrency: more loads in flight, not more FLOPs |
| **Occupancy** | low SM utilisation | fewer registers per thread, smaller tiles |
| **Launch overhead** | time ∝ number of kernels | graph capture, or fewer kernels |

Most kernel libraries are built around the first, because a GEMM is the easiest
thing to benchmark and everybody has one. That is why Triton is excellent at GEMM
and why Triton users hit a wall the moment the problem stops being a GEMM.

gpark starts from the opposite end. Every kernel in `corpus/` is deliberately
bandwidth-bound with arithmetic intensity near zero — a saxpy, an unpack. If gpark
cannot hit roofline on *those*, nothing it does on anything harder will mean
anything, and there is no point measuring a matmul yet.

## Why direct PTX is the foundation

Because the four limits above are decided by instructions, and every abstraction
that hides instructions eventually hides the decision.

Concretely, in a Triton kernel you can ask for a tile size and get something
sensible. When it is slow you can rephrase the same computation three ways and get
roughly the same throughput, with no way to ask why. In gpark the register count
*is* your problem statement. You can see that `unpack_u4_f32` writes eight f32
stores per thread, that this costs eight live address registers, and that on a part
with 255 registers per thread the tile could go to sixteen elements instead. That is
a decision you cannot make from Triton.

The cost of this choice is real and worth stating: no nvcc, no vendor libraries, no
`cublasLt` heuristics, no Tensor Cores in v0.1. gpark is slower to write kernels
in than Triton. It is faster to *understand* them, and the understanding is the
point.

## Register pressure is the program

`Gpark.IR` has no SSA and no hidden register allocator. Registers are numbered by
hand. This is not asceticism — it is a refusal to have the interesting variable be
implicit.

On a bandwidth-bound kernel, the number of loads in flight per thread *is* the
memory-level parallelism, which *is* the latency hiding, which *is* how close to
roofline you get. A compiler that allocates registers for you optimises for
something else, usually occupancy, and hands you a kernel that is correct and
invisible to reason about. gpark makes you write the number down.

The consequence: `remote/ptxas_check.sh` treats a register spill as a hard failure.
A spill means the hand-written register assignment did not fit, and the fix is to
restructure the kernel, not to hope ptxas coped.

## Where the wins actually are

Ranked by how much they matter for the workloads gpark targets, best first.

**1. Bytes moved.** The only limit you can beat structurally. `unpack_u4_f32`
moves 36 bytes per input word, 32 of which are the output — so a fused
dequantise-and-multiply that keeps the weight in `bf16` instead of `f32` moves 18.
That is a 2× win available with no cleverness at all, just a decision about the
output type. Every quant kernel should be measured in bytes first.

**2. Launch overhead.** An RL step is often 40 tiny kernels; 40 launches at ~5 µs is
200 µs of pure overhead against arithmetic that takes 20 µs. CUDA Graphs collapse
this to one replay. Taichi has no equivalent at all, and runs on a single stream,
which is why its async work is a research feature rather than a shipped one. This is
the clearest place gpark can be categorically better than an existing project, and
`remote/graph_bench.cu` measures the floor a graph implementation has to beat.

**3. Register pressure and tiling.** Second-order, and the thing Triton was designed
for. gpark is not yet better here and does not claim to be.

**4. Occupancy.** Usually a proxy problem. A kernel at 25% occupancy is usually
register-bound, not occupancy-bound; fix the registers first or you will spend a
week improving a number that was never binding.

## What is deliberately not optimised yet

- **Shared memory tiling.** Deferred. Tiling is how you beat bandwidth on
  *reductions* and *matmuls* by reusing loaded data, and gpark has neither in a form
  that needs it yet. The first kernel that does will drive the design.
- **Tensor Cores (`mma.sync`).** Deferred. They change the shape of a kernel
  completely — fragment layouts, ldmatrix, double buffering — and doing that before
  the scalar path is measured and understood would be optimising a program nobody
  has validated yet.
- **Vectorised access (`.v4`).** In the IR and emitter, but no kernel uses it yet.
  `vec_add_f32` is scalar on purpose, as a control: if the vectorised version does
  not measurably beat the scalar one, the addressing is wrong.

## The rule for adding a kernel

A kernel goes in `corpus/` only if it forces something new in the backend — a new
opcode shape, a new addressing form, a new register class, a new validator rule.
A kernel that exercises nothing new is a test, not a kernel, and belongs in the
test suite.

That rule is why the corpus is four kernels and not forty, and why
`test "covers the intended kernel families"` asserts exactly which four. A corpus
that grows by accretion stops being evidence of anything.

## Cross-backend honesty

gpark will emit Metal and AMDGCN. When it does, the risk is not that the backends
are bad; it is that they are *equally mediocre* and nobody notices because nothing
measures them.

So: every backend must pass the same corpus, and CI fails if a backend silently
loses a capability. A backend that cannot do `shfl.sync.bfly` needs to say so
loudly, not quietly emit a scalar fallback. See `ARCHITECTURE.md` for the
capability matrix and `ROADMAP.md` for the parity gate.

Taichi's failure mode here is the cautionary example, and it is worth being precise
about: Taichi is not badly built, it is *differently* built — its `full_simplify`
fixpoint and backend-neutral SIR are genuinely better than most projects manage.
But its Metal path needs zero language extensions while its AMDGPU path is an
assertion, so the abstraction is genuinely neutral on one target and nominally
neutral on the others. Marketing "vendor neutral" without a parity gate is how you
get that.
