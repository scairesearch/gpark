# Validation

What gpark checks, what it deliberately does not check, and why the gap is not a
bug in the validator but the reason it is useful.

The short version: gpark can prove that generated PTX is *internally consistent*.
It cannot prove that PTX is *correct*, because nothing here has ever run on a GPU.
The two claims are independent, and conflating them is the failure mode this
document exists to prevent.

## The three tiers

| Tier | Runs on | Needs | Proves |
|---|---|---|---|
| **Static** — `make test` | any dev machine | nothing | PTX is well-formed, internally consistent, and byte-identical across two independent implementations |
| **Assembly** — `make remote` | CUDA toolkit, no GPU | `ptxas` | real hardware accepts the module; register/spill/stack numbers |
| **Execution** — `make remote-build` | NVIDIA host + GPU | `nvcc`, `libcuda` | kernels compute the right answer and hit a bandwidth number worth quoting |

Only tiers two and three tell you anything about performance. Only tier one runs in
CI on every commit. That split is intentional: the tier that catches regressions
must be the tier that is cheap enough to always run.

## Tier one: static checks

`Gpark.Validate` walks the IR and the emitted PTX together. It is not a PTX
parser — it checks the IR it is handed, and cross-checks the text the emitter
produced from it, which catches emitter bugs that a purely IR-level validator would
miss.

What it checks:

- **Structure.** Every block ends in a terminator. Labels are unique. Every branch
  target names a label that exists. Module headers (`.version`, `.target`,
  `.address_size`) are present.
- **Registers.** Every `.reg` declaration exists, has a known width and type, and
  every register used is declared with a consistent type. Declared registers must
  not be used before definition on a path.
- **Types.** Operand types match the opcode signature. Declared operand types must
  equal the opcode's declared output type — an instruction whose register is `.u32`
  cannot produce `f32`.
- **Bit containers.** `shl`/`shr`/`and`/`or`/`xor`/`not` on `u4`/`s4`-style types
  operate on `u32` storage, and the validator enforces that. Sub-byte values are not
  given their own registers or their own arithmetic.
- **Initialisation.** Reads on a register that no reachable path has written are
  reported. (This check is a linear approximation, not a dataflow fixpoint — see
  the limits section below.)
- **Exit.** Blocks with no reachable successor, and terminators, are flagged.

Both implementations run the same checks independently over the same corpus.

### The cross-implementation check

This is the part worth defending. Elixir and Python each build the PTX from the same
JSON IR spec, and the test asserts the resulting bytes are identical.

That catches a whole class of bug that a single implementation cannot: an emitter
fix applied to one language and forgotten in the other, a type table that drifted, a
sub-byte lowering that only exists in one place. Those bugs are invisible until a
user picks the other language.

A separate check hashes the opcode tables on both sides and compares them, so the
48-op tables cannot drift apart unnoticed. The Python test shells out to
`mix run` to compute the Elixir digest — an awkward dependency, and a cheap one,
versus discovering the drift six months later.

### Byte-for-byte, not just equivalent

Equivalent PTX is not the contract. **Identical bytes** are. Rationale: once real
`nvcc` is available, goldens get diffed against compiler output to track
divergence. That only works if the input is stable, and "stable modulo whitespace"
is not a thing anyone can rely on.

The cost is that the emitter must be deterministic — no map iteration order, no
timestamps, no environment-dependent formatting. That constraint is the reason the
emitter is as boring as it is.

## Tier two: assembly

`remote/ptxas_check.sh` assembles every golden at `-O3` for a given architecture
and parses the resulting resource usage.

**Register spills are a hard failure, not a warning.** This is the important part
and it deserves justification. gpark assigns register identifiers by hand — there is
no allocator — so a spill means the kernel as written could not hold its working
set in registers. The compiler papered over it by going to local memory, which turns
a bandwidth-bound kernel into a bandwidth-bound kernel plus a `local` round trip.

Accepting spills silently would mean the hand-written register assignment, which is
the entire premise of the direct-PTX approach, was wrong, and nothing said so.

Spill *stores* are checked separately from spill *loads*, because they are different
bugs: a store means a value was live across too much, a load means it was rematerialised.

The script also reports stack frame size, so a kernel quietly spilling to local
memory shows up in review rather than in a profile.

Requires `ptxas` but not a GPU. Pass `GPARK_PTXAS=/path/to/ptxas` if it is not on
`PATH`, and `GPARK_ARCH=sm_90` to target something other than the default.

## Tier three: execution

`remote/exec_harness.cu` JITs each golden through the **driver API** and compares
output against a CPU reference, then reports achieved bandwidth.

Why the driver API and not nvcc: routing the harness through nvcc would let the
toolchain normalise the input, so the harness would test nvcc's opinion of the PTX
rather than the PTX. The `.ptx` files are read from `corpus/golden/` at runtime —
`make corpus && make remote-build` is the entire loop, with no build step that could
quietly regenerate the thing under test.

Checks are exact where the operation is exact. Integer ops and the unpack
conversion are bit-compared. SAXPY uses a tolerance, because `fma.rn.f32` and
`alpha*x+y` are *not* required to agree — the GPU contracts the multiply-add into one
rounding, and the CPU may or may not. A harness that demanded bit equality there
would be asserting something PTX does not promise.

Each kernel reports GB/s alongside microseconds. Correctness and speed are separate
questions, and the corpus kernels are bandwidth-bound on purpose so that the second
question has a clean answer.

`remote/graph_bench.cu` measures launch overhead against CUDA Graph replay. Read the
notes at the bottom of that file before quoting it: it establishes the floor a graph
implementation must beat, and it is not a MAGMA comparison.

## Known limits

Stated plainly, because a validation document that claims completeness is worse than
none.

- **No GPU has ever run this.** Tiers two and three are written but unexecuted.
  Treat "validated" in any commit message as "static checks pass".
- **The initialisation check is linear, not a fixpoint.** It tracks a single
  register state and does not merge across branches, so a register written on one
  path and read on another can be reported wrongly in either direction. A real
  dataflow analysis is deferred until a kernel needs it.
- **`:multiple_exits` is misnamed.** The check flags *any* explicit `exit`, not
  exits beyond the first. The intent was to catch redundant exits; the implementation
  catches all of them. Currently every `exit` in the corpus would be reported. It
  has not been fixed because no kernel uses `exit`, and renaming it would change
  behaviour nobody depends on yet.
- **Two backend implementations are already duplicated.** Elixir and Python exist to
  catch drift, which they do — and they cost real effort. If a third backend is
  added before the corpus grows, the maintenance cost may exceed the drift cost.
  `ARCHITECTURE.md` records the reasoning.
- **Sub-byte arithmetic is not validated against hardware.** `u4`/`s4` are packed
  bit containers with no native arithmetic. The validator can prove the bit
  manipulation is self-consistent. It cannot prove a packed multiply is right,
  because packed multiply does not exist yet.
- **`signed_type?`/`unsigned_type?` share an implementation.** `Gpark.IR` delegates
  both to `Gpark.Type.int?/1`, so neither currently distinguishes sign. Unused by
  any kernel, untested, and a trap for whoever writes the first signed kernel.
