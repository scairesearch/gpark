# gpark

Direct PTX code generation from Elixir and Python, built around one claim: a GPU
kernel is slow for one of four reasons — bandwidth, latency, occupancy, or launch
overhead — and you cannot fix any of them through an abstraction that hides the
instructions.

```elixir
Gpark.Kernels.VecAddF32.build()
|> Gpark.Validate.validate!()
|> Gpark.PTX.emit()
|> then(&IO.write(IO.iodata_to_binary(&1)))
```

```python
from gpark import ir, ops, ptx, validate

kernel = ir.kernel("vec_add_f32", params=[...], blocks=[...])
validate.validate(kernel)
print(ptx.emit(kernel))
```

## Status

**No GPU has ever run this.** Everything below tier one is written and unexecuted.
Treat that as the current state rather than a footnote.

| Tier | Runs on | Proves |
|---|---|---|
| Static — `make test` | any machine | PTX is well-formed, internally consistent, identical across both implementations |
| Assembly — `make remote` | CUDA toolkit, no GPU | real hardware accepts it; register/spill numbers |
| Execution — `make remote-build` | NVIDIA host + GPU | correct results, achieved bandwidth |

48 opcodes, 30 types, 4 kernels, 31 Elixir tests, 4 Python tests, all static checks
green.

## The corpus

Four kernels, chosen to cover four capability families. All are bandwidth-bound with
near-zero arithmetic intensity, on purpose: a bandwidth-bound kernel has one expected
answer, so a bad number means the addressing or the registers are wrong rather than
that a scheduling subtlety bit.

| Kernel | Forces |
|---|---|
| `vec_add_f32` | addressing, 64-bit widening, the control case |
| `saxpy_f32` | fused arithmetic, the tolerance boundary |
| `reduce_sum_f32` | `shfl.sync` butterflies, warp collectives, a lane-dependent store |
| `unpack_u4_f32` | packed `u4` bit containers, multi-word expansion, sub-byte quantisation |

`unpack_u4_f32` is the reason the type system splits *containers* (`b1`…`b64`) from
*element formats* (`u4`, `s4`, `f16`, `bf16`, `e4m3`, …). PTX has no 4-bit float
arithmetic, so a `u4` value lives in a `.b32` register and every operation on it is a
32-bit op plus a mask. Anything that treats it as occupying 4 bits is wrong about
occupancy, registers, and memory traffic.

## Running it

```sh
make test        # both implementations: format, warnings-as-errors, tests, goldens
make corpus      # regenerate specs and goldens from the Elixir kernels
make remote      # assemble every golden with ptxas, fail on spills
make remote-build # build the CUDA harnesses (needs nvcc + libcuda)
```

The Python suite shells out to `mix run` to compare hashed opcode tables, so
`elixir/` and `python/` cannot drift apart silently. That is the whole reason there
are two implementations.

## Documentation

Read in this order; each earns the next.

- [`docs/SYSTEMS.md`](docs/SYSTEMS.md) — why this project exists. Start here.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how the layers fit
- [`docs/PTX-SUBSET.md`](docs/PTX-SUBSET.md) — what it can emit, and what it refuses to
- [`docs/KERNELS.md`](docs/KERNELS.md) — the four kernels and why each exists
- [`docs/VALIDATION.md`](docs/VALIDATION.md) — what is checked, and what is not
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — what is next, and what is deliberately not
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — decisions with reasoning and trade-offs
- [`docs/TAICHI-NOTES.md`](docs/TAICHI-NOTES.md) — prior art, and what not to copy

## Design commitments

**No SSA in v0.1, hand-assigned registers.** When a bandwidth kernel sits at 40% of
roofline the cause is nearly always how many loads are in flight per thread, which
is decided by the register assignment. An allocator optimises for something else and
hands you a correct kernel you cannot reason about. The cost is real — no loops, so
`unpack_u4_f32` unrolls eight shift/mask/convert/store blocks — and the trade is
deliberate.

**Spills are a hard failure.** Registers are assigned by hand, so a spill means the
assignment could not hold the working set and ptxas quietly hid it by spilling to
local memory. `make remote` fails rather than accepting one.

**Byte-stable goldens.** Output is deterministic to the byte, so it can eventually be
diffed against real `nvcc` output to track divergence from the actual compiler. "Stable
modulo whitespace" is not a contract.

**Byte equality is not correctness.** `unpack_u4_f32` shipped with its bounds guard
emitted *after* the work it guarded, so every out-of-range lane performed a 32-byte
out-of-bounds write. The golden was self-consistent, the validator passed, every test
was green — both had been handed valid PTX that was merely not the intended program.
Byte comparison detects change, not incorrectness. Tests now assert properties
directly, and a new assertion is checked against the code it should catch before
being trusted.

**Vendor neutrality needs a gate.** "Supports three backends" is a claim; a parity
gate where every backend passes the same corpus and CI fails on a silently lost
capability is what would make it a fact. Not built yet — only CUDA exists.

## Licence

AGPL-3.0. `LICENSE` currently carries the notice and a pointer; the canonical text
must be added before any public release.
