# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-02

First release. Everything here is statically verified; **nothing in it has run on a
GPU**. See "Known limits" in `docs/VALIDATION.md` before drawing any conclusion from
this release.

### Added

**Backend, Elixir**
- `Gpark.IR` — kernel IR: blocks, instructions, explicit register identifiers, no SSA
- `Gpark.Ops` — 48-opcode typed table (arity, operand types, address spaces, modifiers)
- `Gpark.Type` — 30 types: 26 native PTX plus the non-native logical `s2 u2 s4 u4`
- `Gpark.PTX` — emitter producing byte-stable PTX (`.version 8.7`, `.target sm_80`)
- `Gpark.Validate` — structural, type, bit-container, initialisation and exit checks
- `Gpark.IR.JSON` — canonical serialisation for the corpus format
- `Gpark.Backend` — the backend contract and `require!/2`, which refuses a kernel the
  backend cannot emit and names every missing opcode and type at once
- `Gpark.Opt.Simplify` — fixpoint simplification: unreachable blocks and dead
  side-effect-free instructions, run until no rule fires

**Backend, Python** — an independent implementation of the same IR, emitter, validator
and capability gate in `python/gpark/`. Not a binding to the Elixir one; both are held
to identical bytes for the same spec.

**Corpus** — four bandwidth-bound kernels with byte-exact golden PTX:
`vec_add_f32`, `saxpy_f32`, `reduce_sum_f32`, `unpack_u4_f32`.

**Tooling**
- `Makefile` — `make test` runs both suites; `make corpus` regenerates goldens;
  `make remote-build` builds the CUDA harnesses
- `remote/ptxas_check.sh` — architecture assembly with spills as a hard failure
- `remote/exec_harness.cu` — driver-API JIT, CPU reference, CUDA-event timing
- `remote/graph_bench.cu` — plain launches vs stream capture vs hand-built `cuGraph`,
  normalised to microseconds per kernel launch
- GitHub Actions CI: static suites on every push, plus a GPU-less `ptxas` spill gate

**Docs** — `ARCHITECTURE`, `SYSTEMS`, `KERNELS`, `PTX-SUBSET`, `VALIDATION`, `ROADMAP`,
`DECISIONS`, `TAICHI-NOTES`, `LICENSING`.

### Fixed

- `unpack_u4_f32` emitted its bounds guard *after* the work it guarded, so every
  out-of-range lane performed a 32-byte out-of-bounds write. Property tests now assert
  that stores follow their guard.
- Signedness predicates counted floats as integers; sub-byte types answered `nil` from
  `width/1`, `kind/1` and `sign/1` because each consulted the native table alone.
- The validator's exit counter fired on the *first* `exit` rather than the second and
  later ones.

### Deliberate limits, documented rather than hidden

- No GPU has ever run this. There is no performance claim to make yet.
- The CUDA harnesses have only been syntax-checked against hand-written stub headers,
  never compiled against a real toolkit.
- `Gpark.Opt.Simplify` never removes a load: a dead load can still fault, and removing
  it would hide exactly the out-of-bounds bug above.
- `reduce_sum_f32` trusts the caller for a warp-multiple `n` and reads out of bounds
  otherwise.
- No corpus kernel uses `Gpark.Type.Packed`; the packed path is exercised only by
  doctests.

### Licence

AGPL-3.0-or-later, with the canonical licence text at the repository root. Provenance
and a SHA-256 for the fetched text are recorded in `docs/LICENSING.md`.

## [Unreleased]

Nothing yet.