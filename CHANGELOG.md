# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Monorepo layout: `elixir/`, `python/`, `corpus/`, `kernels/`, `remote/`, `docs/`
- `Gpark.IR` (Elixir) — kernel IR structs plus a typed opcode table
- `Gpark.PTX` (Elixir) — PTX module/function emitter
- `Gpark.Validate` (Elixir) — IR type-checker and PTX structural validator
- `gpark.ir`, `gpark.ptx`, `gpark.validate` (Python) — independent implementation
- `corpus/` — kernel spec format and byte-exact golden `.ptx` fixtures
- `remote/` — `ptxas_check.sh` and driver-API execution harness for an NVIDIA host
- `docs/` — CONTEXT, ROADMAP, ARCHITECTURE, QUANT, PTX-SUBSET, VALIDATION,
  DECISIONS, WORKLOG, GLOSSARY

### Notes
- v0.1 emits PTX only. Metal (P4) and AMDGCN (P5) backends are roadmap items.
- No kernel has yet recorded a measured speedup; see `docs/ROADMAP.md` phase P2.