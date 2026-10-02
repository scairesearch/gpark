# gpark (Python)

The Python half of [gpark](../README.md). An independent implementation of the
same kernel IR and PTX emitter as the Elixir half — not a binding to it.

## The one rule

For the same kernel spec in `corpus/specs/`, both implementations must emit the
**same bytes** of PTX. That is enforced by `tests/test_corpus.py`, which reads
the goldens the Elixir side produced. The two opcode tables are also
cross-checked against each other on every test run.

If the formatting were merely similar, every diff against real `nvcc` output
would be noise, and the shared corpus would prove nothing.

## Usage

```python
from gpark import addr, block, check, emit, imm, instr, kernel, param, param_decl, reg

k = kernel("scale_f32", params=[param_decl("x", "u64"), param_decl("n", "u32")], blocks=[
    block("entry", [
        instr("mov", dtype="u32", dest=reg("u32", 1), ops=[imm(0)]),
        instr("ret", ),
    ]),
])

ok, issues = check(k)
if ok:
    print(emit(k))
else:
    for i in issues:
        print(f"{i.kind}: {i.message}")
```

## Tests

```sh
make -C .. test      # both Elixir and Python suites
make test            # just this one
```

The opcode cross-check shells out to `mix`; if Elixir is not installed that one
test skips rather than failing, and the golden check still runs.

## Status

The backend compiles, validates and emits PTX. Nothing has been run on a GPU
yet — see `../docs/ROADMAP.md`.
