"""The typed opcode table.

Data rather than IR, and kept as a flat list of tuples so the entries stay
readable. Both the validator and the emitter read this one table, which is what
stops them drifting apart.

Each entry is ``(name, parts, dtype, ndest, nops, otypes, extras)`` where:

``parts``
    Dotted components the emitter renders, in PTX order: ``name``, ``space``,
    ``sync``, ``modifier``, ``vec``, ``dtype``.
``dtype``
    Accepted data/result types; empty means "not applicable".
``ndest``, ``nops``
    Destination and operand counts.
``otypes``
    ``"same"`` (every operand type must equal ``dtype``), ``"any"``, or a list
    of allowed types.
``spaces``, ``modifiers``
    Permitted address spaces and modifiers, when the opcode has them. A
    ``modifiers`` list of ``[None]`` means the opcode takes no modifier.
``sync``
    True when the opcode participates in warp-synchronous execution.

See ``docs/PTX-SUBSET.md`` for what v0.1 covers and what is deliberately
deferred.
"""

from __future__ import annotations

from typing import Any, NamedTuple

_INT32 = ("s32", "u32")
_INT64 = ("s64", "u64")
_INT_ALL = _INT32 + _INT64
_FLOATS = ("f32", "f64")
_MEM_TYPES = _INT_ALL + _FLOATS + ("b64",)

_LOAD_SPACES = ("global", "shared", "local", "const", "param")
_MEM_SPACES = ("global", "shared", "local", "param")
_ATOMIC_SPACES = ("global", "shared")


class OpSpec(NamedTuple):
    """One row of the opcode table."""

    name: str
    parts: tuple[str, ...]
    dtype: tuple[str, ...]
    ndest: int
    nops: int
    otypes: Any
    spaces: tuple[str, ...] | None = None
    modifiers: tuple[str | None, ...] | None = None
    sync: bool = False


_SPECS: tuple[OpSpec, ...] = (
    # data movement / conversion
    OpSpec("mov", ("name", "dtype"), _MEM_TYPES + ("pred",), 1, 1, "same", sync=True),
    OpSpec("cvt", ("name", "dtype"), _MEM_TYPES + _FLOATS, 1, 1, "any"),
    # integer arithmetic
    OpSpec("add", ("name", "dtype"), _MEM_TYPES, 1, 2, "same", sync=True),
    OpSpec("sub", ("name", "dtype"), _MEM_TYPES, 1, 2, "same", sync=True),
    OpSpec("mul", ("name", "dtype"), _MEM_TYPES, 1, 2, "same", sync=True),
    OpSpec("mad", ("name", "dtype"), _INT_ALL, 1, 3, "same", sync=True),
    OpSpec("neg", ("name", "dtype"), _MEM_TYPES, 1, 1, "same"),
    OpSpec("abs", ("name", "dtype"), _INT_ALL, 1, 1, "same"),
    OpSpec("min", ("name", "dtype"), _INT_ALL + _FLOATS, 1, 2, "same", sync=True),
    OpSpec("max", ("name", "dtype"), _INT_ALL + _FLOATS, 1, 2, "same", sync=True),
    OpSpec("rem", ("name", "dtype"), _INT_ALL, 1, 2, "same", sync=True),
    OpSpec("div", ("name", "dtype"), _INT_ALL, 1, 2, "same", sync=True),
    OpSpec("shl", ("name", "dtype"), _INT_ALL, 1, 2, "same", sync=True),
    OpSpec("shr", ("name", "dtype"), _INT_ALL, 1, 2, "same", sync=True),
    OpSpec("and", ("name", "dtype"), _MEM_TYPES, 1, 2, "same", sync=True),
    OpSpec("or", ("name", "dtype"), _MEM_TYPES, 1, 2, "same", sync=True),
    OpSpec("xor", ("name", "dtype"), _MEM_TYPES, 1, 2, "same", sync=True),
    OpSpec("not", ("name", "dtype"), _MEM_TYPES + ("pred",), 1, 1, "same"),
    OpSpec("popc", ("name", "dtype"), _INT_ALL, 1, 1, "same"),
    OpSpec("clz", ("name", "dtype"), _INT_ALL, 1, 1, "same"),
    OpSpec("brev", ("name", "dtype"), _INT_ALL, 1, 1, "same"),
    # 32-bit widening helpers — the backbone of 64-bit address arithmetic
    OpSpec("mul.wide", ("name", "dtype"), _INT32, 1, 2, _INT32, sync=True),
    OpSpec("mad.lo", ("name", "dtype"), _INT_ALL, 1, 3, "same", sync=True),
    OpSpec("mad.hi", ("name", "dtype"), _INT_ALL, 1, 3, "same", sync=True),
    # floating point. PTX has no add.f.f32: the .f32 suffix already implies
    # float, so these share the integer opcodes. Only fma needs a modifier.
    OpSpec("fma", ("name", "modifier", "dtype"), _FLOATS, 1, 3, _FLOATS,
           sync=True, modifiers=("rn", "approx")),
    OpSpec("rcp", ("name", "dtype"), ("f32", "f64"), 1, 1, _FLOATS, sync=True),
    OpSpec("rsqrt", ("name", "dtype"), ("f32",), 1, 1, _FLOATS, sync=True),
    OpSpec("sqrt", ("name", "dtype"), ("f32", "f64"), 1, 1, _FLOATS, sync=True),
    # comparison and predication
    OpSpec("setp", ("name", "modifier", "dtype"), _MEM_TYPES, 1, 2, "same",
           modifiers=("eq", "ne", "lt", "le", "gt", "ge")),
    OpSpec("selp", ("name", "dtype"), _MEM_TYPES, 1, 3, "any", sync=True),
    OpSpec("slct", ("name", "dtype"), _INT_ALL, 1, 3, "same", sync=True),
    # memory
    OpSpec("ld", ("name", "space", "modifier", "vec", "dtype"), _MEM_TYPES, 1, 1, "any",
           spaces=_LOAD_SPACES,
           modifiers=(None, "nc", "volatile", "cv")),
    OpSpec("st", ("name", "space", "modifier", "vec", "dtype"), _MEM_TYPES, 0, 2, "any",
           spaces=_MEM_SPACES,
           modifiers=(None, "wb", "cg", "cs", "wt")),
    OpSpec("prefetch", ("name", "space"), (), 0, 1, "any",
           spaces=("global", "local")),
    # atomics
    OpSpec("atom", ("name", "space", "modifier", "dtype"), _MEM_TYPES, 1, 2, "any",
           spaces=_ATOMIC_SPACES,
           modifiers=("add", "sub", "min", "max", "and", "or", "xor", "exch", "cas")),
    OpSpec("red", ("name", "space", "modifier", "dtype"), _MEM_TYPES, 0, 2, "any",
           spaces=_ATOMIC_SPACES,
           modifiers=("add", "sub", "min", "max", "and", "or", "xor")),
    # warp-level
    OpSpec("shfl", ("name", "sync", "modifier", "dtype"), ("b32", "u32", "f32", "f64"),
           1, 3, "any", modifiers=("down", "up", "bfly", "idx")),
    OpSpec("vote", ("name", "sync", "modifier", "dtype"), ("pred", "u32"), 1, 2, "any",
           modifiers=("any", "all", "ballot")),
    OpSpec("activemask", ("name", "dtype"), ("b32",), 1, 0, "any"),
    OpSpec("bar", ("name", "sync", "modifier"), (), 0, 1, "any",
           modifiers=(None, "arrive", "red")),
    # control
    OpSpec("bra", ("name",), (), 0, 1, ("label",)),
    OpSpec("brx", ("name", "modifier", "dtype"), (), 0, 2, ("label",),
           modifiers=(None, "uni")),
    OpSpec("ret", ("name",), (), 0, 0, ()),
    OpSpec("call.uni", ("name",), (), 0, 0, ("label",)),
    OpSpec("exit", ("name",), (), 0, 0, ()),
    OpSpec("s2r", ("name", "dtype"), _INT_ALL, 1, 1, "sreg"),
    # A nil entry means "any space", including the implied default.
    OpSpec("cvta", ("name", "space", "dtype"), _MEM_TYPES, 1, 1, "any",
           spaces=_LOAD_SPACES + (None,)),
    OpSpec("nop", ("name",), (), 0, 0, ()),
)

_TABLE: dict[str, OpSpec] = {spec.name: spec for spec in _SPECS}


def table() -> dict[str, OpSpec]:
    """The whole opcode table, keyed by base name."""
    return _TABLE


def fetch(name: str) -> OpSpec | None:
    """Look up one opcode spec, or None."""
    return _TABLE.get(name)


def names() -> list[str]:
    return sorted(_TABLE)


def sync_ops() -> list[str]:
    """Warp-synchronous opcodes. The scheduler must not move these across a barrier."""
    return sorted(name for name, spec in _TABLE.items() if spec.sync)


def memory_ops() -> tuple[str, ...]:
    """Memory-addressed opcodes, which are ordering-sensitive."""
    return ("ld", "st", "prefetch", "atom", "red")


def digest() -> str:
    """A canonical one-line-per-opcode rendering of the whole table.

    Used to cross-check this implementation against the Elixir one. Comparing a
    text digest rather than the in-memory structures keeps the check honest: the
    two tables are separate implementations on purpose, so the test has to compare
    something both sides agree on how to produce without sharing any code.
    """
    lines = []
    for name in names():
        spec = _TABLE[name]
        otypes = spec.otypes
        if otypes == "same":
            otypes_text = "same"
        elif otypes == "any":
            otypes_text = "any"
        elif otypes == "sreg":
            otypes_text = "sreg"
        else:
            otypes_text = ",".join(sorted(otypes))
        lines.append("|".join([
            spec.name,
            # `parts` order is semantic (it is PTX's dotted-part order). The rest
            # are sets, so they are sorted: the two implementations are free to
            # list the same types in a different order without that being drift.
            ",".join(spec.parts),
            ",".join(sorted(spec.dtype)),
            str(spec.ndest),
            str(spec.nops),
            otypes_text,
            _join_optional(spec.spaces),
            _join_optional(spec.modifiers),
            "true" if spec.sync else "false",
        ]))
    return "\n".join(lines)


def _join_optional(values) -> str:
    if values is None:
        return "-"
    # nil is a real entry in some of these lists ("any space", "no modifier"), so
    # it sorts to the front and stays distinguishable from an empty list.
    return ",".join(sorted("-" if v is None else v for v in values))
