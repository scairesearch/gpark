"""The capability gate: check a backend can handle a kernel before emitting it.

The Python half of :mod:`gpark.backend`'s counterpart in Elixir. Both implementations
expose the same contract because the project's first principle is that neither
language is the reference -- whichever one you use, a backend you cannot satisfy
should refuse the kernel rather than quietly emit something slower and still correct.

That failure mode is the one worth guarding. A kernel that falls back to a
three-instruction sequence where the backend wanted one still produces the right
answer, still passes the golden, and is three times slower. Nothing except an
explicit capability check will ever tell you.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from gpark import ops as ops_mod
from gpark import type as type_mod
from gpark.ir import all_instrs, instr_regs

# A capability problem is a pair: the kind, and the thing that is missing. Kept as
# tuples rather than exceptions so a caller can inspect every missing capability at
# once instead of fixing one and re-running.
Issue = tuple[str, Any]


@runtime_checkable
class Backend(Protocol):
    """What a gpark backend implements. See :mod:`gpark.ptx` for the only one."""

    def name(self) -> str:
        ...

    def ops(self) -> list[str]:
        """Opcode base names this backend can emit."""

    def types(self) -> list[str]:
        """Types this backend can represent."""

    def emit(self, k: dict) -> str:
        ...

    def check(self, k: dict) -> tuple[dict, list]:
        """Structural validation, independent of this backend's capabilities."""


def required_ops(k: dict) -> set[str]:
    """Every opcode base used by ``k``, across instructions and terminators."""
    return {i["base"] for i in all_instrs(k)}


def required_types(k: dict) -> set[str]:
    """Every type ``k`` mentions, including in operands and parameter declarations."""
    found: set[str] = set()

    for i in all_instrs(k):
        # `None` is what an untyped instruction -- `ret`, `bra` -- carries. Including
        # it would report every terminator as an unsupported type named None.
        for key in ("dtype", "srctype"):
            value = i.get(key)
            if isinstance(value, str):
                found.add(value)
        for type_name, _klass, _id in instr_regs(i):
            found.add(type_name)

    for decl in k.get("params", []):
        found.add(decl["type"])

    return found


def supports_op(backend: Any, base: str) -> bool:
    """Whether ``backend`` can emit ``base``."""
    return base in backend.ops()


def supports_type(backend: Any, type_name: str) -> bool:
    """Whether ``backend`` can represent ``type_name``."""
    return type_name in backend.types()


def require(backend: Any, k: dict) -> tuple[dict | None, list[Issue]]:
    """Check ``backend`` can handle ``k``.

    Returns ``(k, [])`` when it can, and ``(None, issues)`` when it cannot. The
    kernel is returned unchanged rather than transformed: this is a gate, not a
    lowering step, so a caller that gets a kernel back knows nothing was rewritten on
    the way through.
    """
    supported_ops = backend.ops()
    supported_types = backend.types()

    issues: list[Issue] = [
        ("unsupported_op", base)
        for base in sorted(required_ops(k))
        if base not in supported_ops
    ]
    issues += [
        ("unsupported_type", type_name)
        for type_name in sorted(required_types(k))
        if type_name not in supported_types
    ]

    return (None, issues) if issues else (k, [])


def emit_or_raise(backend: Any, k: dict) -> str:
    """:func:`require` then emit, raising ``ValueError`` if the backend cannot.

    Raising is deliberate. A missing capability is a programming error that should
    stop the run, not produce output that is correct and unexpectedly slow.
    """
    checked, issues = require(backend, k)
    if checked is None:
        detail = ", ".join(f"opcode {v}" if kind == "unsupported_op" else f"type {v}"
                           for kind, v in issues)
        raise ValueError(
            f"{backend.name()} cannot emit {k['name']}: it has no {detail}. "
            "Emit a slower sequence explicitly, or extend the backend."
        )
    return backend.emit(checked)


def describe(backend: Any, k: dict) -> str:
    """A human-readable list of what ``k`` needs that ``backend`` lacks. Empty if none."""
    _checked, issues = require(backend, k)
    if not issues:
        return f"{backend.name()} can emit {k['name']}"
    return f"{backend.name()} cannot emit {k['name']}: " + ", ".join(
        f"{kind} {value}" for kind, value in issues
    )
