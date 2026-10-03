"""PTX emitter.

Renders a kernel to PTX text. The output style deliberately imitates ``nvcc`` —
tab indentation, contiguous register ranges like ``%f<8>``, hex float
immediates — so ``ptxas`` accepts it and so goldens in ``corpus/`` can be diffed
against what the CUDA toolchain produces for an equivalent kernel.

The emitter is *unopinionated about correctness*: it will happily render a kernel
the validator rejects, because getting close to the metal is the point. Validate
before you reach for ``ptxas``. See ``docs/VALIDATION.md``.
"""

from __future__ import annotations

import struct

from gpark import ir, ops as ops_mod, type as type_mod

# PTX has no fixed register classes, but nvcc's ordering is a good convention:
# predicates, then 32-bit, then 64-bit, then the float banks.
_CLASS_ORDER = {"p": 0, "r": 1, "rd": 2, "f": 3, "fd": 4}

_TAB = "\t"

# Mirrors the Elixir op_spec(order): match a class prefix exactly, longest first,
# so "fd" never matches "f" and silently renames %fd1 to %f1.
_CLASS_PREFIXES = {"fd": "fd", "rd": "rd", "p": "p", "r": "r", "f": "f"}


def emit(k: dict) -> str:
    """Render a kernel to PTX text."""
    return "".join([_header(k), _signature(k), _body(k)])


def _header(k: dict) -> str:
    return f".version {k['ptx_version']}\n.target {k['target']}\n.address_size 64\n"


def _signature(k: dict) -> str:
    name = k["name"]
    params = k["params"]
    if not params:
        return f".visible .entry {name}()\n"
    lines = ",\n".join(
        f"{_TAB}.param .{p['type']} {p['name']}" for p in params
    )
    return f".visible .entry {name}(\n{lines}\n)\n"


def _body(k: dict) -> str:
    instrs = ir.all_instrs(k)
    return "".join([
        "{\n",
        "".join(_declarations(instrs)),
        _shared_decl(k),
        "".join(_block(b) for b in k["blocks"]),
        "}\n",
    ])


def _declarations(instrs: list[dict]) -> list[str]:
    """``.reg`` declarations sized from actual usage.

    A kernel never allocates a register it does not need: on a register-starved
    quant kernel one spare vector slot is a whole extra load in flight.
    """
    if not instrs:
        return []

    seen: dict[tuple[str, str], set[int]] = {}
    for i in instrs:
        for type_name, cls, id in ir.instr_regs(i):
            if cls is not None:
                seen.setdefault((cls, type_name), set()).add(id)

    ordered = sorted(seen, key=lambda key: (_CLASS_ORDER[key[0]], key[0], key[1]))
    # One declaration per register, not the `%rd1<4>` vector form. A real
    # ptxas rejects the vector form, and unhelpfully: it reports the first
    # *instruction* rather than the declaration, so the error cascades through
    # the whole kernel and reads like an addressing bug. The Elixir side makes
    # the same choice, and these two must agree byte for byte.
    return [
        f"{_TAB}.reg .{type_name} {_register(cls, [id])};\n"
        for cls, type_name in ordered
        for id in sorted(seen[(cls, type_name)])
    ]


def _shared_decl(k: dict) -> str:
    # Shared scratch is declared in the body, which is where PTX requires it.
    if not k["shared"]:
        return ""
    return f"{_TAB}.extern .shared .align 4 .b8 __gpark_shared[{k['shared']}];\n"


def _block(b: dict) -> str:
    out = [f"$L__{b['label']}:\n"]
    for i in b["instrs"]:
        out.append(f"{_TAB}{_line(i)}\n")
    if b["term"]:
        out.append(f"{_TAB}{_line(b['term'])}\n")
    return "".join(out)


def _line(i: dict) -> str:
    return f"{_guard(i['pred'])}{opcode(i)}{_join_args(i['dest'], i['ops'])};"


# PTX puts exactly one space after the opcode and separates the optional
# destination from the operand list with ", ", matching nvcc so goldens stay
# diffable against real CUDA output rather than against our own formatting.
def _join_args(dest, ops_list) -> str:
    if dest is None and not ops_list:
        return ""
    if dest is None:
        return " " + ", ".join(_operand(o) for o in ops_list)
    if not ops_list:
        return " " + _operand(dest)
    return " " + _operand(dest) + ", " + ", ".join(_operand(o) for o in ops_list)


def _guard(pred) -> str:
    # "@%p" prefixes a predicated statement. PTX has no "branch if false", so a
    # negated branch is expressed by negating the predicate explicitly upstream.
    return "" if pred is None else f"@{_operand(pred)} "


# ---------------------------------------------------------------------------
# Opcodes
# ---------------------------------------------------------------------------


def opcode(i: dict) -> str:
    """Render an instruction's opcode string, dotted parts in PTX order."""
    spec = ops_mod.fetch(i["base"])
    if spec is None:
        raise ValueError(f"unknown opcode {i['base']!r}")

    out = []
    for part in spec.parts:
        if part == "name":
            out.append(i["base"])
        elif part == "space":
            out.append(f".{i['space']}" if i["space"] else "")
        elif part == "sync":
            out.append(".sync")
        elif part == "modifier":
            out.append(f".{i['modifier']}" if i["modifier"] else "")
        elif part == "vec":
            out.append(f".v{i['vec']}" if i["vec"] else "")
        elif part == "dtype":
            out.append(f".{i['dtype']}" if i["dtype"] else "")
        elif part == "srctype":
            out.append(f".{i['srctype']}" if i["srctype"] else "")
    return "".join(out)


def _operand(o) -> str:
    if not isinstance(o, tuple):
        raise TypeError(f"not an operand: {o!r}")

    tag = o[0]
    if tag == "reg":
        return _one(_reg_class(o[1]), o[2])
    if tag == "pred":
        return f"%p{o[1]}"
    if tag == "imm":
        return str(o[1])
    if tag == "immf":
        return hex_float(o[2], o[1])
    if tag == "param":
        return f"[{o[1]}]"
    if tag == "sreg":
        return "%" + ir.SREG_NAMES[o[1]]
    if tag == "label":
        return f"$L__{o[1]}"
    if tag == "addr":
        base, idx, scale = o[1], o[2], o[3]
        if scale is None and idx == ("imm", 0):
            return f"[{_operand(base)}]"
        if scale is None and idx[0] == "imm":
            return f"[{_operand(base)}+{idx[1]}]"
        # A register offset is not expressible as a PTX addressing mode. PTX
        # ld/st accept [reg] or [reg+imm] only; register+register does not
        # parse. An index unknown until launch cannot become an immediate
        # either, so the sum has to be built in a register first:
        #
        #     add.u64 <addr_reg>, <base>, <off>;
        #     ld.global.f32 <dst>, [<addr_reg>];
        #
        # Refusing here keeps the mistake at the line that made it, instead of
        # emitting text that passes Gpark.Validate -- it only checks the IR --
        # and then fails at ptxas far from the cause.
        if scale is None:
            raise ValueError(
                f"register offsets cannot be rendered as a PTX addressing mode: "
                f"{_operand(base)}+{_operand(idx)}. "
                f"PTX ld/st accept [reg] or [reg+imm] only. Materialise the "
                f"address into an explicitly allocated register instead."
            )
        raise ValueError(
            f"strided addresses ({_operand(base)}+{_operand(idx)}*{scale}) cannot "
            f"be rendered as a PTX addressing mode. PTX ld/st accept [reg] or "
            f"[reg+imm] only. Widen the index, then materialise the address into "
            f"an explicitly allocated register."
        )
    raise ValueError(f"unknown operand tag {tag!r}")


def _reg_class(type_name) -> str:
    from gpark.type import reg_class

    cls = reg_class(type_name)
    if cls is None:
        raise ValueError(f"type {type_name!r} has no register class")
    return cls


def hex_float(value: float, type_name: str) -> str:
    """Render a PTX hex float immediate.

    PTX rejects bare floating-point literals in most contexts, so f32 is written
    as ``0f`` plus 8 hex digits and f64 as ``0d`` plus 16.
    """
    if type_name == "f32":
        (bits,) = struct.unpack(">I", struct.pack(">f", value))
        return "0f" + f"{bits:08X}"
    if type_name == "f64":
        (bits,) = struct.unpack(">Q", struct.pack(">d", value))
        return "0d" + f"{bits:016X}"
    raise ValueError(f"no hex float encoding for {type_name!r}")


# ---------------------------------------------------------------------------
# Registers and labels
# ---------------------------------------------------------------------------


def _one(cls: str, id: int) -> str:
    """A single register: ``%f3``."""
    return "%" + _CLASS_PREFIXES[cls] + str(id)


def _register(cls: str, ids: list[int]) -> str:
    """A ``.reg`` declaration listing: ``%f1<3>``."""
    prefix = "%" + _CLASS_PREFIXES[cls]
    return ", ".join(prefix + _run_text(first, last) for first, last in _compress(ids))


def _run_text(first: int, last: int) -> str:
    return str(first) if first == last else f"{first}<{last - first + 1}>"


def _compress(ids: list[int]) -> list[tuple[int, int]]:
    """Compress a sorted id list into PTX runs like ``%r<4><2>``.

    Declarations then look like nvcc output rather than a wall of singles.
    """
    runs: list[tuple[int, int]] = []
    current: tuple[int, int] | None = None

    for id in sorted(ids):
        if current is None:
            current = (id, id)
        elif id <= current[1] + 1:
            current = (current[0], id)
        else:
            runs.append(current)
            current = (id, id)

    if current is not None:
        runs.append(current)
    return runs


# --- Gpark.Backend capability gate -------------------------------------------
#
# The Elixir backend declares these as callbacks. Both languages expose the same
# contract, so a Python user gets the same refusal rather than a silent fallback.


def name() -> str:
    """A short name for this backend, for diagnostics."""
    return "ptx"


def ops() -> list[str]:
    """Every opcode base gpark can render.

    Taken from the table rather than hand-kept, so it cannot drift from what
    :func:`emit` is willing to spell.
    """
    return ops_mod.names()


def types() -> list[str]:
    """Every type gpark can represent.

    Includes ``s2 u2 s4 u4``, which have no direct PTX spelling -- ``ptx_type`` returns
    ``None`` for a bare ``:u4`` -- but are representable packed into a container or
    widened before arithmetic.
    """
    return type_mod.all_types()


def check(k: dict) -> tuple[dict | None, list]:
    """Structural validation, delegated to :mod:`gpark.validate`.

    Separate from the capability gate on purpose: this reports a malformed kernel,
    where the gate reports a valid kernel this backend cannot handle. A kernel can
    pass one and fail the other.
    """
    from gpark import validate

    return validate.check(k)
