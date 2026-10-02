"""Kernel IR: plain dicts and tagged tuples, nothing clever.

Kernels are dicts, instructions are dicts, operands are tuples. There is no SSA
form and no hidden register allocator, because on the kernels gpark cares about
the register pressure *is* the program: a bandwidth-bound quant kernel lives or
dies on how many loads it can keep in flight, and hiding that behind an
allocator would make it impossible to reason about or to tune.

Operand tags mirror the Elixir implementation exactly, since both read the same
corpus:

===========================  ==================================
tag                          meaning
===========================  ==================================
``("reg", type, id)``        typed register
``("pred", id)``             predicate register
``("imm", value)``           integer immediate
``("immf", type, value)``    float immediate, emitted as PTX hex float
``("param", name)``          kernel parameter
``("sreg", name)``           special register, e.g. ``ctaid_x``
``("label", name)``          branch target
``("addr", base, idx, s)``   ``[base+idx*s]``
===========================  ==================================
"""

from __future__ import annotations

from typing import Any

from gpark.type import Packed, reg_class

DEFAULT_TARGET = "sm_80"
DEFAULT_PTX_VERSION = "8.7"

SPACES = ("global", "shared", "local", "const", "param")

SREG_NAMES = {
    "tid_x": "tid.x",
    "tid_y": "tid.y",
    "tid_z": "tid.z",
    "ctaid_x": "ctaid.x",
    "ctaid_y": "ctaid.y",
    "ctaid_z": "ctaid.z",
    "ntid_x": "ntid.x",
    "ntid_y": "ntid.y",
    "ntid_z": "ntid.z",
    "nctaid_x": "nctaid.x",
    "nctaid_y": "nctaid.y",
    "nctaid_z": "nctaid.z",
    "laneid": "laneid",
    "warpid": "warpid",
    "nwarpid": "nwarpid",
    "griddep": "griddep",
}

# Keys every instruction dict carries, in the order the JSON codec writes them.
INSTR_KEYS = ("base", "space", "modifier", "vec", "dtype", "dest", "ops", "pred")


# ---------------------------------------------------------------------------
# Operand constructors
# ---------------------------------------------------------------------------


def reg(type_name: str | Packed, id: int) -> tuple:
    """A typed register operand."""
    return ("reg", type_name, id)


def pred(id: int) -> tuple:
    """A predicate register operand."""
    return ("pred", id)


def imm(value: int) -> tuple:
    """An integer immediate."""
    return ("imm", value)


def immf(type_name: str, value: float) -> tuple:
    """A float immediate. PTX rejects bare float literals, so this becomes hex."""
    return ("immf", type_name, value)


def param(name: str) -> tuple:
    """A kernel parameter operand."""
    return ("param", name)


def sreg(name: str) -> tuple:
    """A special register operand, e.g. ``sreg("ctaid_x")``."""
    return ("sreg", name)


def label(name: str) -> tuple:
    """A branch target."""
    return ("label", name)


def addr(base: tuple, offset_or_index: tuple | int = 0, scale: int | None = None) -> tuple:
    """A memory address operand, ``[base + index*scale]``.

    A bare integer offset is normalised into an ``imm`` operand. The invariant
    that every element of an ``ops`` list is an operand tuple is relied on by the
    validator and the codec alike, and breaking it silently only shows up later as
    an unencodable kernel.
    """
    return ("addr", base, _as_operand(offset_or_index), scale)


def _as_operand(value: Any) -> tuple:
    if isinstance(value, tuple):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return imm(value)
    raise TypeError(f"cannot use {value!r} as an operand")


# ---------------------------------------------------------------------------
# Structural constructors
# ---------------------------------------------------------------------------


def param_decl(name: str, type_name: str, space: str | None = None) -> dict:
    """Declare a kernel parameter."""
    return {"name": name, "type": type_name, "space": space}


def instr(base: str, **opts: Any) -> dict:
    """Build an instruction.

    ``vec`` widens a load or store to ``.v2``/``.v4``, which is how gpark gets
    vectorised memory traffic without asking a compiler to do it.
    """
    unknown = set(opts) - set(INSTR_KEYS)
    if unknown:
        raise TypeError(f"unknown instruction fields: {sorted(unknown)}")

    return {
        "base": base,
        "space": opts.get("space"),
        "modifier": opts.get("modifier"),
        "vec": opts.get("vec"),
        "dtype": opts.get("dtype"),
        "dest": opts.get("dest"),
        "ops": list(opts.get("ops", [])),
        "pred": opts.get("pred"),
    }


def block(label_name: str, instrs: list | None = None, term: dict | None = None) -> dict:
    """Build a basic block, optionally with a terminator."""
    return {"label": label_name, "instrs": list(instrs or []), "term": term}


def kernel(name: str, target: str = DEFAULT_TARGET, ptx_version: str = DEFAULT_PTX_VERSION,
           params: list | None = None, blocks: list | None = None,
           shared: int = 0, maxntid: int | None = None) -> dict:
    """Build a kernel."""
    return {
        "name": name,
        "target": target,
        "ptx_version": ptx_version,
        "params": list(params or []),
        "blocks": list(blocks or []),
        "shared": shared,
        "maxntid": maxntid,
    }


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def all_instrs(k: dict) -> list[dict]:
    """Every instruction in a kernel, including terminators, in program order."""
    return [i for b in k["blocks"] for i in b["instrs"] + ([b["term"]] if b["term"] else [])]


def operand_regs(operand: Any) -> list[tuple[str, str | None, int]]:
    """``(type, register_class, id)`` triples for registers named by an operand.

    Addresses carry registers too: ``[base+idx]`` reads two of them. The validator
    counts those, so the emitter has to as well or a base register used only in an
    address would be missing from the ``.reg`` declarations, which ptxas rejects.
    """
    if not isinstance(operand, tuple):
        return []
    tag = operand[0]
    if tag == "reg":
        return [(operand[1], reg_class(operand[1]), operand[2])]
    if tag == "pred":
        return [("pred", "p", operand[1])]
    if tag == "addr":
        return operand_regs(operand[1]) + operand_regs(operand[2])
    return []


def instr_regs(i: dict) -> list[tuple[str, str | None, int]]:
    """Every register an instruction reads or writes."""
    out: list[tuple[str, str | None, int]] = []
    for op in i["ops"]:
        out += operand_regs(op)
    out += operand_regs(i["dest"])
    out += operand_regs(i["pred"])
    return out


def read_ids(i: dict) -> list[tuple[str, int]]:
    """``(class, id)`` pairs for registers the instruction *reads*."""
    reads: list[tuple[str, int]] = []

    def walk(operand: Any) -> None:
        if not isinstance(operand, tuple):
            return
        tag = operand[0]
        if tag == "reg":
            cls = reg_class(operand[1])
            if cls:
                reads.append((cls, operand[2]))
        elif tag == "pred":
            reads.append(("p", operand[1]))
        elif tag == "addr":
            walk(operand[1])
            walk(operand[2])

    for op in i["ops"]:
        walk(op)
    walk(i["pred"])
    return reads


def dest_ids(i: dict) -> list[tuple[str, int]]:
    """``(class, id)`` pairs for registers the instruction *writes*."""
    dest = i["dest"]
    if not isinstance(dest, tuple):
        return []
    if dest[0] == "reg":
        cls = reg_class(dest[1])
        return [(cls, dest[2])] if cls else []
    if dest[0] == "pred":
        return [("p", dest[1])]
    return []


def max_regs(instrs: list[dict]) -> dict[str, int]:
    """Highest register id used per class."""
    out: dict[str, int] = {}
    for entry in (r for i in instrs for r in instr_regs(i)):
        _type, cls, id = entry
        if cls and id > out.get(cls, 0):
            out[cls] = id
    return out
