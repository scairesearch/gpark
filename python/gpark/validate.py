"""Kernel validation.

Catches the mistakes that are cheap to make and expensive to debug, because
``ptxas`` reports them as one-line assembler errors with no indication of which
kernel line caused them.

What is checked:

* the opcode exists, and its operand/destination counts match the table
* the opcode accepts the modifier and address space given
* result and operand types agree
* parameters, special registers and branch targets exist
* register banks and types agree — ``%r1`` is never used as an f32
* every register is written before it is read, since PTX does not zero registers

What is **not** checked, and is stated plainly rather than implied:

* control flow. Initialisation analysis is linear in program order, so it is an
  approximation for anything with real branches. A CFG walk is deferred.
* liveness, register pressure, alignment, coalescing. Those belong to
  ``Gpark.Mid`` and to measurement, not to a syntax check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gpark import ir, ops
from gpark.type import reg_class


@dataclass(frozen=True)
class Issue:
    kind: str
    message: str
    block: str | None = None
    index: int | None = None


@dataclass
class Context:
    params: set = field(default_factory=set)
    labels: set = field(default_factory=set)
    typed: dict = field(default_factory=dict)
    written: set = field(default_factory=set)
    issues: list = field(default_factory=list)
    exits: int = 0

    def issue(self, kind: str, message: str, block=None, index=None) -> "Context":
        self.issues.append(Issue(kind, message, block, index))
        return self


def check(k: dict) -> tuple[dict | None, list[Issue]]:
    """Validate a kernel.

    Returns ``(kernel, [])`` when valid, or ``(None, issues)``. ``None`` is
    returned deliberately: gpark should not emit PTX it believes to be broken,
    because the entire value of a direct-PTX toolchain is that the output is
    something you can reason about.
    """
    ctx = Context()
    shape_issues = _validate_shape(k)
    ctx.issues.extend(shape_issues)

    if not shape_issues:
        _scan_blocks(k, ctx)

    return (k, []) if not ctx.issues else (None, list(reversed(ctx.issues)))


def _validate_shape(k: dict) -> list[Issue]:
    issues = []
    if not k.get("blocks"):
        issues.append(Issue("no_blocks", "kernel has no blocks", None, None))
    if not isinstance(k.get("params"), list):
        issues.append(Issue("bad_params", "params must be a list", None, None))

    seen_labels = set()
    for block in k["blocks"]:
        label = block["label"]
        if label in seen_labels:
            issues.append(Issue("duplicate_label", f"duplicate block label {label!r}", label, None))
        seen_labels.add(label)

    return issues


def _scan_blocks(k: dict, ctx: Context) -> None:
    ctx.labels = {b["label"] for b in k["blocks"]}
    ctx.params = {p["name"] for p in k["params"]}

    for block in k["blocks"]:
        # The terminator is a real instruction and must be checked like any
        # other. Skipping it once meant a single-instruction block — which puts
        # everything in `term` — validated nothing at all.
        instrs = block["instrs"] + ([block["term"]] if block["term"] else [])

        for index, i in enumerate(instrs):
            _check_instruction(i, block, index, ctx)

        if block["term"]:
            _note_exit(block["term"], ctx)
        else:
            ctx.issue("unterminated_block",
                      f"block {block['label']!r} has no terminator",
                      block["label"], len(block["instrs"]) - 1)


def _note_exit(term: dict, ctx: Context) -> None:
    # Deliberately narrow: only an explicit `exit` is a hazard. A kernel with an
    # early-return block and a `done` block legitimately contains several `ret`s,
    # and PTX is perfectly happy with that, so counting returns would fire on
    # correct code. Multiple `exit`s really do mean divergent threads are being
    # torn down twice.
    ctx.exits += 1
    if term["base"] == "exit":
        ctx.issue("multiple_exits", "kernel contains more than one exit")


def _check_instruction(i: dict, block: dict, index: int, ctx: Context) -> None:
    spec = ops.fetch(i["base"])
    if spec is None:
        ctx.issue("unknown_opcode", f"unknown opcode {i['base']!r}", block["label"], index)
        return

    _check_arity(spec, i, block, index, ctx)
    _check_modifiers(spec, i, block, index, ctx)
    _check_types(spec, i, block, index, ctx)
    _check_references(i, block, index, ctx)
    _check_typing(i, block, index, ctx)
    _check_initialisation(i, block, index, ctx)
    _record_written(i, block, index, ctx)


def _check_arity(spec, i, block, index, ctx) -> None:
    if len(i["ops"]) != spec.nops:
        ctx.issue("wrong_operand_count",
                  f"{i['base']} takes {spec.nops} operand(s), got {len(i['ops'])}",
                  block["label"], index)

    ndest = 1 if i["dest"] is not None else 0
    if ndest != spec.ndest:
        ctx.issue("wrong_destination_count",
                  f"{i['base']} takes {spec.ndest} destination(s), got {ndest}",
                  block["label"], index)


def _check_modifiers(spec, i, block, index, ctx) -> None:
    if spec.modifiers is not None and i["modifier"] not in spec.modifiers:
        allowed = ", ".join(str(m) for m in spec.modifiers)
        ctx.issue("bad_modifier",
                  f"{i['base']} does not take modifier {i['modifier']!r} (allowed: {allowed})",
                  block["label"], index)


def _check_types(spec, i, block, index, ctx) -> None:
    # An empty dtype list means the opcode has no type suffix at all (`bra`,
    # `ret`, `bar`, ...), so there is nothing to check.
    if spec.dtype and i["dtype"] not in spec.dtype:
        allowed = ", ".join(spec.dtype)
        ctx.issue("bad_type",
                  f"{i['base']} does not support type {i['dtype']!r} (allowed: {allowed})",
                  block["label"], index)

    if spec.spaces is not None and i["space"] not in spec.spaces:
        allowed = ", ".join(str(s) for s in spec.spaces)
        ctx.issue("bad_space",
                  f"{i['base']} does not support address space {i['space']!r} (allowed: {allowed})",
                  block["label"], index)

    _check_operand_types(spec, i, block, index, ctx)


def _check_operand_types(spec, i: dict, block: dict, index: int, ctx: Context) -> None:
    """The ``otypes`` column, enforced.

    ``"same"`` means every typed register operand must match the opcode's own
    type. This is what catches ``add.u32`` quietly operating on an f32 register.
    """
    if spec.otypes == "same":
        allowed: tuple | None = spec.dtype
    elif spec.otypes in ("any", "sreg"):
        allowed = None
    else:
        allowed = spec.otypes

    if allowed is None:
        return

    for op in i["ops"]:
        if not isinstance(op, tuple) or op[0] != "reg":
            continue
        if op[1] not in allowed:
            ctx.issue("operand_type_mismatch",
                      f"{i['base']}.{i['dtype']} does not accept a {op[1]!r} operand "
                      f"(allowed: {', '.join(allowed)})",
                      block["label"], index)


def _check_references(i: dict, block: dict, index: int, ctx: Context) -> None:
    for op in i["ops"]:
        if not isinstance(op, tuple):
            continue
        if op[0] == "label":
            if op[1] not in ctx.labels:
                ctx.issue("unknown_label",
                          f"branch to undefined block {op[1]!r}", block["label"], index)
        elif op[0] == "param":
            if op[1] not in ctx.params:
                ctx.issue("unknown_param",
                          f"reference to undefined parameter {op[1]!r}", block["label"], index)
        elif op[0] == "sreg":
            if op[1] not in ir.SREG_NAMES:
                ctx.issue("unknown_sreg",
                          f"unknown special register {op[1]!r}", block["label"], index)


def _check_typing(i: dict, block: dict, index: int, ctx: Context) -> None:
    for type_name, cls, id in ir.instr_regs(i):
        if cls is None:
            continue
        key = (cls, id)
        existing = ctx.typed.get(key)
        if existing is None:
            ctx.typed[key] = type_name
        elif existing != type_name:
            ctx.issue("register_type_conflict",
                      f"%{cls}{id} used as both {existing} and {type_name}",
                      block["label"], index)


def _check_initialisation(i: dict, block: dict, index: int, ctx: Context) -> None:
    # Only `written` suppresses this. Consulting the typed table would be
    # circular: _check_typing runs first and records every register it *sees*, so
    # a register that was only ever read would look initialised.
    for cls, id in ir.read_ids(i):
        if (cls, id) not in ctx.written:
            ctx.issue("uninitialised_register",
                      f"read of never-written register %{cls}{id}", block["label"], index)


def _record_written(i: dict, block: dict, index: int, ctx: Context) -> None:
    for cls, id in ir.dest_ids(i):
        ctx.written.add((cls, id))
