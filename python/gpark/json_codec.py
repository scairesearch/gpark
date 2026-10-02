"""JSON codec for kernel specs.

The wire format is shared with the Elixir implementation: ``corpus/specs/*.json``
is read by both, and both must produce identical PTX from it. Key order is fixed
so the files diff cleanly.

The encoder drops keys whose value is ``None``, which is what makes the codec a
fixed point: stringifying a missing ``space`` as ``""`` would decode back as an
atom rather than as absent.
"""

from __future__ import annotations

import json

from gpark import ir


def encode(k: dict, pretty: bool = True) -> str:
    """Encode a kernel to JSON text, with stable key order."""
    return json.dumps(_to_map(k), indent=2 if pretty else None, sort_keys=False) + "\n"


def decode(text: str) -> dict:
    """Decode a kernel from JSON text."""
    return _from_map(json.loads(text))


def _to_map(k: dict) -> dict:
    return {
        "name": k["name"],
        "target": k["target"],
        "ptx_version": k["ptx_version"],
        "shared": k["shared"],
        "maxntid": k["maxntid"],
        "params": [_to_param(p) for p in k["params"]],
        "blocks": [_to_block(b) for b in k["blocks"]],
    }


def _to_param(p: dict) -> dict:
    return {
        "name": p["name"],
        "type": p["type"],
        "space": p["space"],
    }


def _to_block(b: dict) -> dict:
    return {
        "label": b["label"],
        "instrs": [_to_instr(i) for i in b["instrs"]],
        "term": _to_instr(b["term"]) if b["term"] else None,
    }


def _to_instr(i: dict) -> dict:
    out = {}
    for key in ir.INSTR_KEYS:
        value = i[key]
        if key in ("dest", "pred"):
            out[key] = _to_operand(value) if value is not None else None
        elif key == "ops":
            out[key] = [_to_operand(o) for o in value]
        else:
            out[key] = value
    return out


def _to_operand(o) -> list:
    tag = o[0]
    if tag == "reg":
        return ["reg", o[1], o[2]]
    if tag == "pred":
        return ["pred", o[1]]
    if tag == "imm":
        return ["imm", o[1]]
    if tag == "immf":
        return ["immf", o[1], o[2]]
    if tag == "param":
        return ["param", o[1]]
    if tag == "sreg":
        return ["sreg", o[1]]
    if tag == "label":
        return ["label", o[1]]
    if tag == "addr":
        return ["addr", _to_operand(o[1]), _to_operand(o[2]), o[3]]
    raise ValueError(f"unknown operand tag {tag!r}")


def _from_map(m: dict) -> dict:
    return ir.kernel(
        m["name"],
        target=m["target"],
        ptx_version=m["ptx_version"],
        shared=m["shared"] or 0,
        maxntid=m["maxntid"],
        params=[_from_param(p) for p in m["params"] or []],
        blocks=[_from_block(b) for b in m["blocks"]],
    )


def _from_param(p: dict) -> dict:
    return ir.param_decl(p["name"], p["type"], p["space"])


def _from_block(b: dict) -> dict:
    return ir.block(
        b["label"],
        [_from_instr(i) for i in b["instrs"] or []],
        _from_instr(b["term"]) if b["term"] else None,
    )


def _from_instr(m: dict) -> dict:
    return ir.instr(
        m["base"],
        space=m["space"],
        modifier=m["modifier"],
        vec=m["vec"],
        dtype=m["dtype"],
        dest=_from_operand(m["dest"]) if m["dest"] else None,
        ops=[_from_operand(o) for o in m["ops"] or []],
        pred=_from_operand(m["pred"]) if m["pred"] else None,
    )


def _from_operand(o: list) -> tuple:
    tag = o[0]
    if tag == "reg":
        return ir.reg(o[1], o[2])
    if tag == "pred":
        return ir.pred(o[1])
    if tag == "imm":
        return ir.imm(o[1])
    if tag == "immf":
        return ir.immf(o[1], o[2])
    if tag == "param":
        return ir.param(o[1])
    if tag == "sreg":
        return ir.sreg(o[1])
    if tag == "label":
        return ir.label(o[1])
    if tag == "addr":
        return ir.addr(_from_operand(o[1]), _from_operand(o[2]), o[3])
    raise ValueError(f"unknown operand tag {tag!r}")
