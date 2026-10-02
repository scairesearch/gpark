"""The gpark type system: a container width, and an element format.

Sub-byte types are the reason this is two levels rather than one flat list. A
4-bit signed integer has *no* PTX spelling: PTX's narrowest integer register is
16 bits, and ``.b4`` exists only as a storage format. So ``s4`` is modelled as
"four-bit signed elements packed into a 16-bit container", and arithmetic happens
after widening.

Keeping those two facts separate is what lets ``Gpark.Quant`` unpack a packed
u4 buffer into f32 without ever inventing an opcode ptxas will not accept. The
nearest existing project, Taichi, has no bf16, no fp8 and no sub-byte primitives
at all, only an optional ``quant`` extension that does not lower to hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

# type name -> (width in bits, kind, sign)
_NATIVE: dict[str, tuple[int, str, str]] = {
    # integers
    "s8": (8, "int", "signed"),
    "u8": (8, "int", "unsigned"),
    "s16": (16, "int", "signed"),
    "u16": (16, "int", "unsigned"),
    "s32": (32, "int", "signed"),
    "u32": (32, "int", "unsigned"),
    "s64": (64, "int", "signed"),
    "u64": (64, "int", "unsigned"),
    # raw bit containers
    "b1": (1, "bit", "unsigned"),
    "b2": (2, "bit", "unsigned"),
    "b4": (4, "bit", "unsigned"),
    "b8": (8, "bit", "unsigned"),
    "b16": (16, "bit", "unsigned"),
    "b32": (32, "bit", "unsigned"),
    "b64": (64, "bit", "unsigned"),
    # floats
    "f16": (16, "float", "signed"),
    "bf16": (16, "float", "signed"),
    "f32": (32, "float", "signed"),
    "f64": (64, "float", "signed"),
    # fp8
    "e4m3": (8, "float", "signed"),
    "e5m2": (8, "float", "signed"),
    # fp6 / fp4
    "e2m3": (6, "float", "signed"),
    "e3m2": (6, "float", "signed"),
    "e2m1": (4, "float", "signed"),
    "e8m0": (8, "float", "unsigned"),
    # predicate: 32 bits wide, but not an integer
    "pred": (32, "pred", "unsigned"),
}

# s2/u2/s4/u4 are not PTX types at all; their width is implied by the name.
_SUB_BYTE: dict[str, int] = {"s2": 2, "u2": 2, "s4": 4, "u4": 4}

# The native type a sub-byte element widens to before arithmetic. Widening to the
# smallest type that holds every value avoids a redundant shift afterwards, and
# keeps s4/u4 in a single 16-bit bank.
_WIDEN: dict[str, str] = {
    "s2": "s16",
    "u2": "u16",
    "s4": "s16",
    "u4": "u16",
    "e2m1": "f32",
    "e4m3": "f32",
    "e5m2": "f32",
    "e2m3": "f32",
    "e3m2": "f32",
    "e8m0": "f32",
}


@dataclass(frozen=True)
class Packed:
    """Elements narrower than the container they are stored in.

    ``Packed("b16", "s4", 4)`` is four signed 4-bit elements in one 16-bit
    register. There is no PTX arithmetic for this; it lowers to shifts and masks,
    which is exactly why the type has to say so rather than pretend otherwise.
    """

    container: str
    elem: str
    count: int

    def __post_init__(self) -> None:
        if not native(self.container):
            raise ValueError(f"packed container must be native, got {self.container!r}")
        if sub_byte_width(self.elem) is None:
            raise ValueError(f"packed element must be sub-byte, got {self.elem!r}")
        if width(self.container) != sub_byte_width(self.elem) * self.count:
            raise ValueError(
                f"{self.count} x {self.elem!r} does not fill a {self.container!r}"
            )


def native(type_name: object) -> bool:
    """True when the type has a direct PTX spelling."""
    return isinstance(type_name, str) and type_name in _NATIVE


def sub_byte_width(type_name: object) -> int | None:
    """Element width of a sub-byte type, or None if it is not one."""
    if isinstance(type_name, str):
        return _SUB_BYTE.get(type_name)
    if isinstance(type_name, Packed):
        return _SUB_BYTE.get(type_name.elem)
    return None


def width(type_name: object) -> int | None:
    """Storage width in bits. None for packed types, which have no single width."""
    if isinstance(type_name, Packed):
        return None
    entry = _NATIVE.get(type_name) if isinstance(type_name, str) else None
    return entry[0] if entry else None


def kind(type_name: object) -> str | None:
    """One of ``int``, ``bit``, ``float``, ``pred``."""
    entry = _NATIVE.get(type_name) if isinstance(type_name, str) else None
    return entry[1] if entry else None


def is_signed(type_name: object) -> bool:
    """True for types with a sign bit. Unlike integers, ``.u32`` is also a float."""
    entry = _NATIVE.get(type_name) if isinstance(type_name, str) else None
    return bool(entry and entry[2] == "signed")


def widen(type_name: str | Packed) -> str:
    """The native type a sub-byte element widens to for arithmetic."""
    elem = type_name.elem if isinstance(type_name, Packed) else type_name
    return _WIDEN.get(elem, "u32")


def native_names() -> list[str]:
    return sorted(_NATIVE)


def all_types() -> list[str]:
    """Every type gpark knows about, for exhaustive tests and golden generation."""
    return sorted(native_names() + list(_SUB_BYTE))


def reg_class(type_name: object) -> str | None:
    """PTX register bank for a type, e.g. ``rd`` for ``u64``.

    Note the banks are *not* the storage widths: PTX keeps fp8 and fp4 values in
    16-bit registers and converts to and from them, so an fp8 value lives in the
    32-bit ``r`` bank via a ``b16`` view.
    """
    if isinstance(type_name, Packed):
        return reg_class(type_name.container)
    if not isinstance(type_name, str):
        return None
    if type_name in ("s64", "u64", "b64"):
        return "rd"
    if type_name in ("f32", "f16", "bf16"):
        return "f"
    if type_name == "f64":
        return "fd"
    if type_name == "pred":
        return "p"
    return "r" if native(type_name) else None
