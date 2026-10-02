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
# s2/u2/s4/u4 have no direct PTX spelling, so they are deliberately absent from
# _NATIVE -- that absence is what stops the backend believing `u4` is arithmetic.
# They still carry (bits, kind, sign), because a packed element type that cannot
# answer width, kind or sign is not much of a type.
_SUB_BYTE: dict[str, tuple[int, str, str]] = {
    "s2": (2, "int", "signed"),
    "u2": (2, "int", "unsigned"),
    "s4": (4, "int", "signed"),
    "u4": (4, "int", "unsigned"),
}

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
    entry = _SUB_BYTE.get(type_name) or (
        _SUB_BYTE.get(type_name.elem) if isinstance(type_name, Packed) else None
    )
    return entry[0] if entry else None


def describe(type_name: object) -> tuple[int, str, str] | None:
    """The ``(bits, kind, sign)`` description of a type, or None.

    Checks ``_NATIVE`` first, then ``_SUB_BYTE``. Funnelling ``width``, ``kind`` and
    ``sign`` through one lookup means they cannot disagree about what a type is --
    each used to consult ``_NATIVE`` alone, so every sub-byte type answered None
    from all three and ``s4`` was indistinguishable from a typo.
    """
    if not isinstance(type_name, str):
        return None
    return _NATIVE.get(type_name) or _SUB_BYTE.get(type_name)


def width(type_name: object) -> int | None:
    """Width in bits. None for packed values, which have no single width."""
    if isinstance(type_name, Packed):
        return None
    entry = describe(type_name)
    return entry[0] if entry else None


def kind(type_name: object) -> str | None:
    """One of ``int``, ``bit``, ``float``, ``pred``."""
    entry = describe(type_name)
    return entry[1] if entry else None


def sign(type_name: object) -> str | None:
    """``"signed"``, ``"unsigned"``, or None."""
    entry = describe(type_name)
    return entry[2] if entry else None


def is_signed(type_name: object) -> bool:
    """True for types with a sign *bit*. Note that floats qualify too; see
    :func:`signed_int` for the integer-only question."""
    return sign(type_name) == "signed"


def signed_int(type_name: object) -> bool:
    """True for signed *integers*, including packed sub-byte ones.

    Deliberately not ``is_signed``: ``f32`` has a sign bit but is not a signed
    integer, and conflating the two makes a float compare pick a signed modifier.
    """
    entry = describe(type_name)
    if entry is None and isinstance(type_name, Packed):
        entry = describe(type_name.elem)
    return entry is not None and entry[1] == "int" and entry[2] == "signed"


def unsigned_int(type_name: object) -> bool:
    """True for unsigned *integers*, including packed sub-byte ones."""
    entry = describe(type_name)
    if entry is None and isinstance(type_name, Packed):
        entry = describe(type_name.elem)
    return entry is not None and entry[1] == "int" and entry[2] == "unsigned"


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
