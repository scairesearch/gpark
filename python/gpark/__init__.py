"""gpark — hand-written PTX kernels from Elixir and Python.

This is the Python half of gpark. It is an independent implementation of the
same IR and the same emitter, not a binding to the Elixir one, and the two are
held to a single contract: for the same kernel spec in ``corpus/specs/``, both
must emit the *same bytes* of PTX.

That contract is stricter than it sounds. It is what makes it possible to diff a
gpark golden against what ``nvcc`` produces for an equivalent kernel — if the
formatting were merely similar, every such diff would be noise.

Nothing here needs a GPU. See ``docs/ROADMAP.md`` for status.
"""

from gpark.backend import emit_or_raise, require, required_ops, required_types
from gpark.ir import block, imm, instr, kernel, label, param, param_decl, pred, reg, sreg
from gpark.json_codec import decode as decode_spec
from gpark.json_codec import encode as encode_spec
from gpark.ops import fetch as op_spec
from gpark.ptx import emit
from gpark.type import Packed, all_types, native, reg_class, width
from gpark.validate import check

__all__ = [
    "Packed",
    "all_types",
    "block",
    "check",
    "decode_spec",
    "emit",
    "imm",
    "instr",
    "kernel",
    "label",
    "native",
    "op_spec",
    "param",
    "param_decl",
    "pred",
    "reg",
    "reg_class",
    "sreg",
    "width",
]

__version__ = "0.1.0.dev0"
