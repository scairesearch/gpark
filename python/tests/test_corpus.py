"""The corpus contract, enforced from the Python side.

This is the whole point of having two implementations. If the Elixir and Python
emitters disagree on even one byte the codebases start drifting, and the shared
corpus stops being evidence of anything. So these tests read the same files the
Elixir suite reads and require identical output.
"""

from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path

from gpark import decode_spec, emit, encode_spec, ir, ops
from gpark import backend, ptx as ptx_mod
from gpark import type as type_mod
from gpark.validate import check

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "corpus"
SPECS = sorted((CORPUS / "specs").glob("*.json"))
GOLDENS = sorted((CORPUS / "golden").glob("*.ptx"))

_ELIXIR_DIGEST_SCRIPT = """
Gpark.Ops.table()
|> Enum.sort()
|> Enum.map(fn {name, s} ->
  otypes =
    case s.otypes do
      :same -> "same"
      :any -> "any"
      :sreg -> "sreg"
      other when is_list(other) -> other |> Enum.map(&to_string/1) |> Enum.sort() |> Enum.join(",")
    end

  optional = fn
    nil -> "-"
    list ->
      list
      |> Enum.map(fn v -> if v, do: to_string(v), else: "-" end)
      |> Enum.sort()
      |> Enum.join(",")
  end

  Enum.join([
    name,
    # `parts` order is semantic (it is PTX's dotted-part order). The rest are
    # sets, so they are sorted: the two implementations may list the same types in
    # a different order without that counting as drift.
    Enum.map_join(s.parts, ",", &to_string/1),
    s.dtype |> Enum.map(&to_string/1) |> Enum.sort() |> Enum.join(","),
    to_string(s.ndest),
    to_string(s.nops),
    otypes,
    optional.(s.spaces),
    optional.(s.modifiers),
    optional.(s.srcs),
    to_string(s.sync)
  ], "|")
end)
|> Enum.join("\n")
|> IO.puts()
"""


class TestCorpus(unittest.TestCase):
    def test_corpus_is_not_empty(self):
        # A corpus-driven suite that silently finds no files still passes, so
        # assert the population explicitly.
        self.assertTrue(SPECS, f"no specs found under {CORPUS}")
        self.assertEqual(
            [p.stem for p in SPECS],
            [p.stem for p in GOLDENS],
            "every spec needs a golden and vice versa",
        )

    def test_each_spec_validates_and_reproduces_its_golden_byte_for_byte(self):
        for spec_path in SPECS:
            name = spec_path.stem
            with self.subTest(kernel=name):
                kernel = decode_spec(spec_path.read_text())

                ok, issues = check(kernel)
                self.assertIsNotNone(
                    ok,
                    f"{name} does not validate: "
                    + "; ".join(f"{i.kind}: {i.message}" for i in issues),
                )

                expected = (CORPUS / "golden" / f"{name}.ptx").read_text()
                self.assertEqual(
                    emit(kernel),
                    expected,
                    f"{name}: Python PTX differs from the golden the Elixir side produced",
                )

    def test_codec_is_a_fixed_point(self):
        for spec_path in SPECS:
            name = spec_path.stem
            with self.subTest(kernel=name):
                kernel = decode_spec(spec_path.read_text())
                self.assertEqual(
                    decode_spec(encode_spec(kernel)),
                    kernel,
                    f"{name}: encode/decode is lossy",
                )

    def test_ops_table_has_not_drifted_from_elixir(self):
        # The two implementations define the opcode table separately, on purpose.
        # Comparing canonical digests means a typo in either one fails here
        # instead of surfacing later as ptxas output nobody can explain.
        elixir = _elixir_digest()
        if elixir is None:
            self.skipTest("Elixir toolchain unavailable; cannot cross-check the ops table")
        self.assertEqual(ops.digest(), elixir.strip(), "opcode tables have drifted")

    def test_type_sets_have_not_drifted_from_elixir(self):
        # The digest above covers opcodes but not types, which is how `s2 u2 s4 u4`
        # could sit in the Elixir tables, be used by its doctests, and be absent from
        # the introspection functions -- a type-level drift the opcode digest is blind
        # to by construction.
        elixir = _elixir_types()
        if elixir is None:
            self.skipTest("Elixir toolchain unavailable; cannot cross-check types")
        self.assertEqual(type_mod.all_types(), elixir, "type sets have drifted")

    def test_subbyte_types_are_introspectable(self):
        # The specific failure this guards: `width`, `kind` and `sign` each consulted
        # only the native table, so every sub-byte type answered nil and `s4` was
        # indistinguishable from a typo.
        for name, sign in (("s2", "signed"), ("u2", "unsigned"),
                           ("s4", "signed"), ("u4", "unsigned")):
            self.assertEqual(type_mod.width(name), int(name[1:]))
            self.assertEqual(type_mod.kind(name), "int")
            self.assertEqual(type_mod.sign(name), sign)

        # No direct PTX spelling -- that absence is the whole reason the type exists.
        self.assertFalse(type_mod.native("u4"))
        self.assertTrue(type_mod.signed_int("s4"))
        self.assertTrue(type_mod.unsigned_int("u4"))
        self.assertFalse(type_mod.signed_int("f32"),
                         "f32 has a sign bit but is not a signed integer")


def _elixir_types() -> list[str] | None:
    """Type names as Elixir sees them, for the cross-implementation parity check."""
    mix = shutil.which("mix")
    elixir_dir = REPO / "elixir"
    if mix is None or not elixir_dir.is_dir():
        return None
    script = 'IO.puts(Enum.join(Gpark.Type.all(), ","))'
    try:
        done = subprocess.run(
            [mix, "run", "--no-start", "-e", script],
            cwd=elixir_dir, capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return sorted(f for f in done.stdout.strip().split(",") if f)


def _elixir_digest() -> str | None:
    mix = shutil.which("mix")
    elixir_dir = REPO / "elixir"
    if mix is None or not elixir_dir.is_dir():
        return None
    try:
        done = subprocess.run(
            [mix, "run", "--no-start", "-e", _ELIXIR_DIGEST_SCRIPT],
            cwd=elixir_dir, capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


if __name__ == "__main__":
    unittest.main()


class TestBackendGate(unittest.TestCase):
    """The capability gate: refuse what cannot be emitted, and say why.

    A gate that accepts everything is indistinguishable from no gate, so the bulk of
    this builds kernels the PTX backend genuinely cannot handle and checks that each is
    turned away with a specific reason.
    """

    def test_declared_capabilities_match_the_tables(self):
        self.assertEqual(ptx_mod.ops(), ops.names())
        self.assertEqual(ptx_mod.types(), type_mod.all_types())
        self.assertEqual(len(ptx_mod.ops()), 48)
        self.assertEqual(len(ptx_mod.types()), 30)

    def test_required_ops_reads_terminators_too(self):
        # A body-blind implementation would see no ops at all for a ret-only kernel.
        k = ir.kernel("t", blocks=[ir.block("entry", [], ir.instr("ret"))])
        self.assertEqual(backend.required_ops(k), {"ret"})

    def test_required_types_ignores_untagged_instructions(self):
        # `ret` and `bra` carry no dtype. Collecting that None would report every
        # terminator as an unsupported type named None.
        k = ir.kernel("t", blocks=[ir.block("entry", [], ir.instr("ret"))])
        self.assertNotIn(None, backend.required_types(k))

    def test_passes_every_corpus_kernel(self):
        for path in SPECS:
            k = decode_spec(path.read_text())
            checked, issues = backend.require(ptx_mod, k)
            self.assertEqual(issues, [], f"{path.name}: {issues}")
            self.assertEqual(checked, k, "the gate must not transform the kernel")

    def test_refuses_an_unsupported_opcode(self):
        k = ir.kernel("t", blocks=[
            ir.block("entry", [ir.instr("tensor::mma", dtype="f32")], ir.instr("ret"))
        ])
        _checked, issues = backend.require(ptx_mod, k)
        self.assertEqual(issues, [("unsupported_op", "tensor::mma")])

    def test_refuses_an_unsupported_type(self):
        k = ir.kernel("t", blocks=[
            ir.block("entry",
                     [ir.instr("add", dtype="u128", dest=ir.reg("u128", 1),
                               ops=[ir.reg("u128", 1), ir.imm(1)])],
                     ir.instr("ret"))
        ])
        _checked, issues = backend.require(ptx_mod, k)
        self.assertEqual(issues, [("unsupported_type", "u128")])

    def test_reports_every_missing_capability_at_once(self):
        k = ir.kernel("t", blocks=[
            ir.block("entry",
                     [ir.instr("tensor::mma", dtype="f32"),
                      ir.instr("cvt", dtype="u128", dest=ir.reg("u128", 1),
                               ops=[ir.reg("f32", 1)])],
                     ir.instr("ret"))
        ])
        _checked, issues = backend.require(ptx_mod, k)
        self.assertIn(("unsupported_op", "tensor::mma"), issues)
        self.assertIn(("unsupported_type", "u128"), issues)

    def test_emit_or_raise_names_the_missing_capability(self):
        k = ir.kernel("t", blocks=[
            ir.block("entry", [ir.instr("tensor::mma", dtype="f32")], ir.instr("ret"))
        ])
        with self.assertRaises(ValueError) as ctx:
            backend.emit_or_raise(ptx_mod, k)
        message = str(ctx.exception)
        self.assertIn("tensor::mma", message)
        self.assertIn("slower sequence", message)

    def test_subbyte_types_are_declared_supportable(self):
        # No direct PTX spelling, so `ptx_type` is None -- but they are representable,
        # and the gate must not refuse a kernel on that technicality.
        for name in ("s2", "u2", "s4", "u4"):
            self.assertTrue(backend.supports_type(ptx_mod, name))
            self.assertFalse(type_mod.native(name))

    def test_require_and_check_answer_different_questions(self):
        # A malformed kernel using nothing exotic is still within this backend's
        # capabilities, so conflating the two would hide the real error.
        k = ir.kernel("broken", blocks=[
            ir.block("entry",
                     [ir.instr("st", space="global", dtype="f32",
                               ops=[ir.addr(ir.reg("u64", 1))])],
                     ir.instr("ret"))
        ])
        checked, issues = backend.require(ptx_mod, k)
        self.assertEqual(issues, [])
        self.assertIsNotNone(checked)
        _validated, problems = ptx_mod.check(k)
        self.assertTrue(problems, "structural problems expected")
