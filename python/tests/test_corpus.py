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

from gpark import decode_spec, emit, encode_spec, ops
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
