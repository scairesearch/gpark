"""GPU kernel package for Elixir and Python.

gpark emits PTX directly, in the same spirit as writing CUDA C but from a
language that already has the parts you need: pattern matching, exhaustive
case, and a real compiler. It also aims to replace the two libraries that
currently dominate this space — Triton for kernel authoring and CuPy for array
kernels — but by keeping a path down to raw PTX rather than hiding one.

Status: see ``docs/ROADMAP.md``. The Elixir backend compiles, validates and
emits PTX; no kernel has a measured speedup yet, because no GPU has been
attached to the project.
"""

defmodule Gpark do
  @moduledoc """
  Entry points and project status.
  """

  @doc """
  Compile a kernel IR to PTX text, validating it first.

  Returns `{:ok, ptx}` or `{:error, issues}` — never emits PTX it believes to be
  broken, because the whole value of a direct-PTX toolchain is that the output is
  something you can reason about.
  """
  def compile(kernel) do
    with {:ok, kernel} <- Gpark.Validate.check(kernel) do
      {:ok, Gpark.PTX.emit(kernel)}
    end
  end

  @doc """
  Compile a kernel and write the PTX to `path`.
  """
  def compile!(kernel, path) do
    case compile(kernel) do
      {:ok, ptx} ->
        File.write!(path, ptx)
        ptx

      {:error, issues} ->
        messages = Enum.map_join(issues, "\n", &"  #{&1.kind}: #{&1.message}")
        raise ArgumentError, "refusing to emit invalid PTX:\n#{messages}"
    end
  end

  @doc "Every type gpark understands, for tests and tooling."
  def types, do: Gpark.Type.all()

  @doc "Every opcode in the typed table."
  def ops, do: Gpark.Ops.names()
end
