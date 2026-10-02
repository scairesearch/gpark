# Regenerate corpus/specs/*.json and corpus/golden/*.ptx from the Elixir kernels.
#
# The Elixir implementation is the source of truth for both files: it owns the
# kernels, and the Python implementation must then reproduce the goldens byte for
# byte. Keeping this in a script rather than inline in a Makefile avoids quoting
# an Elixir program through /bin/sh.
alias Gpark.Kernels.{ReduceSumF32, SaxpyF32, UnpackU4F32, VecAddF32}

kernels = [VecAddF32, SaxpyF32, ReduceSumF32, UnpackU4F32]

for module <- kernels do
  kernel = module.build()

  case Gpark.Validate.check(kernel) do
    {:ok, _} ->
      spec = Path.join(["..", "corpus", "specs", "#{kernel.name}.json"])
      golden = Path.join(["..", "corpus", "golden", "#{kernel.name}.ptx"])
      File.write!(spec, Gpark.IR.JSON.encode!(kernel))
      File.write!(golden, Gpark.PTX.emit(kernel))
      IO.puts("  #{kernel.name}")

    {:error, issues} ->
      messages = Enum.map_join(issues, "\n", &"    #{&1.kind}: #{&1.message}")
      raise "#{inspect(module)} does not validate:\n#{messages}"
  end
end
