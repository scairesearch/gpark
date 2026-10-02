ExUnit.start()

# One shared oracle for every test: both implementations must agree on the exact
# PTX text for the same kernel. Byte-identical output is the contract that keeps
# the Elixir and Python ports honest as they diverge.
defmodule Gpark.Golden do
  @corpus Path.expand("../../corpus", __DIR__)

  def corpus_dir, do: @corpus

  @doc "The `corpus/golden` directory."
  def dir, do: Path.join(@corpus, "golden")

  @doc "Paths to all golden `.ptx` files."
  def contents do
    dir() |> Path.join("*.ptx") |> Path.wildcard() |> Enum.sort()
  end

  @doc "All golden `.ptx` files, as `{kernel_name, ptx_text}` pairs."
  def all do
    @corpus
    |> Path.join("golden/*.ptx")
    |> Path.wildcard()
    |> Enum.map(fn path ->
      {path |> Path.basename(".ptx"), File.read!(path)}
    end)
  end

  @doc "Load a kernel IR spec from `corpus/specs`."
  def spec(name) do
    path = Path.join(@corpus, "specs/#{name}.json")

    unless File.exists?(path) do
      raise "no such spec: #{path}"
    end

    path |> File.read!() |> Gpark.IR.JSON.decode!()
  end
end
