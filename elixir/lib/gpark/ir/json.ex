defmodule Gpark.IR.JSON do
  @moduledoc """
  JSON codec for `Gpark.IR`.

  This is what makes `corpus/` possible. Kernel specs are stored as JSON rather
  than as Elixir terms so that the independent Python implementation can read the
  exact same file and must produce byte-identical PTX. Without that, "two
  implementations agreeing" degrades into two implementations that each look
  right on their own test suite.

  Encoding rules are deliberately minimal and total, so both implementations can
  agree without sharing code:

    * every instruction is a map with the keys `base`, `space`, `modifier`,
      `vec`, `dtype`, `dest`, `ops`, `pred`
    * operands are tagged tuples, encoded as arrays
    * atoms are encoded as strings
  """

  alias Gpark.IR

  @doc "Decode a kernel from JSON text."
  def decode!(json) do
    json
    |> Jason.decode!()
    |> from_map()
  end

  @doc "Encode a kernel to JSON text, with stable key order."
  def encode!(kernel, pretty \\ true) do
    to_map(kernel)
    |> Jason.encode!(pretty: pretty)
  end

  # ---------------------------------------------------------------------------
  # Kernel
  # ---------------------------------------------------------------------------

  defp from_map(%{"blocks" => blocks} = map) do
    IR.kernel(binary_to_atom(map["name"]),
      target: map["target"],
      ptx_version: map["ptx_version"],
      shared: map["shared"] || 0,
      maxntid: map["maxntid"],
      params: Enum.map(map["params"] || [], &from_param/1),
      blocks: Enum.map(blocks, &from_block/1)
    )
  end

  defp to_map(kernel) do
    %{
      "name" => to_string(kernel.name),
      "target" => kernel.target,
      "ptx_version" => kernel.ptx_version,
      "shared" => kernel.shared,
      "maxntid" => kernel.maxntid,
      "params" => Enum.map(kernel.params, &to_param/1),
      "blocks" => Enum.map(kernel.blocks, &to_block/1)
    }
  end

  defp from_param(%{"name" => name, "type" => type} = p) do
    IR.param_decl(binary_to_atom(name), binary_to_atom(type),
      p["space"] && binary_to_atom(p["space"]))
  end

  defp to_param(p) do
    %{"name" => to_string(p.name), "type" => to_string(p.type), "space" => to_string(p.space)}
  end

  defp from_block(%{"label" => label} = b) do
    IR.block(binary_to_atom(label),
      Enum.map(b["instrs"] || [], &from_instr/1),
      b["term"] && from_instr(b["term"]))
  end

  defp to_block(b) do
    %{"label" => to_string(b.label), "instrs" => Enum.map(b.instrs, &to_instr/1),
      "term" => b.term && to_instr(b.term)}
  end

  # ---------------------------------------------------------------------------
  # Instructions
  # ---------------------------------------------------------------------------

  @keys ~w(base space modifier vec dtype dest ops pred)

  # Corpus files are trusted input (they live in this repo), and the atom set is
  # closed, so interning is safe here and keeps the codec allocation-free.
  defp binary_to_atom(value) when is_binary(value), do: String.to_existing_atom(value)

  defp from_instr(%{"base" => base} = m) do
    IR.instr(base,
      space: m["space"] && binary_to_atom(m["space"]),
      modifier: m["modifier"],
      vec: m["vec"],
      dtype: m["dtype"] && binary_to_atom(m["dtype"]),
      dest: m["dest"] && from_operand(m["dest"]),
      ops: Enum.map(m["ops"] || [], &from_operand/1),
      pred: m["pred"] && from_operand(m["pred"])
    )
  end

  defp to_instr(i) do
    Map.new(@keys, fn key ->
      {key,
       case key do
         "space" -> i.space && to_string(i.space)
         "base" -> i.base
         "modifier" -> i.modifier
         "vec" -> i.vec
         "dtype" -> i.dtype && to_string(i.dtype)
         "dest" -> i.dest && to_operand(i.dest)
         "ops" -> Enum.map(i.ops, &to_operand/1)
         "pred" -> i.pred && to_operand(i.pred)
       end}
    end)
  end

  # ---------------------------------------------------------------------------
  # Operands, as tagged tuples
  # ---------------------------------------------------------------------------

  defp from_operand(["reg", type, id]), do: IR.reg(binary_to_atom(type), id)
  defp from_operand(["pred", id]), do: IR.pred(id)
  defp from_operand(["imm", value]), do: IR.imm(value)
  defp from_operand(["immf", type, value]), do: IR.immf(binary_to_atom(type), value)
  defp from_operand(["param", name]), do: IR.param(binary_to_atom(name))
  defp from_operand(["sreg", name]), do: IR.sreg(binary_to_atom(name))
  defp from_operand(["label", name]), do: IR.label(binary_to_atom(name))

  defp from_operand(["addr", base, idx, scale]),
    do: IR.addr(from_operand(base), from_operand(idx), scale)

  defp from_operand(other), do: raise(ArgumentError, "unknown operand #{inspect(other)}")

  defp to_operand({:reg, type, id}), do: ["reg", to_string(type), id]
  defp to_operand({:pred, id}), do: ["pred", id]
  defp to_operand({:imm, value}), do: ["imm", value]
  defp to_operand({:immf, type, value}), do: ["immf", to_string(type), value]
  defp to_operand({:param, name}), do: ["param", to_string(name)]
  defp to_operand({:sreg, name}), do: ["sreg", to_string(name)]
  defp to_operand({:label, name}), do: ["label", to_string(name)]
  defp to_operand({:addr, base, idx, scale}),
    do: ["addr", to_operand(base), to_operand(idx), scale]
end
