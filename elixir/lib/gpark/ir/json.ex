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
    preload_vocabulary()

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
    IR.kernel(name(map["name"]),
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
    IR.param_decl(
      name(name),
      vocab(type),
      p["space"] && vocab(p["space"])
    )
  end

  defp to_param(p) do
    %{
      "name" => to_string(p.name),
      "type" => to_string(p.type),
      # nil must stay nil: `to_string(nil)` is "", which decodes back as an atom
      # rather than as "absent", so the codec would stop being a fixed point.
      "space" => p.space && to_string(p.space)
    }
  end

  defp from_block(%{"label" => label} = b) do
    IR.block(
      name(label),
      Enum.map(b["instrs"] || [], &from_instr/1),
      b["term"] && from_instr(b["term"])
    )
  end

  defp to_block(b) do
    %{
      "label" => to_string(b.label),
      "instrs" => Enum.map(b.instrs, &to_instr/1),
      "term" => b.term && to_instr(b.term)
    }
  end

  # ---------------------------------------------------------------------------
  # Instructions
  # ---------------------------------------------------------------------------

  @keys ~w(base space modifier vec dtype srctype dest ops pred)

  # Atom handling is split deliberately, because the two kinds of name have
  # opposite safety properties:
  #
  #   * vocab/1 uses String.to_existing_atom/1. Type names, address spaces and
  #     special registers come from a closed set, so refusing an unknown value
  #     turns a typo into an error instead of silently inventing a type. That
  #     only works if the vocabularies have been loaded, which is why
  #     decode!/1 preloads them: decoding a spec before ever calling the emitter
  #     used to crash with "not an already existing atom".
  #
  #   * name/1 uses String.to_atom/1. Kernel, parameter and block names are
  #     open-ended, so they cannot come from a fixed set. Corpus files are trusted
  #     project data, not user input.
  defp preload_vocabulary do
    _ = Gpark.Type.all()
    _ = Gpark.Ops.names()
    _ = IR.sreg_names()
    :ok
  end

  defp vocab(value) when is_binary(value), do: String.to_existing_atom(value)
  defp name(value) when is_binary(value), do: String.to_atom(value)

  defp from_instr(%{"base" => base} = m) do
    IR.instr(base,
      space: m["space"] && vocab(m["space"]),
      modifier: m["modifier"],
      vec: m["vec"],
      dtype: m["dtype"] && vocab(m["dtype"]),
      srctype: m["srctype"] && vocab(m["srctype"]),
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
         "srctype" -> i.srctype && to_string(i.srctype)
         "dest" -> i.dest && to_operand(i.dest)
         "ops" -> Enum.map(i.ops, &to_operand/1)
         "pred" -> i.pred && to_operand(i.pred)
       end}
    end)
  end

  # ---------------------------------------------------------------------------
  # Operands, as tagged tuples
  # ---------------------------------------------------------------------------

  defp from_operand(["reg", type, id]), do: IR.reg(vocab(type), id)
  defp from_operand(["pred", id]), do: IR.pred(id)
  defp from_operand(["imm", value]), do: IR.imm(value)
  defp from_operand(["immf", type, value]), do: IR.immf(vocab(type), value)
  defp from_operand(["param", name]), do: IR.param(name(name))
  defp from_operand(["sreg", name]), do: IR.sreg(vocab(name))
  defp from_operand(["label", name]), do: IR.label(name(name))

  defp from_operand(["addr", base, idx, scale]),
    do: IR.addr(from_operand(base), from_operand(idx), scale)

  defp from_operand(value) when is_integer(value), do: {:imm, value}

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

  # Tolerate bare integers so a hand-written spec stays decodable even though
  # IR.addr/3 always normalises them into `imm` first.
  defp to_operand(value) when is_integer(value), do: ["imm", value]
end
